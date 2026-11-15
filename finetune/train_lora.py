"""QLoRA fine-tuning of a small open model on the support tickets (GPU needed: Colab T4/L4, Kaggle, local CUDA).

    pip install -U "torch>=2.4" "transformers>=4.46" "peft>=0.13" "bitsandbytes>=0.44" "accelerate>=1.0" datasets
    python finetune/train_lora.py --base Qwen/Qwen2.5-3B-Instruct --train finetune/data/train.jsonl \
        --val finetune/data/val.jsonl --out finetune/out/fluxo-lora --epochs 2 --merge

The loss is computed on the assistant turn only (the JSON with labels + reply), not on the prompt.
`--merge` also writes a merged fp16 model to <out>-merged, ready for GGUF conversion (see finetune/README.md).
Training time is roughly linear in examples x tokens: start with --limit 5000 to check the setup.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_examples(path: Path, limit: int | None) -> list[dict]:
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    return rows[:limit] if limit else rows


def tokenize(example: dict, tokenizer, max_len: int) -> dict:
    """input_ids = prompt + answer + eos; labels = -100 on the prompt so only the answer is learned."""
    messages = example["messages"]
    prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    answer = messages[-1]["content"] + tokenizer.eos_token
    p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    a_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    p_ids = p_ids[-(max_len - min(len(a_ids), max_len // 2)):]     # keep the end of very long tickets
    a_ids = a_ids[: max_len - len(p_ids)]
    return {"input_ids": p_ids + a_ids, "attention_mask": [1] * (len(p_ids) + len(a_ids)),
            "labels": [-100] * len(p_ids) + a_ids}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct",
                    help="e.g. Qwen/Qwen2.5-3B-Instruct, meta-llama/Llama-3.2-3B-Instruct (gated), Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--train", type=Path, default=Path("finetune/data/train.jsonl"))
    ap.add_argument("--val", type=Path, default=Path("finetune/data/val.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("finetune/out/fluxo-lora"))
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--limit", type=int, help="use only the first N training examples")
    ap.add_argument("--val-limit", type=int, default=500)
    ap.add_argument("--merge", action="store_true", help="also save a merged fp16 model for GGUF export")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorForSeq2Seq,
                              Trainer, TrainingArguments, set_seed)

    set_seed(args.seed)
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train = Dataset.from_list(load_examples(args.train, args.limit))
    val = Dataset.from_list(load_examples(args.val, args.val_limit))
    cols = train.column_names
    train = train.map(lambda e: tokenize(e, tokenizer, args.max_len), remove_columns=cols)
    val = val.map(lambda e: tokenize(e, tokenizer, args.max_len), remove_columns=cols)
    print(f"train {len(train)} | val {len(val)} | base {args.base}")

    model = AutoModelForCausalLM.from_pretrained(
        args.base, device_map="auto", torch_dtype=dtype,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True))
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=args.alpha, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(args.out), num_train_epochs=args.epochs, learning_rate=args.lr,
            per_device_train_batch_size=args.batch, per_device_eval_batch_size=args.batch,
            gradient_accumulation_steps=args.grad_accum, warmup_ratio=0.03, lr_scheduler_type="cosine",
            logging_steps=20, eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
            load_best_model_at_end=True, metric_for_best_model="eval_loss", bf16=bf16, fp16=not bf16,
            gradient_checkpointing=True, report_to="none", seed=args.seed),
        train_dataset=train, eval_dataset=val,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100))
    trainer.train()
    trainer.save_model(str(args.out))
    tokenizer.save_pretrained(str(args.out))
    (args.out / "training_args.json").write_text(json.dumps(vars(args), default=str, indent=2))
    print(f"adapter saved to {args.out}")

    if args.merge:
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=dtype, device_map="auto")
        merged = PeftModel.from_pretrained(base, str(args.out)).merge_and_unload()
        merged_dir = Path(f"{args.out}-merged")
        merged.save_pretrained(str(merged_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(merged_dir))
        print(f"merged model saved to {merged_dir}")


if __name__ == "__main__":
    main()
