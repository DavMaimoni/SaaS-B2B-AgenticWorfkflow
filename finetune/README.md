# LoRA fine-tuning

The LoRA teaches a small open model to do, in one call, what the graph's `triage` node asks:
read a ticket and return `{"queue", "priority", "type", "answer"}` as JSON, in the support tone.
Facts specific to Fluxo come from the MCP tools at run time, not from the weights.

## 1. Build the training files (CPU, a few seconds)

```bash
uv run python scripts/prepare_data.py            # data/processed/{train,val,test}.jsonl
uv run python finetune/build_sft.py              # finetune/data/{train,val}.jsonl
```

The test split is never used here: it is reserved for the 200 evaluation triples.

## 2. Train (GPU)

On Colab (T4 is enough for a 1.5B–3B model in 4 bits; L4/A100 is faster), upload `finetune/` then:

```bash
pip install -U "torch>=2.4" "transformers>=4.46" "peft>=0.13" "bitsandbytes>=0.44" "accelerate>=1.0" datasets
python finetune/train_lora.py --base Qwen/Qwen2.5-3B-Instruct --limit 5000 --epochs 1   # setup check
python finetune/train_lora.py --base Qwen/Qwen2.5-3B-Instruct --epochs 2 --merge          # full run
```

Default hyper-parameters: QLoRA 4-bit NF4, rank 16, alpha 32, dropout 0.05, all attention and MLP projections,
lr 2e-4 cosine, effective batch 16, max length 1024 tokens, best checkpoint on validation loss.
Write the final values and the training time in `MODEL_CARD.md`.

## 3. Export to Ollama (GGUF)

```bash
git clone https://github.com/ggml-org/llama.cpp && pip install -r llama.cpp/requirements.txt
python llama.cpp/convert_hf_to_gguf.py finetune/out/fluxo-lora-merged --outfile fluxo-lora-f16.gguf --outtype f16
# optional, smaller and faster: build llama.cpp, then
# ./llama.cpp/build/bin/llama-quantize fluxo-lora-f16.gguf fluxo-lora-q4_k_m.gguf Q4_K_M
ollama create fluxo-support-lora -f finetune/Modelfile
ollama run fluxo-support-lora   # quick manual check
```

Tool names and flags of llama.cpp change between versions: check its README if a command fails.

## 4. Measure the gain (same 200 triples)

```bash
# base model, prompt only
SMALL_PROVIDER=ollama SMALL_MODEL=qwen2.5:3b uv run python eval/run_eval.py --variant raw_small --out eval/runs/base
# the LoRA
SMALL_PROVIDER=ollama SMALL_MODEL=fluxo-support-lora uv run python eval/run_eval.py --variant raw_small --out eval/runs/lora
uv run python eval/summary.py compare eval/runs/base eval/runs/lora
```

Then the full system (`--variant graph`) and the large model alone: see the main README, "Experiments".
