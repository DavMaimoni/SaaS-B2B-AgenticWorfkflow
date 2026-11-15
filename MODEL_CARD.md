# Model card: fluxo-support-lora

To complete after training. Every number comes from `eval/runs/*/summary.csv`.

| | |
| --- | --- |
| Base model | e.g. Qwen/Qwen2.5-3B-Instruct (licence: ...) |
| Method | QLoRA, 4-bit NF4, rank 16, alpha 32, dropout 0.05, all attention + MLP projections |
| Training data | `finetune/data/train.jsonl` (n = ...), from the train split of Tobi-Bueck tickets, English |
| Epochs / lr / effective batch / max length | ... / 2e-4 / 16 / 1024 |
| Hardware, training time | ... |
| Output format | one JSON object: queue, priority, type, answer |
| Served by | Ollama (`finetune/Modelfile`), GGUF Q4_K_M |

## Intended use

First-line triage and draft replies for routine support tickets, inside the graph: high-priority, sensitive,
low-confidence and invalid outputs are routed to the large model or a human.

## Results on the 200 test triples

| Variant | Triage exact | Queue macro-F1 | Judge score (95 % CI) | Valid JSON | Latency p50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base model, prompt only | | | | | |
| LoRA | | | | | |
| Paired bootstrap, LoRA - base | | | | | |

## Limitations and risks

- Learned the tone and labels of a synthetic, multi-domain dataset; it has no knowledge of Fluxo beyond the
  articles given in its prompt.
- May still invent commitments (refunds, dates): the validator, the routing rules and escalation are the safeguards.
- Non-commercial licence of the training data.

## Rollback

Keep the previous GGUF and Ollama tag (`fluxo-support-lora:vN-1`). Rolling back = `SMALL_MODEL=fluxo-support-lora:vN-1`.
