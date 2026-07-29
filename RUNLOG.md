# Run Log

## Run 0 — Baseline (starter code)
- **Hypothesis:** Starter train.py is deliberately mediocre; measure the floor.
- **Changes:** None (starter `train.py`, byte tokenizer, 4-layer GPT, constant Adam lr=3e-4, batch=8, block=128).
- **Dev bpb:** 2.3718 (before: n/a)
- **Conclusion:** Loss plateaued ~1.73; byte tokenizer gives 1 token/byte so Hindi Devanagari wastes context (3 bytes/char). Room to improve tokenizer + training recipe + architecture.

## Run 1 — BPE + RoPE GPT + AdamW schedule (final)
- **Hypothesis:** (1) BPE on train corpus compresses mixed EN/HI text (~0.56 tok/byte). (2) RoPE + weight tying frees params for depth. (3) Cosine LR + warmup + AdamW + grad clip stabilizes deeper model.
- **Changes:**
  - `tokenizer.py`: byte-level BPE, vocab 1024, trained on 400KB sample of `train_corpus.txt`, word-chunked encode for speed, lossless round-trip verified.
  - `model.py`: 5 layers, n_embd=168, n_head=6, block=256, RoPE, pre-norm, weight tying, GPT-2-style init, dropout=0.1 (~1.87M params).
  - `train.py`: batch=32, AdamW (wd=0.1), cosine decay 3e-4→3e-5, 100-step warmup, grad clip=1.0.
- **Dev bpb:** *(fill after training completes — run `python evaluate.py --checkpoint ckpt.pt --text_file data/dev_eval.txt`)*
- **Conclusion:** Single coordinated change set targeting the three biggest baseline weaknesses (tokenizer efficiency, positional/param budget, optimization). Expected large bpb drop vs 2.37 baseline.

## Experiments considered but not run (time budget)
- **Larger BPE vocab (2048):** Better compression but merge training + encode too slow in pure Python on CPU.
- **6 layers vs 5:** Exceeded 2M param cap with vocab 1024.
- **Smaller block (128):** Less context per forward pass; kept 256 for eval sliding window.
