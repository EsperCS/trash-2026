# 🛡️ RMIT Hackathon 2026: Empirical Benchmark (LLMs vs. Safety Encoders)

> **Execution Environment**: Google Colab GPU (Tesla T4 / A100)  
> **Dataset**: `/data/rmit-hackathon-2026/train.csv` (14,000 samples across Portuguese `pt`, French `fr`, Italian `it`, and English `en`).

---

## 📊 1. Head-to-Head Binary Evaluation (`BENIGN` vs `PROMPT_INJECTION`)

*Evaluated on the binary subset (N = 9,660; `JAILBREAK` records excluded).*

| Model Name | Paradigm | Macro-F1 | Accuracy | F1 (Benign) | F1 (Injection) | PT F1 | FR F1 | IT F1 | EN F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `patronus-studio/wolf-defender-prompt-injection` | Safety Encoder (ModernBERT) | **0.7517** | **0.8837** | **0.9328** | **0.5706** | 0.7182 | 0.7854 | 0.7124 | **0.9451** |
| `protectai/deberta-v3-base-prompt-injection-v2` | Safety Encoder (DeBERTa-v3) | **0.6825** | 0.8172 | 0.8893 | 0.4757 | 0.6411 | 0.7390 | 0.6341 | **0.9372** |
| `mistralai/Mistral-7B-Instruct-v0.3` | LLM (4-bit Logits Extraction) | **0.6128** | 0.8861 | 0.9381 | 0.2876 | 0.6373 | 0.5997 | 0.5996 | 0.5708 |
| `ibm-granite/granite-3.3-2b-instruct` | LLM (4-bit Logits Extraction) | **0.3931** | 0.4330 | 0.5487 | 0.2375 | 0.4054 | 0.3939 | 0.3460 | 0.4224 |

---

## 🏆 2. Full 3-Class Benchmark (BTC Standard: `BENIGN`, `JAILBREAK`, `PROMPT_INJECTION`)

*Evaluated on the full competition dataset (N = 14,000 samples).*

| Model Name | Paradigm | 3-Class Macro-F1 | Accuracy | F1 (Benign) | F1 (Jailbreak) | F1 (Injection) | PT F1 | FR F1 | IT F1 | EN F1 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `mistralai/Mistral-7B-Instruct-v0.3` | LLM (4-bit Logits Extraction) | **0.4646** | **0.7153** | **0.8144** | 0.5488 | 0.0305 | 0.4179 | 0.4904 | 0.4788 | **0.5336** |
| `ibm-granite/granite-3.3-2b-instruct` | LLM (4-bit Logits Extraction) | **0.3839** | 0.4871 | 0.4519 | **0.5660** | **0.1339** | 0.4331 | 0.3485 | 0.3393 | 0.3491 |

---

## 🔬 3. Critical Empirical Insights

### A. The "Prompt Injection Blindspot" in Zero-Shot LLMs
- **Mistral-7B** achieves a high Accuracy of **71.53%** and F1 Benign of **0.8144**, but its F1 on `PROMPT_INJECTION` in 3-class drops to an alarming **0.0305 (3%)**!
- **Reason**: Standard Instruction-tuned LLMs are heavily aligned against explicit harms and toxic content (hence F1 Jailbreak is decent at **0.5488**), but they lack discriminatory representation for subtle prompt injection payloads (e.g. delimiters, system prompt leaks, context overriding) unless fine-tuned.

### B. Safety Encoders Dominate Binary Prompt Injection Defense
- `wolf-defender` (**ModernBERT**) outperforms all models with **0.7517 Macro-F1** and **0.9328 Benign F1**, running in just **~98 seconds** for 9,660 samples (~10ms/sample, ~8x faster than Mistral-7B).
- `deberta-v3-base-prompt-injection-v2` attains **0.6825 Macro-F1**, also outperforming LLMs on injection detection.

### C. Severe Multilingual Disparity (The "English Gap")
- On English (`en`), DeBERTa reaches **0.9372 F1**, but on Portuguese (`pt`) and Italian (`it`) it collapses to **0.6411** and **0.6341** (a **~30% performance drop**!).
- Safety encoders trained solely on English datasets fail to generalize robustly to Romance languages.

---

## 🎯 4. Strategic Recommendations for RMIT Hackathon 2026

1. **Deploy Cascaded Hybrid Guardrails**:
   - **Stage 1 (High-throughput Filter)**: Use `wolf-defender` (ModernBERT) or fine-tuned DeBERTa-v3 to quickly filter safe traffic.
   - **Stage 2 (Disambiguation)**: Escalate ambiguous samples or jailbreak candidates to an ensemble/LLM classifier.
2. **LoRA Fine-Tuning is Mandatory**:
   - Zero-shot LLM logits alone are insufficient for winning the competition (Macro-F1 0.46 vs competitive target > 0.90).
   - Fine-tuning DeBERTa-v3 or LLaMA/Mistral with cross-lingual multilingual data is required to bridge the Portuguese/Italian gap.
