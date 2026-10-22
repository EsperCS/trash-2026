# RMIT Hackathon 2026 — Hắc Minh Quân (HMQ): Project Brief for an AI Coding/Research Agent

**Prepared:** 17 October 2026  
**Team deadline:** Tuesday, 20 October 2026, 21:00 (Vietnam time, as stated in the team meeting notes)  
**Hackathon dates:** 22–23 October 2026  
**Primary task owner:** Hắc Minh Quân (HMQ)  
**Primary responsibility:** Research and experimentally evaluate fine-tuning methods for AI Security tasks.  

> **Reliability rule:** This brief consolidates the conversation and the linked/reference materials. It does not make the undisclosed 2026 challenge details up. The agent must distinguish (1) official 2026 facts, (2) the team's meeting decisions, (3) prior-year reference material, and (4) hypotheses. Never invent data, metrics, model capabilities, task requirements, or successful runs. A zero-error result cannot be guaranteed, but the workflow below requires explicit verification and honest reporting.

---

## 1. Mission and expected outcome

Do the technical work assigned to HMQ before the team deadline: **investigate and implement a reproducible fine-tuning/learning pipeline for an AI Security classification task**, using the RMIT Hackathon 2025 Kaggle competition as a practice reference if access and terms permit.

Deliver a clean, CV-quality project/research contribution that another teammate can run and understand. The work should help the team prepare for the 2026 combination of:

- **Security:** prompt injection, data poisoning, adversarial inputs and related AI-system vulnerabilities.
- **Generative AI:** improving accuracy, reliability and grounding of LLM-based systems.
- **Low-Resource Languages:** reducing reliability and safety gaps where training data is limited.

Security is the team's main preparation focus. The exact 2026 tasks are not public yet; the official event page says a warm-up and two challenges will be revealed on the hackathon day.

### Required final outcome

At minimum, provide:
1. Verified dataset inventory and a written record of what was actually accessible.
2. A working, reproducible baseline classifier.
3. One fine-tuning experiment if the dataset, hardware, dependencies and time allow. If blocked, document the exact blocker and still finish the baseline and data audit.
4. Comparable validation metrics and error analysis; no fabricated scores.
5. A clean README, environment/dependency specification, experiment results table and a short recommendation for the team.
6. A clear handoff explaining what Quân Hoàng, Phú and Nhân can use from this work.

## 2. Confirmed assignment and division of work

The uploaded meeting notes (`12_10_2026 - RMIT_MEET.md`) say the team wants to prepare a clean GitHub project that could later be included in CVs. All four members should study **SECURITY**.

| Member | Assignment recorded in the meeting notes | HMQ's relationship to it |
|---|---|---|
| **HMQ — Hắc Minh Quân** | Investigate how to fine-tune models for Security tasks | **This brief's owner.** Build/evaluate the training approach and report findings. |
| Quân Hoàng | Blue Team: train a model to distinguish risky inputs | Give him reproducible baseline/model findings and error analysis; do not silently replace his ownership. |
| Phú | Red Team: investigate image-based attacks | Coordinate categories of cases and model-evaluation needs; don't take over his assignment. |
| Nhân | Red Team: investigate prompt-based attacks | Coordinate text attack categories and safe evaluation needs; don't take over his assignment. |

**Deadline from meeting notes:** 21:00 on Tuesday, 20 October 2026.  
**Important distinction:** HMQ's assigned focus is fine-tuning for Security, not necessarily owning all Blue Team classification, all image attacks, or all prompt attacks.

## 3. RMIT Hackathon 2026 — facts from the official event page

Official site: <https://rmit-hackathon.com/>

- **Dates/times:** 08:00–18:00 on 22 October 2026; 08:00–12:00 on 23 October 2026.
- **Locations:** RMIT Vietnam Saigon South and Hanoi; RMIT Melbourne; Taylor's University Lakeside Campus in Malaysia.
- **Challenge platform:** Kaggle, hosted by SkywardAI Labs.
- **Competition language stated on the page:** English. This does not prove that all task inputs or datasets will be English.
- **Format:** one warm-up task and two challenges; specific details will be revealed on the hackathon day.
- **Participation:** every team member must attend the same venue; online or cross-venue participation is not permitted.
- **Eligibility/team:** four students per team; currently enrolled undergraduate or postgraduate students.
- **Registration:** the site says registration closed early; deadline changed from 15 October to 13 October 2026. This brief assumes the user/team is already registered; do not assume late registration is possible.
- **Fee/prize:** participation is free; total prize value stated as up to VND 100 million.
- **Themes:** Security × Generative AI × Low-Resource Languages.
- **Official contact:** GenAICC@rmit.edu.au.
- **Organiser/partners shown:** RMIT University School of Computing Technologies; RMIT RACE; Taylor's University (university partner); Tide (industry challenge sponsor); DesignLive Technologies/Compnow, Katalon and Orange Logic Vietnam (industry partners).
- **Past-event figures on the page:** more than 700 participants, 165+ teams and 30 universities are reported for RMIT Hackathon 2025 highlights.

Do not infer the undisclosed challenge statements, evaluation weights, exact 2026 dataset, exact leaderboard metric, or submission schema from the event overview.

## 4. Reference material and what it supports

### Source A — Prompt Injection & Jailbreak Techniques Gist

<https://gist.github.com/kibotu/c06f54d6fbc4705e886a50fb2e59e6ae>

The Gist calls itself a defensive/educational knowledge base, with deliberately defanged structural examples. Relevant topics include:

- Definitions and differences among prompt injection, jailbreak, direct injection, indirect injection, prompt leaking and multimodal injection.
- Direct jailbreak patterns; indirect injection through external web pages, documents, emails, retrieved/RAG content, tool outputs or code; encoding/obfuscation; multimodal/image-based injection; automated/optimization-based attacks.
- Low-resource-language generalization gaps and why a detector trained on a narrow dataset may fail on unseen languages or variants.
- Defensive approaches including instruction hierarchy, marking/separating untrusted content, input/output classifiers, adversarial training and layered controls.
- Benchmark/evaluation concerns.

**Critical caveat from the Gist:** attack-success-rate claims are date- and model-version-dependent, reported results may be inflated by lenient evaluation, and its status column describes reported research/vendor status rather than live testing. Never repeat an attack percentage as current fact without checking its primary source and context.

Use the Gist to build a high-level test taxonomy and literature pointers. Do not copy attack strings blindly or treat static reports as proof that an attack currently works.

### Source B — Microsoft Learn: Defend against indirect prompt injection

<https://learn.microsoft.com/en-us/security/zero-trust/sfi/defend-indirect-prompt-injection>

Microsoft's guidance describes external content (emails, documents, web pages, plugins, and similar sources) containing malicious instructions that an AI may wrongly treat as commands. It recommends **defense in depth**, including:

- Prompt Shields / detection and sanitization.
- Spotlighting, data marking and isolation of untrusted content.
- Plan-drift monitoring and critic agents.
- Tool-chain risk analysis and security guardrails.
- Information-flow control (IFC).
- Least privilege and short-lived privileges.
- Human confirmation for risky actions.
- Continuous monitoring and context-specific mitigations.

Implication for HMQ: fine-tuning or a text classifier is only one possible layer. Do not claim that fine-tuning alone solves indirect prompt injection in a full AI-agent system.

### Source C — RMIT Hackathon 2025 Kaggle competition

Overview: <https://www.kaggle.com/competitions/rmit-hackathon-2025/overview>  
Dataset page supplied in the meeting notes: <https://www.kaggle.com/competitions/rmit-hackathon-2025/data?select=test.csv>

The official Kaggle content did not expose its detailed page text in this review session. The agent must access the competition directly, verify permissions/rules and inspect the actual downloadable data. Do not rely on assumed schema, row counts or labels.

### Source D — Secondary 2025 challenge repository referenced during this conversation

<https://github.com/ngucocansang/RMIT-HACKATHON-2025>

The README retrieved during this conversation describes an earlier challenge setup, including:

- Text classification of prompts labelled `Jailbreak` or `Benign`.
- Challenge 1 (20%) described as a fundamentals/setup task; Challenge 2 (50%) as a binary classification task whose leaderboard metric is ROC-AUC.
- Candidate approaches include TF-IDF + Logistic Regression and transformer-based models such as DistilBERT/BERT.
- It describes a Kaggle prediction CSV with columns `id,target`, where `target` is the predicted probability of jailbreak, and says submission attempts were limited to 10.
- It describes Challenge 3 (15%) as a browser-only 2D web game using HTML/CSS/JavaScript, focused on a social issue.
- It describes Challenge 4 (15%) as a small Red Team evaluation set of five crafted prompts with recorded model responses/status codes and a PKL deliverable.

**Status:** this is a secondary repository describing a prior-year activity. Treat all percentages, limits, schemas, formats and challenge requirements as **2025-only reference information**, not 2026 instructions. Confirm against the actual competition rules before using any of them.

### Source E — Official 2026 event page

<https://rmit-hackathon.com/>

This is authoritative for the publicly posted event dates, themes, general rules and statement that the detailed warm-up/challenge content is revealed on the event day. It does not disclose the challenge-specific task or metric.

### Source F — Team meeting notes uploaded in this conversation

`12_10_2026 - RMIT_MEET.md`

Exact project-relevant points:
- The group wants a clean GitHub repo usable in a CV.
- Research 2026 themes: Security in AI, GenAI, Low-Resource training, with Security as the focus.
- Investigate the previous-year Kaggle materials as preparation.
- Meeting notes list DistilBERT, Qwen 3, Gemini 2.5 and Gemini 2 Flash as possible tools/models to consider for a temporary solution. This is a brainstorm, not a confirmed model requirement or guarantee that each model supports the same fine-tuning method.
- Work is divided among HMQ, Quân Hoàng, Phú and Nhân as shown in Section 2.
- Deadline: 21:00, Tuesday, 20 October 2026.

---

## 5. Mandatory first actions for the AI agent

Follow this order. Do not skip the audit and jump straight to training.

### Step 0 — Inspect the workspace and protect existing work

1. Inspect the current working directory, repository status, existing README, project files, installed Python/GPU environment and available disk space.
2. Identify whether a team repository already exists. Reuse it if available and appropriate; do not overwrite unrelated files or reset/delete existing work.
3. Do not publish a public repo, push changes, create a Kaggle submission, use paid APIs, expose secrets, or trigger external side effects without the user's/authorized team's permission. Prepare local work and report exactly what requires permission.
4. Record Python/package/device versions, CPU/GPU availability and the practical compute limits.

### Step 1 — Verify Kaggle access and inspect the real data

1. Open the 2025 competition and confirm access/eligibility and competition rules.
2. Identify the actual files, columns, target-label mapping, data types, nulls, duplicates, class counts, possible leakage, language distribution and whether the test labels are hidden.
3. Inspect a few records safely; do not dump private or sensitive material into the public repo.
4. Check dataset rules before redistributing data. Do not commit competition data, API keys, cookies, access tokens or private outputs to Git.
5. Record findings in `reports/data_audit.md`, including what was verified and what could not be accessed.
6. **Do not assume** the dataset has 5,000 records, 4,000 training rows and 1,000 test rows. Those numbers were mentioned in an earlier chat answer but were not verified during the current web review. Determine the actual counts from downloaded files.

If access is blocked, state the exact error and proceed with a clearly labelled synthetic/small local example only for pipeline tests. Do not pretend it is the actual competition dataset and do not invent leaderboard results.

### Step 2 — Build the simplest valid baseline

Start with a reproducible `TF-IDF + Logistic Regression` baseline if and only if the inspected data supports text classification.

- Use a train/validation split that matches the data, preferably stratified for class balance where appropriate.
- Fix and report random seed(s); use `42` as a default if the repository does not already specify one.
- Check exact duplicates and near-duplicates across splits to avoid misleading validation results.
- Save the label mapping and preprocessing decisions.
- Report ROC-AUC when probability-based binary classification and both validation classes make it valid; also report precision, recall, F1, confusion matrix and PR-AUC when useful/valid.
- Calculate all metrics on held-out validation data, not the training data.
- Do not tune against hidden test labels or claim local validation is a Kaggle leaderboard score.

### Step 3 — Fine-tune one transformer model

Choose a model only after inspecting data language, data size, licence, compute availability and the challenge's likely task format.

- If the observed task is English text classification and resources are limited, **DistilBERT** is a reasonable first candidate.
- If the actual data includes multiple languages and multilingual robustness is a real requirement, consider a multilingual model such as **XLM-RoBERTa**, and justify the choice using the audit.
- Treat Qwen 3 or another generative model as a separate option only when it is appropriate and resources/API permissions permit. Do not assume that a hosted Gemini API supports fine-tuning or behaves like an open-weight trainable model; verify the model/provider's current capabilities first.
- Establish a minimal fine-tuning run before attempting expensive experiments. Save model/config versions, tokenizer, hyperparameters, seed, epoch, batch size, learning rate, max sequence length, early-stopping/checkpoint policy and run time.
- Compare with the baseline under the same validation split and metrics. Report any failure, instability, out-of-memory problem or incomplete run.
- If the 2025 dataset is inaccessible or unsuitable, do not fabricate results. Finish baseline/pipeline scaffolding and provide exact next steps to run once authorised data is available.

### Step 4 — Robustness and security analysis

Build a **defensive, authorised evaluation set** using safe and benign examples or redacted placeholders. Include categories only if data and time permit:

- ordinary benign instructions and difficult-but-legitimate requests;
- direct prompt-injection/jailbreak indicators;
- indirect instructions embedded in documents or retrieved content;
- paraphrase/format/obfuscation variations, tested safely;
- language groups actually represented in the dataset;
- image/multimodal-related cases only if a permitted dataset and the team's image pipeline make this feasible.

Measure false positives and false negatives. Break down metrics by available categories/languages. Do not create or execute harmful payloads, target systems without permission, exfiltrate data, or use real credentials/secrets. For risky examples, use placeholders such as `[REDACTED_REQUEST]` and score whether the system follows untrusted instructions rather than reproducing harmful content.

### Step 5 — Translate Microsoft guidance into system-level recommendations

In `reports/security_recommendations.md`, explain where a detector/fine-tuned model fits into a broader defence-in-depth design. Consider, as applicable:

- separating trusted instructions from untrusted retrieved/user content;
- marking/delimitating untrusted data and provenance;
- input/output screening;
- least-privilege and short-lived tool access;
- risky tool-call review and user confirmation;
- monitoring for task/plan drift;
- false-positive, latency and operational trade-offs.

Only implement features that fit the actual project scope and timeline. Don't pretend a simple classifier implements all of these controls.

---

## 6. Recommended local project structure

Adapt this to the existing repository rather than forcing a structure over existing team conventions.

```text
rmit-hackathon-security/
├── README.md
├── requirements.txt                 # or pyproject.toml; use one coherent approach
├── .gitignore
├── .env.example                     # variable names only; never secrets
├── configs/
│   └── baseline.yaml                # or documented config file
├── notebooks/
│   └── 01_data_audit_and_baseline.ipynb  # optional if scripts are clearer
├── src/
│   ├── data_audit.py
│   ├── train_baseline.py
│   ├── train_transformer.py
│   └── evaluate.py
├── tests/
│   └── test_data_and_metrics.py
└── reports/
    ├── data_audit.md
    ├── experiment_results.csv
    ├── error_analysis.md
    ├── security_recommendations.md
    └── HMQ_handoff.md
```

Keep code as simple as possible. Avoid unnecessary dependencies and architectural overengineering. Do not commit model checkpoints, secrets or competition data unless access/licence terms explicitly permit it and the team approves.

---

## 7. Experiment plan and result table

Run these in priority order:

1. Data audit and split validation.
2. TF-IDF + Logistic Regression baseline.
3. One Transformer fine-tuning run.
4. Error/robustness analysis.
5. Optional improvements only if time remains: word + character n-gram comparison, threshold tuning using validation data, or a second model that answers a specific question.

Use a result table like the following and fill it only with measured values:

| Run ID | Model | Train/validation split | Seed | ROC-AUC | PR-AUC | Precision | Recall | F1 | Runtime | Notes |
|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| baseline-01 | TF-IDF + Logistic Regression | actual split recorded in report | 42 | **measured** | **measured** | **measured** | **measured** | **measured** | **measured** | Baseline |
| transformer-01 | selected Transformer | same held-out split where possible | 42 | **measured** | **measured** | **measured** | **measured** | **measured** | **measured** | Fine-tuning |

If a metric is undefined for the actual data/split, write `N/A` and explain why. Never use example placeholders as if they were results. State whether reported numbers come from training, local validation, cross-validation or an official leaderboard.

---

## 8. Suggested timeline

### 17 October — audit and baseline setup
- Inspect repository/environment, verify Kaggle access, audit dataset and rules.
- Build/run baseline; record an initial result only if it truly ran.
- Send teammates an early summary of access blockers and findings.

### 18–19 October — fine-tuning and evaluation
- Select model based on observed data and compute.
- Run one controlled fine-tuning experiment.
- Compare metrics, inspect false positives/negatives, test robust generalization where feasible.
- Coordinate relevant findings with Quân Hoàng, Phú and Nhân.

### 20 October — reproducibility and handoff
- Re-run essential commands from a clean environment if possible.
- Finalize README, dependencies, audit, results and recommendation.
- Deliver the handoff no later than **21:00 on 20 October 2026**.

Prioritise a verified baseline and honest report over unfinished complex experiments.

---

## 9. Acceptance checklist — do not call the task complete before checking these

### Correctness and reproducibility
- [ ] Actual dataset path, files, columns, target mapping, row counts and class counts are documented from observed data.
- [ ] Competition access and relevant dataset/submission rules were checked, or the access blocker is stated.
- [ ] Baseline runs end to end without relying on undocumented local files.
- [ ] Fine-tuning either runs end to end or has a precise, evidence-based blocker report.
- [ ] Train/validation separation is respected; duplicates/leakage are examined.
- [ ] Metrics are generated by executable code and correspond to the stated split.
- [ ] Random seeds, dependencies, model/tokenizer identifiers and hyperparameters are recorded.
- [ ] No metric, result, citation, dataset detail or successful command is fabricated.

### Security and data handling
- [ ] No API key, password, cookie, access token or secret appears in files/logs/commits.
- [ ] No dataset is published without checking licence/competition terms.
- [ ] Security evaluation stays within authorised, controlled settings and uses defanged/redacted examples as appropriate.
- [ ] The report explains false positives, false negatives and known limitations.
- [ ] The write-up does not claim fine-tuning alone eliminates prompt injection.

### Deliverables and handoff
- [ ] README contains prerequisites, setup, data placement instructions, commands, expected outputs and caveats.
- [ ] `reports/data_audit.md` and a measured experiment table exist.
- [ ] `reports/security_recommendations.md` separates model-level results from system-level defence ideas.
- [ ] `reports/HMQ_handoff.md` gives teammates the key results, intended use, failure cases and next action.
- [ ] Existing team work was not deleted or overwritten.
- [ ] No public push/Kaggle submission/paid API action occurred without appropriate authorisation.

---

## 10. What the agent must report at the end

Return a concise execution report with:

1. **Completed:** exact files created/edited and commands actually run.
2. **Environment:** Python, key package versions, CPU/GPU and constraints.
3. **Data:** exact observed data files/schema/row and class counts, or an exact access error.
4. **Results:** measured metrics and which split produced them; link to result table.
5. **Model decision:** why the selected model fits the observed data; compare against baseline.
6. **Limitations:** untested hypotheses, likely distribution shift, unsupported modalities/languages and any incomplete step.
7. **Handoff:** recommendations for Quân Hoàng, Phú and Nhân.
8. **External actions:** whether anything was pushed/submitted/paid for (default should be no unless authorised).
9. **Next best action:** one specific action if a blocker remains.

Do not merely say “done”; include verifiable evidence. If time, data access or hardware prevents completion, deliver the best working partial result, say exactly what failed and why, and do not disguise an incomplete run as complete.

---

## 11. Hypotheses for the 2026 challenge — clearly unconfirmed

Based on the public themes, previous-year material and the team's assignments, reasonable preparation hypotheses are:

1. **Prompt/jailbreak risk classification:** a model may be expected to distinguish benign from suspicious inputs, possibly with a probability-based evaluation.
2. **Robustness across wording/languages:** a detector or GenAI system may be assessed on input variations or language/resource gaps.
3. **Indirect prompt injection:** a system may need to handle untrusted instructions hidden in documents, web pages, or other retrieved content.
4. **Multimodal/image-related security:** the team's separate image-based Red Team task suggests it is worth familiarising with the concept, but it is not confirmation of the official 2026 challenge.
5. **Red Team versus Blue Team testing:** last year's secondary repository included an attack-evaluation task, but that does not mean 2026 will reuse the same format.

**Preparation priority:** (1) reproducible classification baseline, (2) one fine-tuned model, (3) robust validation/error analysis, (4) concise defence-in-depth design notes. Do not build a solution that depends on any one guessed prompt, dataset, model, score or output schema.

---

## 12. Final source list

1. RMIT Hackathon 2026 — official event page: <https://rmit-hackathon.com/>
2. kibotu, “Prompt Injection & Jailbreak Techniques — Comprehensive Reference”: <https://gist.github.com/kibotu/c06f54d6fbc4705e886a50fb2e59e6ae>
3. Microsoft Learn, “Defend against indirect prompt injection attacks”: <https://learn.microsoft.com/en-us/security/zero-trust/sfi/defend-indirect-prompt-injection>
4. RMIT Hackathon 2025 — Kaggle overview supplied by the user: <https://www.kaggle.com/competitions/rmit-hackathon-2025/overview>
5. RMIT Hackathon 2025 — Kaggle data page supplied in meeting notes: <https://www.kaggle.com/competitions/rmit-hackathon-2025/data?select=test.csv>
6. Secondary repository describing prior-year tasks, retrieved during this conversation: <https://github.com/ngucocansang/RMIT-HACKATHON-2025>
7. Team meeting notes supplied in this conversation: `12_10_2026 - RMIT_MEET.md`.

**Final instruction to the agent:** Treat this document as the project brief, not as proof that data access or experiments succeeded. Inspect the actual workspace and source files, verify every data-dependent assumption, run what can be run, preserve existing work, and make every claim auditable.
