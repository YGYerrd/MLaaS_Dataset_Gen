# MLaaS / MSC Presentation — Speaker Notes

Aligned with:
- **MATH3004 Project Proposal** — *Benchmark Dataset Generator for Machine Learning-as-a-Service Ecosystems*
- **ICSOC 2026** — *MSC: A Benchmark Dataset for Machine Learning as a Service (MLaaS) Composition*

Slide numbers match `build_presentation.py` / `MLaaS_presentation.pptx`. Target length: **13–16 minutes** plus Q&A.

---

## Slide 1 — Title

**Opening (45 sec)**

- Introduce yourself: **Joshua Boland**, BSc Data Science, **MATH3004 Industrial Project**.
- Name supervisors: **A/Prof Sajib Mistry** and **Deepak Kanneganti** (Kanneganti’s work on adaptive MLaaS composition for IoT is central to the motivation).
- Give **two titles** on purpose:
  - *Proposal title* — benchmark **generator** for MLaaS ecosystems (what you are building).
  - *Paper title* — **MSC dataset** for MLaaS **composition** (the research artefact and ICSOC contribution).
- One-line hook from the paper abstract: *We provide the largest known MLaaS composition dataset—25,900 services and 20,000 requirements with benchmark solutions—because providers do not expose enough detail for fair composition research.*

**Transition:** “I'll start with why MLaaS composition needs its own benchmark tradition.”

---

## Slide 2 — Motivation

**Key points (1–1.5 min)** — *Proposal §1, ICSOC §1*

- **IoT context** (proposal): healthcare, transport, smart cities need ML on sensor data; delivery has moved to **cloud MLaaS** (Ribeiro et al. 2015; Lee et al. 2015).
- **MLaaS composition** (proposal Fig. 1; Kanneganti et al. 2025; Xie et al. 2024): one model is rarely enough—systems chain image recognition, NLP, forecasting, etc.
- **Selection problem**: services differ in accuracy, cost, latency, reliability; orchestrators must pick combinations under constraints.
- **Data gap**: unlike **QWS** (2,507 web services, nine QoS params) and **WS-DREAM** (millions of invocations), there is no equivalent **ML-native** corpus with model specs, training data characteristics, and diverse QoS—including explainability.

**Talking tip:** “Web-service benchmarks enabled decades of composition papers; MLaaS needs the same foundation.”

**Transition:** “Existing datasets stop short—here is how we compare them.”

---

## Slide 3 — Research gap

**Key points (1 min)** — *ICSOC Table 1, §2*

- Walk the table left to right—don't read every cell.
- **QWS / WS-DREAM**: foundational for **web** QoS; no ML model parameters, modalities, or training behaviour.
- **HSC** (Hugging Face AI service composition, Wang et al. 2024): big step for AI services but **sequential-only**, partial QoS/XAI, weak collaborative support.
- **MSC (ours)**: full support across MLaaS-specific attributes, internal parameters, explainability, and **both** composition types.
- Optional: mention **IMLaaS** (Kanneganti) as related IoT-focused work—partial composability in the paper’s table.

**If asked “why not scrape Hugging Face only?”** Metadata is incomplete; MSC **runs** services on benchmark splits to measure real behaviour (paper §3.1).

**Transition:** “The MATH3004 project formalised five objectives to close this gap.”

---

## Slide 4 — Project objectives

**Key points (1 min)** — *Proposal §2*

- Read objectives 1–5 succinctly—they map directly to your deliverables.
- **Objective 1–2**: modular Python framework; training **and** inference paths (proposal §4.4).
- **Objective 3**: functional attributes **and** QoS for **services and compositions** (proposal §4.5–4.6).
- **Objective 4**: evaluate existing selection/composition algorithms (paper §5 benchmark).
- **Objective 5**: reproducibility—manifests, CLI, public repo (paper mentions anonymous release).
- **Bridge sentence (important):** *This GitHub repository is the **service-generation engine**—registry, manifest, runner, SQLite. The full MSC pipeline in the paper adds requirement generation and MAB/GA composition on top of that catalogue.*

**Transition:** “The MSC dataset scale is what makes the contribution concrete.”

---

## Slide 5 — MSC dataset at a glance

**Key points (1 min)** — *ICSOC abstract, §3, §9*

- Hit the four stats: **25,900** services, **20,000** requirements, **12** task families, **5** provider sources (HF, BERT family, sklearn, Keras, etc.—see paper Table 2).
- **Composition split**: 10k sequential + 10k collaborative scenarios (paper §5.1).
- **Solutions**: not just requirements—**MAB-based composability** produces benchmark composition answers (paper Algorithm 2).
- Claim carefully: “largest **known** MLaaS composition dataset” is the paper’s wording—use the same phrasing.

**Transition:** “Composition takes two qualitatively different forms.”

---

## Slide 6 — MLaaS composition types

**Key points (1 min)** — *ICSOC §1, Proposal Fig. 1*

- **Sequential**: pipeline/workflow; paper healthcare example—image → text → summary. Composability aggregates stage-wise (Table 4, sequential columns).
- **Collaborative**: merge complementary capabilities; HAR example from paper—walking/running vs sitting. Compatibility matters more than strict ordering.
- Tie to **proposal**: monitoring systems combining specialised models is the same idea at application level.

**Transition:** “Figure 2 in the paper is the macro pipeline—here is our version.”

---

## Slide 7 — MSC collection workflow

**Key points (1 min)** — *ICSOC Fig. 2, §3*

- Five-step narrative: **collect → measure → requirements → optimise → publish**.
- **§3.1**: ~1,500 top-ranked models per task family → 25,900 configured **service instances** (not just model cards).
- **§3.2**: synthetic but principled requirements—profiles for healthcare (accuracy + low latency) vs finance (reliability + explainability).
- **§3.3**: combinatorial explosion—paper’s C(100,2)=4950 example; MAB for smaller spaces, GA for larger.
- Point to diagram while speaking—audience should see this as **downstream** of service collection.

**Transition:** “What exactly is stored per service?”

---

## Slide 8 — What each MLaaS service records

**Key points (1 min)** — *ICSOC Table 3, §3.1*

- Table symbols are from the paper—Δ for **IID vs non-IID**, Dₙ sample count, **w** for parameters, etc.
- **Explainability (X)**: not SHAP in the paper—**input perturbation sensitivity**; stable responses score higher (§3.1).
- **Normalisation**: min–max for fair composability across scales (except already-bounded scores).
- **Runtime variability**: repeated latency measurements—not single-shot metadata.
- Link to repo: same **metric domains** in `service_metrics` (quality, latency, resource, explainability, …).

**Transition:** “Explainability was one of the hardest QoS attributes to define—here is how we handled it.”

---

## Slide 9 — Explainability as a QoS metric

**Key points (1.5–2 min)** — *Proposal §4.5, ICSOC §3.1, `services/perturbation.py`*

**Set the scene (20 sec)**

- Traditional composition research optimises **latency, reliability, cost**—measurable and comparable (QWS, WS-DREAM).
- **MLaaS adds interpretability**: stakeholders ask *why* a model predicted this—not only *how fast* or *how accurate*.
- **Domain stakes**: healthcare (patient safety, clinical trust) and finance (regulation, auditability) weight explainability heavily—matches paper requirement profiles and proposal significance §3.

**The challenge — say this clearly (30 sec)**

- “A major challenge was defining explainability **consistently across different task families**.”
- Text needs token/word-level methods; images need attribution maps; tabular needs feature importance—**no single off-the-shelf XAI tool** spans all twelve MSC families.
- A unified **SHAP + Grad-CAM + LIME** pipeline would become **twelve bespoke integrations**, each with different dependencies and runtime—**not viable** when generating tens of thousands of service records.

**Engineering decision — core of the slide (40 sec)**

- “We did **not** claim to solve XAI academically. We made a **practical engineering decision**.”
- **Perturbation-based sensitivity analysis** as a **generalised explainability proxy** (aligned with ICSOC paper §3.1: input-variation sensitivity).
- Procedure in plain language:
  1. Run normal benchmark evaluation.
  2. **Perturb inputs** (targeted units—tokens, regions, features—and benign random noise).
  3. **Observe output stability**—confidence drops, prediction flips, task-quality degradation.
  4. Models that stay **stable** under small, meaningful perturbations score as **more explainable / trustworthy** → `explainability_score` (higher is better).

**Implementation — show you built it (30 sec)**

- Implemented in **`mlaas_data_generator/services/perturbation.py`**, invoked from the service runner after evaluation.
- **Task-family-aware** perturbation units; **sample limits** and **time budgets** so batch generation stays feasible; **best-effort**—failures degrade the score, they do not abort the service row.
- Registry still records preferred methods (e.g. attention, SHAP) as **metadata**; runtime metric is the **perturbation proxy** so every family gets a **comparable scalar X** for composition (MSC Table 3).
- Registries may skip heavy multimodal probes where cost would dominate—another pragmatic trade-off.

**Address proposal criticism proactively (15 sec)**

- Proposal asks for rich QoS including explainability; reviewers may ask “why not SHAP?”
- Answer: “**Recognised the challenge**, chose a **cross-task, scalable proxy** documented in the paper, implemented and stored per service—sufficient for **honours/project-level** composition benchmarking.”

**If asked “is this real XAI?”**

- “It is an **operational trust proxy** for **dataset generation and composition**, not a clinical-grade explanation. Future work can attach SHAP/Grad-CAM per family where budget allows.”

**Transition:** “Explainability is one behavioural dimension—next, how we simulate the whole provider ecosystem.”

---

## Slide 10 — Simulating realistic ML providers

**Key message (1.5–2 min)** — *This is a core contribution slide, not a proposal summary*

**Open with the banner line (say it slowly):**

> “The goal was not to find the best ML model. The goal was to simulate many realistic ML providers with diverse behaviours.”

Pause. That separates you from ordinary ML benchmarking.

**What you are NOT doing (15 sec)**

- Not training one champion model.
- Not optimising accuracy alone.
- Not publishing a single SOTA result.

**What you ARE doing (20 sec)**

- Generating **many** ML-as-a-Service **providers**.
- Simulating **heterogeneous** behaviours across tasks, configs, and runtimes.
- Building a **service ecosystem** dataset for composition research.

**Walk the pipeline (left side, 25 sec)**

Point top → bottom:

1. **Base models** — curated registries (HF, sklearn, Keras paths).
2. **Different datasets** — benchmark splits per task family.
3. **Training conditions** — where heterogeneity is **created** (emphasise).
4. **Runtime behaviours** — emerge from real execution.
5. **ML service records** — one catalogue entry per provider instance.

**Card 1 — Task & model diversity (15 sec)**

- NLP, vision, tabular — twelve task families in MSC framing.
- Multiple architectures—not one template model.

**Card 2 — Service variation (40 sec) — SPEND TIME HERE**

This is the **most important** card (highlighted on slide).

- **Hyperparameters**: learning rate, batch size, epochs, optimiser—manifest knob variants.
- **Dataset conditions**: **IID vs non-IID**, split strategies, skew axes—providers trained on different data **behave** differently.
- **Training regimes**: **inference-only** vs **fine-tune / transfer**—mirrors real MLaaS offerings.

Say explicitly: “This is how we get thousands of distinct providers, not thousands of copies of the same model.”

**Card 3 — Runtime execution (20 sec)**

- Actually run train/inference workflows.
- Latency and resource usage come from **architecture + hardware + workload**—not hand-written fake QoS tables.
- Reinforces: **“not synthetic metadata.”**

**Card 4 — Behavioural measurement (20 sec)**

- Frame as **service behaviour**, not a disconnected metrics list.
- Prediction quality, latency, reliability under execution, resource use, explainability proxy.
- Ties to previous XAI slide without re-explaining perturbation.

**Card 5 — Dataset generation (15 sec)**

- Standardised, comparable **service records** for downstream composition.
- Reproducible manifest-driven experiments.
- **Do not** dwell on SQLite here—that belongs on the implementation slide.

**Proposal mapping (only if asked)**

- §4.1–4.6 in the proposal; implementation is the next slide.

**Transition:** “Here is how the repository implements that pipeline end to end.”

---

## Slide 11 — Implementation architecture

**Key points (1.5–2 min)** — *Replaces old implementation + pipeline slides*

**Open (10 sec):** “From configuration to a benchmark-ready dataset—five stages.”

**Walk the coloured pipeline top → bottom with columns (90 sec)**

1. **Registry (blue)** — `models.py` / `datasets.py`; curated tasks, HF + sklearn/Keras paths; not random downloads at run time.

2. **Manifest generation (green)** — `hf-manifest` combinatorially builds **many provider configurations**: hyperparameters, dataset/split variants, IID vs non-IID, inference vs fine-tune. This is where scale comes from.

3. **Service execution (orange)** — `run-manifest` runs the real workflow: validate row → load splits → build model → train or inference → benchmark eval. Mention **dry-run**, **grouped HF**, **`--workers`** only if asked—keep conceptual.

4. **Metrics collection (purple)** — frame as **behavioural capture**, not “logging accuracy”:
   - Prediction quality + task metrics
   - Latency, reliability, resource/cost
   - Explainability score (perturbation—link slide 9 briefly)

5. **Benchmark dataset (dark blue)** — `services.db`; **one manifest row → one service record**; feeds MSC-style composition research.

**Dark banner (20 sec)—say verbatim:**

> “One manifest row = one simulated ML provider. Each row is a unique configuration; execution produces one catalogue record.”

**Light banner (10 sec):** Heterogeneous, realistic providers; reproducible benchmarking—not a single-model leaderboard.

**What NOT to do on this slide:** Don’t read seven pipeline sub-steps from the old table; the five stages subsume them. Don’t deep-dive SQLite schema.

**Transition:** “The architecture spans multiple task domains — but why does that matter?”

---

## Slide 12 — Why task diversity matters

**Key message (1.5 min)** — *Behavioural diversity, not a model census*

**Open (15 sec)**

- Different **ML tasks behave differently** at runtime.
- That is **why** QWS/WS-DREAM-style generic QoS tables are insufficient.
- We need **ML-specific** measurement: latency distributions, resource footprints, explainability proxies.

**Do NOT lead with:** “We have 25,900 services” or reading Table 2 counts. Save scale for MSC slide (slide 5) if needed.

**Intro line on slide (10 sec)**

- Web benchmarks assume homogeneous services; ML breaks that assumption.

**Five domain cards (70 sec)** — emphasise **behaviour**, not inventory:

1. **NLP** — sequence length drives **latency spread**; token vs sentence workloads; perturbation on **text units** for explainability.

2. **Vision** — **high GPU/compute** cost; batch size swings throughput; explainability behaviour ≠ text.

3. **Segmentation** — **memory/VRAM** heavy; slower than classification; pixel-level outputs → different QoS trade-offs.

4. **Classification** — often **faster inference** but **accuracy-sensitive** ranking; class skew hits **reliability**.

5. **Regression / tabular** — **lower compute**; feature-space logic; **RMSE/MAE** and different cost curves.

Tie back: “When we compose services, these behavioural differences compound — composition research needs to see them.”

**Banner — say verbatim (20 sec):**

> “The goal was not simply to include many tasks. The goal was to generate **realistic variation in ML service behaviour**.”

**Footnote only if asked (10 sec):** Generator covers **12 MSC families** via `hf-manifest --task-keys`; counts on slide 5 support **breadth**, but this slide is about **why breadth matters behaviourally**.

**Transition:** “The catalogue also supports downstream research—without being the core of this project.”

---

## Slide 13 — Downstream research on the service catalogue

**Key message (1–1.5 min)** — *Secondary support · not the core contribution*

**Frame first (15 sec):**

- Your project is a **dataset generation framework**.
- Composition and algorithm benchmarks in the **ICSOC paper** are an **application layer** on top—not what the repo CLI primarily builds today.

**Left column — Extending to composition research (35 sec)**

- Simple flow: **service records → requirements → applications**.
- Requirements in plain language: **accuracy needs, latency limits, cost constraints** (not F, QoS, PW, C acronyms unless asked).
- Applications: **recommendation, orchestration, service selection, evaluation**.
- **Do NOT** present MAB, GA, composability math, or 20,000 requirement pairs as your implementation focus.
- If asked: MSC paper generates those; **this repo** publishes the provider catalogue they consume.

**Right column — Benchmarking ML service behaviour (35 sec)**

- Walk the **comparison table** row by row—this is your stronger research point vs algorithm names.
- **Traditional datasets**: thin QoS, static advertised values, little ML structure, small homogeneous provider sets.
- **MLaaS generator**: rich behavioural metrics, **measured** runtime variability, accuracy + explainability + resources, **heterogeneous** realistic providers.
- Three enables: **fair comparison**, **reproducible evaluation**, **realistic optimisation problems**.
- **Avoid** listing DAAGA, MWOA, ASQ unless examiner drills into ICSOC §5—one line max: “paper shows richer attributes beat HSC; details in publication.”

**Banner — say verbatim (20 sec):**

> “Once realistic ML service records exist, they can support higher-level research tasks like composition and orchestration.”

**Transition:** “Here is the key contribution—in one sentence.”

---

## Slide 14 — Key contribution

**Key points (45 sec–1 min)** — *Not “impact” or deliverables laundry list*

**Read the headline (slowly):**

> “A reproducible framework for generating heterogeneous ML service ecosystems.”

**Enables (pick 3–4 if short on time):**

- **Fair benchmarking** — comparable records, not ad hoc notebooks.
- **Reproducible evaluation** — manifest → execution → catalogue.
- **Runtime QoS analysis** — measured latency, resources, reliability—not static tables.
- **Orchestration / selection research** — downstream use of the catalogue.
- **Future composition research** — application layer (ICSOC), not the generator core.

**Do NOT:** List ICSOC publication, composition dataset deliverables, or “QWS moment” here—that was old framing.

**Transition:** “Every strong project has limitations—here are ours.”

---

## Slide 15 — Limitations and open challenges

**Key points (1 min)** — *Shows academic maturity; addresses proposal feedback*

- **Explainability:** no universal XAI across NLP, vision, tabular—we use a **perturbation proxy** by design (link slide 9 briefly).
- **Runtime cost:** thousands of real train/eval runs are **expensive**; manifests use caps, tiers, grouped execution.
- **Real-world realism:** public HF models ≠ full commercial MLaaS deployments (optimisations, SLAs, hidden metadata).
- **Ecosystem evolution:** models and tasks change—catalogues need **versioning** and extension via registries.

Tone: confident, not apologetic—“these define future work,” not “the project failed.”

**Transition:** “You can run a smoke version locally—or jump to the close.”

---

## Slide 16 — Demo

**Key points (1–2 min if live; 30 sec if not)**

- Three commands from README—stress **`--dry-run`** before GPU time.
- If not live: show pre-built `service_manifest.xlsx` or SQLite query:  
  `SELECT service_id, task_type, training_regime FROM services LIMIT 5;`
- Mention ICSOC anonymous repo URL if audience wants full MSC artefacts.

**Transition:** “To close—five lines and one takeaway.”

---

## Slide 17 — Summary

**Closing (45–60 sec)** — *Very focused; no MSC statistics here*

Walk the five rows—do not add extra bullets:

1. **Problem** — no realistic benchmark for **heterogeneous ML service behaviour**.
2. **Solution** — configurable **dataset generator** (this project).
3. **Method** — execute diverse providers, **capture runtime behaviour**.
4. **Output** — standardised **benchmark-ready service records**.
5. **Contribution** — reproducible MLaaS benchmarking + **future** orchestration/composition research.

**Final sentence (verbatim):**

> “The project shifts benchmarking from static web services toward realistic, behaviour-driven ML service ecosystems.”

Pause. **Questions?**

**Do NOT** end on 25,900 services, MAB, or composition as the main punchline—that undercuts the generator framing.

---

## Appendix A — Anticipated Q&A

| Question | Answer |
| --- | --- |
| Proposal vs paper vs repo? | **Proposal** = research plan; **ICSOC paper** = MSC dataset & benchmark results; **Repo** = generator/runner for service records (core of §4.1–4.5). |
| Is the full 25,900-service DB in this repo? | The paper describes the collected MSC; this repo **produces** compatible records. Point to paper/repo URL for released artefacts. |
| Why SQLite? | Proposal §4.6 normalised DB; portable, queryable; per-GPU shards under parallel workers. |
| Federated learning in repo? | Experimental `federated/` code is archived; active generation is **service-only**. Paper cites FL literature for composability inspiration (Fu et al.). |
| Explainability score vs metrics table? | Paper defines X via perturbation sensitivity; repo implements `run_perturbation_stage()` → `explainability_score` in `service_metrics`. Not full SHAP—by design. |
| Why not SHAP/Grad-CAM? | Task-specific, expensive at 25k+ services; perturbation proxy gives one comparable scalar X per service for composition. |
| Supervisors’ contribution? | Frame as industrial project under Mistry & Kanneganti; cite Kanneganti MLaaS/IoT composition papers in gap slide. |
| Ethics? | Paper §8: public models/datasets only; no publisher identity or private user data. |

---

## Appendix B — Key citations (quick reference)

| Source | Use when discussing |
| --- | --- |
| Al-Masri & Mahmoud 2008 (QWS) | Web-service QoS benchmark precedent |
| Zheng et al. 2013 (WS-DREAM) | Large-scale invocation QoS |
| Kanneganti et al. 2025 | Adaptive MLaaS composition for IoT |
| Xie et al. 2024 (SkyML) | MLaaS federation / multicloud |
| Wang et al. 2024 (HSC) | Prior AI composition dataset; MSC comparison |
| Ribeiro et al. 2015 | MLaaS definition |

---

## Appendix C — Timing guide (15-minute version)

| Slides | Topic | Minutes |
| --- | --- | ---: |
| 1–2 | Title + motivation | 2 |
| 3–5 | Gap + objectives + MSC scale | 3 |
| 6–8 | Composition types + workflow + attributes | 3 |
| **9** | **Explainability as QoS (XAI)** | **1.5–2** |
| 10–11 | Provider simulation + implementation architecture | 2.5 |
| 12–13 | Task diversity + downstream research | 2 |
| 14–15 | Key contribution + limitations | 1.5 |
| 16–17 | Demo (optional) + summary | 1+ |

**Short talk (8 min):** Keep slides 3, 5, **9**, **11**, **15**, **17**; drop 10, 12, 13, 16.

---

## Appendix D — Alignment checklist

Before presenting, confirm you can explain:

- [ ] Difference between **sequential** and **collaborative** composition with paper examples
- [ ] Why **MSC** improves on **HSC** (Table 1 + Fig. 8)
- [ ] Five **proposal objectives** and which are done in the repo today
- [ ] What **MAB** does in the paper (benchmark solutions, not production orchestrator)
- [ ] **Explainability**: challenge → why not unified SHAP → perturbation proxy → `perturbation.py`
- [ ] **Limitations slide**: four cards, confident tone—not defensive
- [ ] **Close**: behaviour-driven ecosystems sentence (not MSC scale)
- [ ] Honest boundary: **service generator now** vs **composition dataset** in paper/proposal
