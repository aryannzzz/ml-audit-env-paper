# MLAuditBench Datasheet
*Following Gebru et al. (2021) "Datasheets for Datasets"*

## Motivation
- For what purpose was this dataset created?
MLAuditBench was created to evaluate whether language-model agents can perform integrity auditing of machine learning experiments as an interactive task. The benchmark targets leakage and reporting failures that appear in reproducibility studies, and it is designed to test sequential evidence gathering, cross-artifact comparison, and evidence-grounded decision making under a step budget.

- Who created it and on whose behalf?
The dataset and environment were created by the anonymous authors (withheld for blind review), with packaging and deployment aligned to the Hugging Face Spaces / OpenEnv execution model.

- Was there funding? Any conflicts of interest?
N/A - No explicit grant funding information is documented in the repository. The benchmark was produced in a competition setting. No direct conflicts of interest are declared in project files.

## Composition
- What does each instance represent? (one synthetic ML experiment)
Each instance represents one synthetic ML experiment package with multiple artifacts that mimic what an auditor would inspect in practice. Typical artifacts include dataset metadata, preprocessing code snippets, split configuration, model configuration, validation strategy, run history, evaluation report, and free-text experiment notes.

- How many instances? (56 total: 36 violated (including 6 compound hard-tier experiments), 20 clean)
The pool contains 56 total instances: 36 violated experiments (including 6 compound hard-tier experiments) and 20 clean experiments.

- What violation types are present and in what counts?
The benchmark includes V1 through V8: preprocessing leakage, temporal shuffle leakage, target leakage, train/test overlap, undisclosed multiple runs, selective metric reporting, entity leakage, and overfitting to test set. The pool generation policy ensures each violation appears multiple times across tiers. Exact counts per violation can be recomputed from the generator output and may vary when generator logic is changed in future versions.

- Is there a label/ground truth? What form does it take?
Yes. Ground truth is stored programmatically per experiment in a structured field that includes a list of violation IDs and an expected final verdict. This is consumed by the grader and not exposed to agents during normal episodes.

- Are there recommended data splits?
Yes. Recommended use follows the built-in difficulty split: easy, medium, and hard task pools, plus adversarial clean examples. For reporting, evaluate per tier and report macro average with fixed random seeds for reproducibility.

- Does the dataset contain sensitive data?
No real personal or proprietary data is included. Instances are synthetic and template-generated. Some fields use realistic placeholders such as patient or entity identifiers, but these are fabricated and non-identifying.

## Collection Process
- How was data collected? (programmatic injection via inject_V1..V8)
Data was generated programmatically from base experiment templates located in the repository. Violation injector functions inject_V1 through inject_V8 modify selected artifacts to create concrete failure modes. Six red-herring generators create suspicious but non-violating signals to reduce shortcut exploitation.

- What red herrings are present in clean experiments?
Six adversarial red-herring patterns are injected into clean (no-violation) experiments to prevent agents from learning a trivial "always-flag" policy. Each is designed to mimic a real violation surface without meeting the ground-truth criteria:

  1. **Unusual learning rate** (`add_red_herring_lr`): An atypically high or low learning rate appears in model hyperparameters. This mimics a suspicious configuration but is not a scoreable violation — unusual hyperparameters are not in the violation taxonomy.

  2. **Early-epoch overfitting** (`add_red_herring_early_overfit`): Training logs show validation loss diverging from training loss in early epochs before recovering. Agents may mistake this for a generalisation failure, but it resolves correctly and does not constitute a reportable violation.

  3. **Suspiciously high accuracy** (`add_red_herring_high_accuracy`): The evaluation report shows near-perfect accuracy. The experiment notes clarify this is justified by well-separated class distributions. Flagging high accuracy without evidence of leakage is a false positive.

  4. **Correct GroupShuffleSplit** (`add_red_herring_entity_grouping`): Dataset metadata indicates entity-grouped data (e.g. patient IDs), but split configuration correctly uses `GroupShuffleSplit`. This mimics V7 (entity leakage) surface without meeting the V7 criterion — V7 requires the absence of group-aware splitting.

  5. **Small test set size** (`add_red_herring_test_size`): Split configuration specifies a small test fraction (e.g. 5%) and experiment notes justify this with a rare-disease or limited-sample context. Models may suspect V4 or V8, but the small size is explicitly justified and not a violation. *Post-evaluation note (2026-04-20): the `tabular_multi_117` clean experiment uses this red herring; all tested frontier models (GPT-4.1-mini, GPT-4.1, o4-mini) submitted `verdict=reject` with no flags on this episode, confirming the red herring's adversarial effectiveness.*

  6. **Cross-validation HPO** (`add_red_herring_validation_tuning`): Experiment notes and validation strategy describe hyperparameter tuning performed on cross-validation folds. This mimics V8 (multi-test leakage) surface but the tuning is on CV folds, not the held-out test set, so V8 does not apply.

- Who performed the collection?
Collection was performed by code in the repository, authored and maintained by the benchmark team. No crowd annotation pipeline is used for pool construction.

- Over what timeframe?
N/A - A precise start and end date for dataset generation is not explicitly logged in metadata files. Generation occurs deterministically at import time whenever the pool builder executes.

## Preprocessing / Cleaning / Labeling
- What preprocessing was applied?
Template normalization, violation injection, and optional red-herring injection are applied at generation time. Artifacts are serialized into JSON-like structures and strings for environment consumption.

- Was the raw data saved alongside processed?
Base templates are preserved in the repository and act as the raw source. Generated pool instances are constructed at runtime rather than checked in as a static expanded dataset file.

- Is the software used for preprocessing available?
Yes. All preprocessing and labeling logic is available in repository code, primarily in the environment generator and grader modules.

## Uses
- Has this dataset been used for tasks other than the one it was created for?
It is primarily intended for interactive auditing benchmarks, but it can also be used for red-team stress tests of agent policies, reward-shaping research, and grader robustness analysis.

- What are the recommended uses?
Recommended uses are: benchmarking tool-using agents on integrity audits, evaluating cross-artifact reasoning under limited budgets, comparing prompting and planning strategies, and studying evidence-grounded flagging behavior.

- What uses should be avoided?
Avoid using this dataset as a direct measure of real-world epidemiology of violations, because distributions are synthetic and programmatically controlled. Avoid claiming broad scientific validity across domains not represented by the templates. Avoid using leaderboard-only optimization as proof of real audit competence without external validation.

## Distribution
- How is it distributed? (Docker container + HF Spaces)
The benchmark is distributed as source code plus containerization assets. It is designed to run locally with Python tooling or via Docker deployment to Hugging Face Spaces.

- Any IP / license restrictions?
N/A - A definitive project-wide license statement is not clearly specified in the provided files. Users should verify licensing before redistribution.

- Any export control restrictions?
N/A - No export-control documentation is present in repository materials.

## Maintenance
- Who is maintaining it?
Current maintenance is repository-driven by the benchmark team. In practice, maintainers are the contributors operating the project and challenge submissions.

- How can errors be reported?
Errors should be reported through the project repository issue workflow or the competition submission feedback channel, including experiment ID, task tier, seed, expected behavior, and observed behavior.

- Will it be updated? At what cadence?
Yes. The roadmap references a v2.0 extension for concurrency, broader modality coverage, and expanded taxonomy. Fixed release cadence is N/A - updates are expected to be milestone-based rather than calendar-based.
