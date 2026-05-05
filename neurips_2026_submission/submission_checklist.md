# NeurIPS 2026 Submission Checklist — MLAuditBench (#2921)
Date prepared: 2026-05-05

## COMPLETED (automated)

### Abstract
- [x] Abstract replaced to match locked OpenReview submission text exactly

### Formatting
- [x] `\usepackage[eandd]{neurips_2026}` — `nonanonymous` removed
- [x] Author block replaced with `\author{Anonymous Authors}`
- [x] No `\date{}` command present
- [x] `\begin{ack}...\end{ack}` block added (withheld for blind review)
- [x] Reproducibility Statement paragraph added before bibliography
- [x] All tables: `booktabs` only (toprule/midrule/bottomrule); no `\hline`, no vertical rules
- [x] All table captions above table bodies
- [x] Violation taxonomy table caption updated with key takeaway
- [x] "Extensions in this release" revision paragraph replaced with "Paper organisation"
- [x] `\bibliography{references}` present
- [x] Violation distribution table added to appendix (Tab. A.1, exact counts from generator.py)

### Anonymization — paper
- [x] main_v2.tex: No author names (Aryan Jain, Shloka Tomar)
- [x] main_v2.tex: No institution (IIT Madras, Chennai, smail.iitm)
- [x] main_v2.tex: No DeltaDreamers URLs
- [x] checklist.tex: DeltaDreamers URLs replaced with anonymous placeholders
- [x] paper/main.tex deleted (contained real name/email at lines 37–40)
- [x] paper/supplement.tex deleted (superseded, no identifying info)

### Anonymization — repo files
- [x] croissant.json: Creator anonymized; URLs replaced; CC BY 4.0 license
- [x] openenv.yaml: Author anonymized
- [x] README.md: Citation, license, and identity-leaking URL anonymized
  - `aryannzzz-ml-audit-env.hf.space` → `mlauditbench-anon/ml-audit-env`
- [x] DATASHEET.md: Creator and license anonymized

### License (CC BY 4.0 everywhere)
- [x] main_v2.tex: "CC~BY~4.0" (was MIT)
- [x] checklist.tex: "CC~BY~4.0" (was MIT, two occurrences)
- [x] croissant.json: `https://creativecommons.org/licenses/by/4.0/` (was MIT URL)
- [x] README.md: "CC BY 4.0" (was MIT)
- [x] DATASHEET.md: "CC BY 4.0" (was unclear/MIT)
- [x] Zero MIT hits in any submission file (confirmed by grep)

### Content
- [x] All three v2 model rows in tab:baseline_v2 and tab:human
- [x] GPT-4.1 v2: Easy 0.9000±0.0000, Medium 0.7832±0.1854, Hard 0.7798±0.1553
- [x] o4-mini v2: Easy 0.5391±0.3980, Medium 0.5876±0.3745, Hard 0.4687±0.2689
- [x] Per-violation recall paragraph in sec:hardanalysis
- [x] V6 dominant failure mode documented
- [x] Participant count: N=10 per tier, N=30 total added to sec:human

### TODO items resolved
- [x] AgentPRM citation (luo2025pitfalls) removed — citation was unrelated; sentence revised
- [x] stratified_seeds.py reference updated → `scripts/run_evaluation.py --stratified`
- [x] Effect size TODO removed — Δ≈0.71 confirmed correct from v2 data, kept in text

### Repo hygiene
- [x] `inference.py.bak` deleted
- [x] `MLAuditBench_NeurIPS_GapAnalysis_v3 (1).docx` deleted
- [x] No other .bak, .docx, .aux files in repo root

### Files
- [x] croissant.json — full RAI fields (10 rai: fields including intendedUse, limitations)
- [x] supplementary.zip rebuilt with fixed README.md (identity-leak URL patched)
  - Includes: all_results.txt, eval_results.json, human_evaluation_kit/, DATASHEET.md,
    openenv.yaml, inference.py, README.md, requirements.txt, croissant.json
  - Identity leak check: CLEAN (no aryan/shloka/aryannzzz/DeltaDreamers/ch24b040)

---

## REQUIRES MANUAL ACTION

### PDF Compilation (CRITICAL — no pdflatex on this machine)
- [ ] Run: `pdflatex main_v2.tex && bibtex main_v2 && pdflatex main_v2.tex && pdflatex main_v2.tex`
  from the `paper/` directory on a machine with TeX Live
- [ ] Verify compiled PDF is ≤9 pages (main text) + references + checklist + appendix
- [ ] Verify line numbers appear in anonymous `[eandd]` mode
- [ ] Confirm checklist PDF renders with all 16 items answered
- [ ] Confirm violation distribution table (Appendix A) compiles correctly
- [ ] Name output `main_v2_final.pdf` and place in this folder

### Post-compilation checks
- [ ] `\ref{sec:design}`, `\ref{sec:scoring}`, `\ref{sec:eval}`, `\ref{app:pool}` — verify no broken refs in PDF
- [ ] `luo2025pitfalls` key removed from main_v2.tex — verify references.bib does not cause bibtex warnings
- [ ] `scripts/run_evaluation.py` with `--stratified` flag exists (or update text if it does not)

### Submission portal
- [ ] Upload `main_v2_final.pdf` to NeurIPS CMT portal as paper #2921
- [ ] Upload `supplementary.zip`
- [ ] Confirm `croissant.json` attached (E&D track requires separate upload)
- [ ] Verify all author information fields in CMT are filled (unblinded there)

---

## RISK ASSESSMENT

| Risk | Severity | Status |
|------|----------|--------|
| `nonanonymous` in package options | CRITICAL | RESOLVED |
| Author names in tex | CRITICAL | RESOLVED |
| main.tex with real name/email in repo | CRITICAL | RESOLVED (deleted) |
| Abstract mismatch with OpenReview | HIGH | RESOLVED |
| License mismatch (MIT vs CC BY 4.0) | HIGH | RESOLVED |
| Identity-leaking URL in README | HIGH | RESOLVED |
| DeltaDreamers URLs in submission materials | HIGH | RESOLVED |
| PDF not compiled | HIGH | **PENDING MANUAL** |
| luo2025pitfalls citation removed — bibtex warning possible | LOW | Verify on compile |
| run_evaluation.py --stratified flag may not exist | LOW | Verify before submission |
