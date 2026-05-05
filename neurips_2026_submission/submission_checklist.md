# NeurIPS 2026 Submission Checklist — MLAuditBench (#2921)
Date prepared: 2026-05-02

## COMPLETED (automated)

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

### Anonymization
- [x] main_v2.tex: No author names (Aryan Jain, Shloka Tomar)
- [x] main_v2.tex: No institution (IIT Madras, Chennai, smail.iitm)
- [x] main_v2.tex: No DeltaDreamers URLs
- [x] checklist.tex: DeltaDreamers URLs replaced with anonymous placeholders
- [x] croissant.json: Creator anonymized; URLs replaced with blind-review placeholders
- [x] openenv.yaml: Author anonymized
- [x] README.md: Citation and license anonymized
- [x] DATASHEET.md: Creator anonymized

### Content
- [x] Reproducibility Statement: seeds, temperature, compute cost, anonymous URL
- [x] All three v2 model rows in tab:baseline_v2 and tab:human
- [x] GPT-4.1 v2: Easy 0.9000±0.0000, Medium 0.7832±0.1854, Hard 0.7798±0.1553
- [x] o4-mini v2: Easy 0.5391±0.3980, Medium 0.5876±0.3745, Hard 0.4687±0.2689
- [x] Per-violation recall paragraph in sec:hardanalysis
- [x] V6 dominant failure mode documented

### Files
- [x] croissant.json — full RAI fields (8 rai: fields)
- [x] supplementary.zip — all_results.txt, eval_results.json, human_evaluation_kit/, DATASHEET.md, openenv.yaml, inference.py, README.md, requirements.txt, croissant.json

---

## REQUIRES MANUAL ACTION

### PDF Compilation (CRITICAL — no pdflatex on this machine)
- [ ] Run `pdflatex main_v2.tex && bibtex main_v2 && pdflatex main_v2.tex && pdflatex main_v2.tex`
- [ ] Verify compiled PDF is ≤9 pages (main text) + references + checklist + appendix
- [ ] Verify line numbers appear in anonymous mode
- [ ] Confirm checklist PDF renders with all 16 items
- [ ] Name output file `main_v2_final.pdf` and place in this folder

### Section References (verify on compile)
- [ ] `\ref{sec:design}`, `\ref{sec:scoring}`, `\ref{sec:eval}` in paper organisation paragraph — verify no broken refs
- [ ] All `\ref{sec:fixes}`, `\ref{sec:hardanalysis}` etc. resolve correctly

### Open TODOs in main_v2.tex
1. Line ~556: AgentPRM citation — add proper bibliographic entry if available
2. Line ~603: Effect size Δ≈0.71 — cross-check against v2 tier-mean avg 0.873 vs random 0.16
3. Line ~617: `stratified_seeds.py` referenced in text but not present in scripts/ — either create placeholder or update reference

### Content gaps (manual judgment required)
- [ ] Section 5.2 (Human Baseline, sec:human): verify explicit participant count per tier is stated (n=10 per tier or total N=30)
- [ ] Section 3 (sec:design): DATASHEET.md reference — add explicit sentence like "A full Gebru et al. datasheet is available in the supplementary material (DATASHEET.md)"
- [ ] Violation distribution table (experiments per V-type per archetype per tier) — user requested from generator.py; not yet created; consider adding to appendix

### Submission portal
- [ ] Upload main_v2_final.pdf to NeurIPS CMT portal as paper #2921
- [ ] Upload supplementary.zip
- [ ] Confirm croissant.json is attached (some E&D tracks require separate upload)
- [ ] Verify all author information fields in CMT are filled (unblinded there)

---

## RISK ASSESSMENT

| Risk | Severity | Status |
|------|----------|--------|
| `nonanonymous` in package options | CRITICAL | RESOLVED |
| Author names in tex | CRITICAL | RESOLVED |
| DeltaDreamers URL in submission materials | HIGH | RESOLVED |
| PDF not compiled | HIGH | **PENDING MANUAL** |
| Missing stratified_seeds.py reference | LOW | TODO in tex |
| Effect size source unclear | LOW | TODO in tex |
| Participant count not explicit | LOW | Verify on read |
