# Kaggle 2-GPU DDP submission package

This revised package is designed for the ML System Optimization assignment and
emphasizes three grading priorities:

1. Rigour of justification
2. Quality of measurements
3. Depth of analysis

## Files
- `train_ddp_kaggle.py` — documented training implementation
- `analyze_results.py` — computes rubric-aligned comparison metrics
- `plot_results.py` — generates report-ready figures
- `submission_notes.md` — checklist for final submission

## Recommended workflow
1. Run 1-GPU baseline.
2. Run 2-GPU DDP experiment.
3. Verify that both runs produced:
   - `epoch_metrics.csv`
   - `summary.json`
4. Run `analyze_results.py`.
5. Run `plot_results.py`.
6. Insert the generated figures into the report.

## Important measurement note
The communication fraction reported by this code is a communication-dominated
proxy measured from the post-backward tail of each step. It is useful for
comparative analysis, but it is not a pure NCCL-only timer.
