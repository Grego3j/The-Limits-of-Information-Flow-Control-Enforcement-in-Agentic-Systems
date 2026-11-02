# agent_ifc

Python simulations and reference results for *The Limits of Information-Flow Control Enforcement in Agentic AI Systems* (AIDC'26).

## Experiments

- **0 — §5.2, Figure 2:** 500 DAGs per task size (4–20 tools), five agents, one sensitive tool.
- **1 — §5.3, Figures 3–4:** 1,000 twenty-tool DAGs, five agents, labeling budgets 4–20; Spearman correlations use budgets 4–10 and all nine physical topologies.
- **2a — §5.4, Figures 5–6:** 5–20 agents, ten sensitive tools, uniform allocation.
- **3a — §5.5, Figure 5:** paired clustered sensitive-tool allocation.

Utility loss is the fraction of high-sensitivity tool calls. Excess utility loss subtracts the task's direct-communication baseline, G_min. Security enforcement cost is represented by the total number of communication edges. Figures show utility loss as percentages; JSON stores fractions under `propagation` and `excessive_propagation`.

[View figures](index.html). Saved result snapshots retain the original full sweeps; the paper figures select the configurations above. ER probabilities 0.1, 0.2 and 0.4 are omitted from line plots for readability, but included in correlations.

## Run

Tested with Python 3.9.6. From this directory:

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python verify_package.py
python -m unittest discover -s agent_ifc/tests
python -m agent_ifc.paper_figures
```

To rerun simulations (use a separate copy to preserve reference results):

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m agent_ifc.experiments.experiment_0.run --workers 4
python -m agent_ifc.experiments.experiment_1.run --workers 4
python -m agent_ifc.experiments.experiment_2a.run --workers 4
```

The last command runs both 2a and 3a; add `--limit 1` for a one-DAG pilot. Their full budget sweeps include the ten-source paper comparison. Figure outputs use the filenames referenced by the paper's LaTeX.

**New runs:** numerical results may differ from the reference files, particularly Experiment 1, which generates new execution samples.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{jg8+aidc,
  author  = {Jonathan Gregory and Kyle Domico and Kunyang Li and Jean-Charles {Noirot Ferrand} and Patrick McDaniel},
  title   = {The Limits of Information-Flow Control Enforcement in Agentic {AI} Systems},
  booktitle = {Proceedings of the Inaugural Workshop on Agentic AI in Offensive and Defensive Cyber Operations (AIDC 2026)},
  month   = dec,
  year   = {2026},
  publisher = {IEEE}
}
```
