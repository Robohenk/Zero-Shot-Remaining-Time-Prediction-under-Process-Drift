> [!WARNING]
> **Temporary Repository** — This repository was created to support the review process of the corresponding scientific manuscript. Upon acceptance, the codebase will be refactored and released as a permanent, fully documented repository with a DOI (e.g., via Zenodo). Until then, contents, structure, and documentation may change without notice.

# Zero-Shot Remaining-Time Prediction under Process Drift

This repository contains the code, paper sources, and exported plot points for the manuscript **"Zero-Shot Remaining-Time Prediction under Process Drift."** It supports reproducing the figures and analyses reported in the manuscript.

## Repository Structure

```text
.
├── src/                  # Scripts for predictions, metrics, and figures
├── paper/                # LaTeX manuscript sources
│   └── figures/          # PDF figures included via \includegraphics{...}
├── artifacts/
│   ├── points/           # Plot-point CSVs (run-level metrics, binned curves)
│   ├── drift/            # Drift-score and drift-vs-error CSVs (Fig. 5 inputs)
│   └── manifests/        # File manifests and checksums
└── outputs/              # Generated outputs, typically not versioned (e.g., Parquets)
```

## Reproducing Results

1. Place the dataset under `data/` (see [Dataset](#dataset) below).
2. Run the scripts in `src/` to:
   - generate predictions (if needed),
   - compute run-level metrics and binned analyses,
   - export plot-point CSVs,
   - produce the final PDF figures.
3. Compile the manuscript from `paper/`.

Exact commands are documented inline in the scripts.

## Dataset

This project uses the simulation event logs described in the paper. If the dataset cannot be redistributed here, place it under a local `data/` directory (or configure the path via script arguments).

The underlying dataset is:

> Bemthuis, R., Mes, M.R.K., Iacob, M.E., & Havinga, P.J.M. (2021). *Data underlying the paper: Using agent-based simulation for emergent behavior detection in cyber-physical systems.* 4TU.ResearchData. CC BY 4.0. [DOI: 10.4121/14743263.v1](https://doi.org/10.4121/14743263.v1)

## Environment

Install dependencies via `requirements.txt` (or `environment.yml` if provided).

| Dependency | Role |
|---|---|
| `numpy`, `pandas` | Data processing |
| `matplotlib`, `seaborn` | Plotting |
| `scipy` | Statistical utilities |
| `tensorflow` *(optional)* | Regenerating baseline model predictions |

## Note on AI-Assisted Code Polishing

Parts of the codebase and documentation were *polished* with assistance from **ChatGPT (GPT-5.2)**, primarily for refactoring, clarity, and robustness. All authors remain fully responsible for the content and results; all generated changes were reviewed and verified against the intended experimental protocol.

## License

| Component | License |
|---|---|
| Code (`src/`, `tools/`) | [MIT](LICENSE) |
| Paper sources, figures, plot points (`paper/`, `artifacts/`) | [CC BY 4.0](LICENSE-CCBY) |
| Dataset (third-party) | CC BY 4.0 — see dataset DOI above |

## Citation

If you use this repository, please cite the manuscript as follows (placeholder until published):

```bibtex
@inproceedings{ivanyi2026zeroshot,
  title     = {Zero-Shot Remaining-Time Prediction under Process Drift},
  author    = {Iványi, Zsombor and Bemthuis, Rob and Monti, Flavia and Mecella, Massimo},
  booktitle = {<Conference Name>},
  year      = {2026},
  note      = {Submitted}
}
```

**Plain text:** Z. Iványi, R. Bemthuis, F. Monti, and M. Mecella, "Zero-Shot Remaining-Time Prediction under Process Drift," in *\<Conference Name\>*, 2026. Submitted.
