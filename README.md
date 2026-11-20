# Causal inference at scale: IHDP, Jobs and a simulation

Analysis code for the course paper **"Causal Inference at Scale: Estimator Behaviour from Benchmark-Sized to Large Samples on IHDP and Jobs"** (MIT 91205 Big Data Analytics, University of Kigali Graduate School).

Author: Belyse Uwase Mahoro

## What the script does

`causal_scale_analysis.py` runs three experiments with four estimators
(difference in means, OLS adjustment, inverse propensity weighting, random-forest T-learner):

1. **Accuracy on Jobs and IHDP.** Jobs: bias against the randomised estimate. IHDP: error against the known true effect, over 10 realisations.
2. **Sample-size study.** Repeated random subsamples of IHDP and Jobs, reporting the standard deviation and RMSE of the estimates.
3. **Simulation at scale.** Simulated data with known effects at n = 1,000, 10,000 and 100,000, reporting ATE error and run time.

## Data

The four `.npz` files in this repository are the IHDP-100 and Jobs-Binary benchmark files (train and test) published by Fredrik D. Johansson at https://www.fredjo.com/ (section Software & Data):

- `ihdp_npci_1-100.train.npz`, `ihdp_npci_1-100.test.npz`
- `jobs_DW_bin.new.10.train.npz`, `jobs_DW_bin.new.10.test.npz`

They are redistributed here only so the results can be reproduced. Please cite the original sources (Hill 2011 for IHDP; LaLonde 1986 and Dehejia and Wahba 1999 for Jobs) and see the website for terms of use.

## Requirements

Python 3.10 or later with `numpy`, `pandas`, `scikit-learn`, `statsmodels` and `matplotlib`:

```bash
pip install numpy pandas scikit-learn statsmodels matplotlib
```

## How to run

From the folder containing the script and the data files:

```bash
python causal_scale_analysis.py all \
  --ihdp-file ihdp_npci_1-100.train.npz --ihdp-test-file ihdp_npci_1-100.test.npz \
  --jobs-file jobs_DW_bin.new.10.train.npz --jobs-test-file jobs_DW_bin.new.10.test.npz
```

You can also run one part at a time by replacing `all` with `ihdp`, `jobs` or `scale`.
Default settings: 200 repetitions in the sample-size study, 10 IHDP realisations, sizes 1000, 10000 and 100000 with 3 repetitions each.

In Google Colab, upload the script and the four `.npz` files, then run the command with a leading `!`.

## Output

Results are written to the `results/` folder:

| File | Content |
| --- | --- |
| `jobs_estimates.csv` | Jobs estimates and bias against the randomised benchmark |
| `jobs_sample_size_study.csv` | Variability of Jobs estimates by sample size |
| `ihdp_summary.csv`, `ihdp_replications.csv` | IHDP error summary and per-realisation results |
| `ihdp_sample_size_study.csv` | Variability of IHDP OLS estimates by sample size |
| `scale_summary.csv`, `scale_raw.csv` | Simulation error and run time by estimator and n |
| `jobs_sd_vs_n.png`, `scale_error_runtime.png` | Figures 1 and 2 of the paper |

Run times in the paper come from a free Google Colab instance (2 CPU cores, Intel Xeon 2.20 GHz, 12 GB RAM) and will differ on other hardware. Random seeds are fixed in the code.


