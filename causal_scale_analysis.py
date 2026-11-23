import argparse
import os
import time
import urllib.request

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "results"
os.makedirs(OUT, exist_ok=True)

NSW_URL = "https://users.nber.org/~rdehejia/data/nsw_dw.dta"
PSID_URL = "https://users.nber.org/~rdehejia/data/psid_controls.dta"
LALONDE_COVS = ["age", "education", "black", "hispanic", "married", "nodegree", "re74", "re75"]


# --------------------------------------------------------------------------
# Estimators
# --------------------------------------------------------------------------
def diff_in_means(y, t):
    y1, y0 = y[t == 1], y[t == 0]
    est = y1.mean() - y0.mean()
    se = np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))
    return est, se


def ols_adjusted(X, t, y):
    """Regress y on treatment + covariates; return treatment coefficient and robust SE."""
    Z = sm.add_constant(np.column_stack([t, X]))
    m = sm.OLS(y, Z).fit(cov_type="HC1")
    return float(m.params[1]), float(m.bse[1])


def _propensity(X, t, clip=0.01):
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
    ps = LogisticRegression(max_iter=2000).fit(Xs, t).predict_proba(Xs)[:, 1]
    return np.clip(ps, clip, 1 - clip)


def ipw_ate(X, t, y):
    """Normalised (Hajek) inverse-propensity-weighted ATE."""
    ps = _propensity(X, t)
    w1, w0 = t / ps, (1 - t) / (1 - ps)
    return float(np.sum(w1 * y) / np.sum(w1) - np.sum(w0 * y) / np.sum(w0))


def ipw_att(X, t, y):
    """IPW estimate of the effect on the treated (the usual Lalonde target)."""
    ps = _propensity(X, t)
    w0 = (1 - t) * ps / (1 - ps)
    return float(y[t == 1].mean() - np.sum(w0 * y) / np.sum(w0))


def t_learner_cate(X, t, y, n_trees=50, seed=0):
    """Random-forest T-learner: one forest per arm, CATE = difference of predictions."""
    kw = dict(n_estimators=n_trees, min_samples_leaf=5, n_jobs=-1, random_state=seed)
    m1 = RandomForestRegressor(**kw).fit(X[t == 1], y[t == 1])
    m0 = RandomForestRegressor(**kw).fit(X[t == 0], y[t == 0])
    return m1.predict(X) - m0.predict(X)


def timed(fn, *a, **k):
    s = time.perf_counter()
    r = fn(*a, **k)
    return r, time.perf_counter() - s


# --------------------------------------------------------------------------
# Lalonde
# --------------------------------------------------------------------------
def load_lalonde(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    frames = []
    for name, url in [("nsw_dw.dta", NSW_URL), ("psid_controls.dta", PSID_URL)]:
        path = os.path.join(data_dir, name)
        if not os.path.exists(path):
            try:
                print(f"Downloading {url}")
                urllib.request.urlretrieve(url, path)
            except Exception as e:  # noqa
                raise SystemExit(
                    f"Could not download {url} ({e}).\n"
                    f"Download it in your browser and save it as {path}"
                )
        frames.append(pd.read_stata(path))
    return frames


def strat_sample(df, frac, seed):
    return df.groupby("treat", group_keys=False).sample(frac=frac, random_state=seed)


def sample_size_study(df, estimators, fractions, reps, benchmark, y_col="re78", covs=None):
    covs = covs or LALONDE_COVS
    rows = []
    for f in fractions:
        for name, fn in estimators.items():
            ests = []
            for r in range(reps):
                s = strat_sample(df, f, seed=1000 * r + int(f * 100))
                ests.append(fn(s[covs].values.astype(float),
                               s["treat"].values.astype(int),
                               s[y_col].values.astype(float)))
            ests = np.array(ests)
            rows.append(dict(fraction=f, n=int(round(len(df) * f)), estimator=name,
                             mean=ests.mean(), sd=ests.std(ddof=1),
                             rmse_vs_benchmark=np.sqrt(np.mean((ests - benchmark) ** 2))))
    return pd.DataFrame(rows)


def plot_sd(study, title, fname):
    plt.figure(figsize=(6, 4))
    for name, g in study.groupby("estimator"):
        plt.plot(g["n"], g["sd"], marker="o", label=name)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Sample size n")
    plt.ylabel("SD of estimates across subsamples")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, fname), dpi=200)
    plt.close()


def run_lalonde(data_dir, reps):
    nsw, psid = load_lalonde(data_dir)
    psid = psid.copy()
    psid["treat"] = 0
    print(f"NSW experimental sample: {len(nsw)} rows "
          f"({int(nsw.treat.sum())} treated); PSID controls: {len(psid)} rows")

    y, t = nsw["re78"].values.astype(float), nsw["treat"].values.astype(int)
    X = nsw[LALONDE_COVS].values.astype(float)
    rows = []
    est, se = diff_in_means(y, t)
    rows.append(["Experimental: difference in means", est, se])
    benchmark = est
    est, se = ols_adjusted(X, t, y)
    rows.append(["Experimental: OLS adjusted", est, se])

    obs = pd.concat([nsw[nsw.treat == 1], psid], ignore_index=True)
    yo, to = obs["re78"].values.astype(float), obs["treat"].values.astype(int)
    Xo = obs[LALONDE_COVS].values.astype(float)
    est, se = diff_in_means(yo, to)
    rows.append(["Observational (NSW treated + PSID): naive difference", est, se])
    est, se = ols_adjusted(Xo, to, yo)
    rows.append(["Observational: OLS adjusted", est, se])
    rows.append(["Observational: IPW (ATT)", ipw_att(Xo, to, yo), np.nan])
    cate, secs = timed(t_learner_cate, Xo, to, yo)
    rows.append([f"Observational: RF T-learner (ATT, {secs:.1f}s)", float(cate[to == 1].mean()), np.nan])

    res = pd.DataFrame(rows, columns=["estimate", "value_usd", "std_error"])
    res["bias_vs_experimental_benchmark"] = res["value_usd"] - benchmark
    res.to_csv(os.path.join(OUT, "lalonde_estimates.csv"), index=False)
    print("\nLALONDE ESTIMATES (effect of job training on 1978 earnings)")
    print(res.round(1).to_string(index=False))

    fr = [0.1, 0.2, 0.4, 0.6, 0.8]
    exp_df = nsw[["treat", "re78"] + LALONDE_COVS]
    s1 = sample_size_study(exp_df, {"Diff. in means (experimental)":
                                    lambda X_, t_, y_: diff_in_means(y_, t_)[0]},
                           fr, reps, benchmark)
    obs_df = obs[["treat", "re78"] + LALONDE_COVS]
    s2 = sample_size_study(obs_df, {"OLS (observational)": lambda X_, t_, y_: ols_adjusted(X_, t_, y_)[0],
                                    "IPW-ATT (observational)": ipw_att}, fr, reps, benchmark)
    study = pd.concat([s1, s2], ignore_index=True)
    study.to_csv(os.path.join(OUT, "lalonde_sample_size_study.csv"), index=False)
    plot_sd(study, "Lalonde: estimate noise vs sample size", "lalonde_sd_vs_n.png")
    print("\nSAMPLE-SIZE STUDY")
    print(study.round(1).to_string(index=False))


# --------------------------------------------------------------------------
# IHDP
# --------------------------------------------------------------------------
def load_npz(path, test_path=None, keys=("x", "t", "yf", "mu0", "mu1")):
    """Load a fredjo.com .npz file; if a test file is given, stack train + test units."""
    d = np.load(path)
    out = {k: d[k] for k in keys if k in d.files}
    print(f"{os.path.basename(path)}: keys = {d.files}, "
          f"x shape = {d['x'].shape}")
    if test_path:
        d2 = np.load(test_path)
        for k in list(out):
            out[k] = np.concatenate([out[k], d2[k]], axis=0)
        print(f"{os.path.basename(test_path)}: x shape = {d2['x'].shape} "
              f"-> combined units = {out['x'].shape[0]}")
    return out


def run_ihdp(path, n_reps, sub_reps, test_path=None):
    d = load_npz(path, test_path)
    X_all, T_all, YF = d["x"], d["t"], d["yf"]
    mu0, mu1 = d["mu0"], d["mu1"]
    R = min(n_reps, X_all.shape[2])
    rows = []
    for r in range(R):
        X = X_all[:, :, r].astype(float)
        t = T_all[:, r].astype(int)
        y = YF[:, r].astype(float)
        tau = mu1[:, r] - mu0[:, r]
        cate, secs = timed(t_learner_cate, X, t, y)
        rows.append(dict(rep=r, n=len(y), true_ate=tau.mean(),
                         naive=diff_in_means(y, t)[0],
                         ols=ols_adjusted(X, t, y)[0],
                         ipw=ipw_ate(X, t, y),
                         rf_tlearner=cate.mean(),
                         rf_pehe=float(np.sqrt(np.mean((cate - tau) ** 2))),
                         rf_seconds=secs))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "ihdp_replications.csv"), index=False)
    summ = []
    for est in ["naive", "ols", "ipw", "rf_tlearner"]:
        err = df[est] - df["true_ate"]
        summ.append(dict(estimator=est, mean_abs_ate_error=err.abs().mean(),
                         sd_of_error=err.std(ddof=1)))
    summ = pd.DataFrame(summ)
    summ.to_csv(os.path.join(OUT, "ihdp_summary.csv"), index=False)
    print(f"\nIHDP: n = {int(df.n.iloc[0])} units, {R} replications")
    print(summ.round(3).to_string(index=False))
    print(f"RF T-learner mean sqrt-PEHE: {df.rf_pehe.mean():.3f}")

    # sample-size study on replication 0
    X = X_all[:, :, 0].astype(float)
    t = T_all[:, 0].astype(int)
    y = YF[:, 0].astype(float)
    true = (mu1[:, 0] - mu0[:, 0]).mean()
    rng = np.random.default_rng(0)
    rows = []
    for f in [0.2, 0.4, 0.6, 0.8]:
        m = int(len(y) * f)
        errs = []
        for _ in range(sub_reps):
            idx = rng.choice(len(y), size=m, replace=False)
            if min(t[idx].sum(), (1 - t[idx]).sum()) < 5:
                continue
            errs.append(ols_adjusted(X[idx], t[idx], y[idx])[0] - true)
        errs = np.array(errs)
        rows.append(dict(n=m, sd_of_ols_error=errs.std(ddof=1), rmse=np.sqrt(np.mean(errs ** 2))))
    study = pd.DataFrame(rows)
    study.to_csv(os.path.join(OUT, "ihdp_sample_size_study.csv"), index=False)
    print("\nIHDP sample-size study (OLS ATE error vs true ATE)")
    print(study.round(3).to_string(index=False))


# --------------------------------------------------------------------------
# Jobs-Binary (fredjo.com version of Lalonde / NSW + PSID)
# --------------------------------------------------------------------------
def run_jobs(path, test_path, reps, realization=0):
    d = load_npz(path, test_path, keys=("x", "t", "yf", "e"))
    x, t, y, e = d["x"], d["t"], d["yf"], d["e"]
    if x.ndim == 2:
        x, t, y, e = x[:, :, None], t[:, None], y[:, None], e[:, None]
    r = min(realization, x.shape[2] - 1)
    X = x[:, :, r].astype(float)
    t = t[:, r].astype(int).ravel()
    y = y[:, r].astype(float).ravel()
    e = e[:, r].astype(int).ravel()
    covs = [f"x{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=covs)
    df["treat"], df["y"], df["e"] = t, y, e
    print(f"Jobs-Binary realization {r}: {len(df)} units, {int(t.sum())} treated, "
          f"{int((e == 1).sum())} in the randomized sample, {int((e == 0).sum())} observational controls")

    exp = df[df.e == 1]
    rows = []
    est, se = diff_in_means(exp["y"].values, exp["treat"].values)
    rows.append(["Randomized sample: difference in means", est, se])
    benchmark = est
    est, se = ols_adjusted(exp[covs].values, exp["treat"].values, exp["y"].values)
    rows.append(["Randomized sample: OLS adjusted", est, se])

    obs = df[(df.treat == 1) | (df.e == 0)]
    Xo, to, yo = obs[covs].values, obs["treat"].values, obs["y"].values
    est, se = diff_in_means(yo, to)
    rows.append(["Observational (treated + PSID controls): naive difference", est, se])
    est, se = ols_adjusted(Xo, to, yo)
    rows.append(["Observational: OLS adjusted", est, se])
    rows.append(["Observational: IPW (ATT)", ipw_att(Xo, to, yo), np.nan])
    cate, secs = timed(t_learner_cate, Xo, to, yo)
    rows.append([f"Observational: RF T-learner (ATT, {secs:.1f}s)", float(cate[to == 1].mean()), np.nan])
    res = pd.DataFrame(rows, columns=["estimate", "value", "std_error"])
    res["bias_vs_randomized_benchmark"] = res["value"] - benchmark
    res.to_csv(os.path.join(OUT, "jobs_estimates.csv"), index=False)
    print("\nJOBS-BINARY ESTIMATES (effect of job training; outcome as coded in the file)")
    print(res.round(3).to_string(index=False))

    fr = [0.1, 0.2, 0.4, 0.6, 0.8]
    s1 = sample_size_study(exp[["treat", "y"] + covs],
                           {"Diff. in means (randomized)": lambda X_, t_, y_: diff_in_means(y_, t_)[0]},
                           fr, reps, benchmark, y_col="y", covs=covs)
    s2 = sample_size_study(obs[["treat", "y"] + covs],
                           {"OLS (observational)": lambda X_, t_, y_: ols_adjusted(X_, t_, y_)[0],
                            "IPW-ATT (observational)": ipw_att},
                           fr, reps, benchmark, y_col="y", covs=covs)
    study = pd.concat([s1, s2], ignore_index=True)
    study.to_csv(os.path.join(OUT, "jobs_sample_size_study.csv"), index=False)
    plot_sd(study, "Jobs-Binary: estimate noise vs sample size", "jobs_sd_vs_n.png")
    print("\nJOBS SAMPLE-SIZE STUDY")
    print(study.round(3).to_string(index=False))


# --------------------------------------------------------------------------
# Synthetic scaling study (known truth)
# --------------------------------------------------------------------------
def simulate(n, d=25, seed=0):
    r = np.random.default_rng(seed)
    X = r.normal(size=(n, d))
    ps = 1 / (1 + np.exp(-(0.5 * X[:, 0] - 0.5 * X[:, 1] + 0.3 * X[:, 2])))
    t = r.binomial(1, ps)
    tau = 1.0 + 0.5 * X[:, 0] - 0.5 * (X[:, 1] > 0)
    mu0 = X[:, 0] + 0.5 * X[:, 1] ** 2 + 0.3 * X[:, 2]
    y = mu0 + t * tau + r.normal(size=n)
    return X, t, y, tau


def run_scale(sizes, reps, forest_max_n):
    rows = []
    for n in sizes:
        for r in range(reps):
            X, t, y, tau = simulate(n, seed=r)
            true = tau.mean()
            for name, fn in [("naive diff", lambda: diff_in_means(y, t)[0]),
                             ("OLS adjusted", lambda: ols_adjusted(X, t, y)[0]),
                             ("IPW (logistic)", lambda: ipw_ate(X, t, y))]:
                est, secs = timed(fn)
                rows.append(dict(n=n, rep=r, estimator=name, ate_error=est - true, seconds=secs))
            if n <= forest_max_n:
                cate, secs = timed(t_learner_cate, X, t, y, 30)
                rows.append(dict(n=n, rep=r, estimator="RF T-learner", ate_error=cate.mean() - true,
                                 seconds=secs, sqrt_pehe=float(np.sqrt(np.mean((cate - tau) ** 2)))))
        print(f"finished n = {n}")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "scale_raw.csv"), index=False)
    summ = (df.groupby(["estimator", "n"])
              .agg(mean_abs_error=("ate_error", lambda s: s.abs().mean()),
                   mean_seconds=("seconds", "mean"))
              .reset_index())
    summ.to_csv(os.path.join(OUT, "scale_summary.csv"), index=False)
    print("\nSYNTHETIC SCALING STUDY")
    print(summ.round(4).to_string(index=False))

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for name, g in summ.groupby("estimator"):
        ax[0].plot(g["n"], g["mean_abs_error"], marker="o", label=name)
        ax[1].plot(g["n"], g["mean_seconds"], marker="o", label=name)
    for a, yl in zip(ax, ["Mean absolute ATE error", "Runtime (seconds)"]):
        a.set_xscale("log")
        a.set_yscale("log")
        a.set_xlabel("Sample size n")
        a.set_ylabel(yl)
        a.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "scale_error_runtime.png"), dpi=200)
    plt.close()


# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("task", choices=["lalonde", "jobs", "ihdp", "scale", "all"])
    p.add_argument("--data-dir", default="data")
    p.add_argument("--ihdp-file", default="data/ihdp_npci_1-100.train.npz")
    p.add_argument("--ihdp-test-file", default=None, help="optional: stack the test units onto the train units")
    p.add_argument("--jobs-file", default="data/jobs_DW_bin.new.10.train.npz")
    p.add_argument("--jobs-test-file", default=None)
    p.add_argument("--reps", type=int, default=200, help="subsamples per size (Lalonde / IHDP)")
    p.add_argument("--ihdp-replications", type=int, default=10)
    p.add_argument("--sizes", type=int, nargs="+", default=[1000, 10000, 100000])
    p.add_argument("--scale-reps", type=int, default=3)
    p.add_argument("--forest-max-n", type=int, default=100000)
    a = p.parse_args()

    if a.task in ("lalonde", "all"):
        run_lalonde(a.data_dir, a.reps)
    if a.task in ("jobs", "all"):
        if os.path.exists(a.jobs_file):
            run_jobs(a.jobs_file, a.jobs_test_file, a.reps)
        elif a.task == "jobs":
            raise SystemExit(f"Jobs-Binary file not found: {a.jobs_file}")
        else:
            print(f"(skipping Jobs-Binary: {a.jobs_file} not found)")
    if a.task in ("ihdp", "all"):
        if not os.path.exists(a.ihdp_file):
            raise SystemExit(f"IHDP file not found: {a.ihdp_file}")
        run_ihdp(a.ihdp_file, a.ihdp_replications, a.reps, a.ihdp_test_file)
    if a.task in ("scale", "all"):
        run_scale(a.sizes, a.scale_reps, a.forest_max_n)
    print(f"\nDone. Files saved in ./{OUT}/")


if __name__ == "__main__":
    main()
