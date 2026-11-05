"""Schaake Shuffle reconstruction for leave-one-month-out CSGD predictions."""

import os
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================
lead_times = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72,
              78, 84, 90, 96, 102, 108, 114, 120, 126, 132,
              138, 144, 150, 156, 162, 168]
lead_time = lead_times[0]

data_path = (
    f"/cw3e/mead/projects/cwp190/projects/post-processing/"
    f"output/v16.12/lead{lead_time}_all_dataset_minimal.npz"
)
out_root  = (
    "/cw3e/mead/projects/cwp190/projects/post-processing/"
    "output/v17/v17.0/lead_time_" + str(lead_time) + "/"
)
N_MEMBERS = 200
SEED      = 42

os.makedirs(out_root, exist_ok=True)
os.makedirs(out_root+"ss/", exist_ok=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def extract_year_month(predict_time_int):
    pt = np.asarray(predict_time_int, dtype=np.int64)
    yr = pt // 1000000
    mo = (pt % 1000000) // 10000
    return yr, mo


def sample_csgd(shift, mu, sigma, n_samples, rng):
    """Sample n_samples from CSGD, censored at 0."""
    eps   = 1e-12
    mu    = float(np.maximum(mu,    eps))
    sigma = float(np.maximum(sigma, eps))
    shape = float(np.maximum((mu / sigma) ** 2, eps))
    scale = float(np.maximum((sigma ** 2) / mu,  eps))
    samples = rng.gamma(shape, scale, size=n_samples) + float(shift)
    return np.maximum(samples, 0.0).astype(np.float32)


# ============================================================
# LOAD DATA
# ============================================================
data    = np.load(data_path, allow_pickle=True)
feature = data["features"]
y_all   = data["y"].astype(np.float32)
pt_all  = data["predict_time"]
lid_all = feature[:, 8].astype(np.int64)

if y_all.ndim == 2:
    y_all = y_all[:, 0]

yr_all, mo_all = extract_year_month(pt_all)
unique_yms     = sorted(set(zip(yr_all.tolist(), mo_all.tolist())))
n_months       = len(unique_yms)

unique_loc_ids = np.unique(lid_all)
n_locs         = len(unique_loc_ids)
loc_id_to_idx  = {int(lid): i for i, lid in enumerate(unique_loc_ids)}

print(f"  Total samples      : {len(pt_all)}")
print(f"  Unique locations   : {n_locs}")
print(f"  Unique year-months : {n_months}")
for ym in unique_yms:
    cnt = np.sum((yr_all == ym[0]) & (mo_all == ym[1]))
    print(f"    {ym[0]}-{ym[1]:02d}  n_samples={cnt}")


# ============================================================
# BUILD OBSERVATION POOL
# ============================================================
unique_times = np.unique(pt_all)
n_times      = len(unique_times)
time_to_idx  = {int(t): i for i, t in enumerate(unique_times)}

obs_matrix = np.full((n_times, n_locs), np.nan, dtype=np.float32)

for i in range(len(pt_all)):
    t_idx   = time_to_idx[int(pt_all[i])]
    l_idx   = loc_id_to_idx[int(lid_all[i])]
    obs_matrix[t_idx, l_idx] = y_all[i]

# Preserve original behavior: missing observations are replaced with zero.
obs_matrix = np.where(np.isnan(obs_matrix), 0.0, obs_matrix)
print(f"  obs_matrix shape: {obs_matrix.shape}  (times x locations)")
del data, feature


# ============================================================
# SCHAAKE SHUFFLE: LEAVE-ONE-MONTH-OUT
# ============================================================
fold_ss_files = []

for fold_idx, (yr, mo) in enumerate(unique_yms):
    fold_tag  = f"{yr}_{mo:02d}"
    pred_file = os.path.join(out_root, f"predictions_{fold_tag}.npz")
    ss_file   = os.path.join(out_root+"ss/", f"schaake_shuffle_{fold_tag}.npz")
    print(f"Fold {fold_idx+1}/{n_months}: {fold_tag}")

    if not os.path.exists(pred_file):
        print(f"  [SKIP] predictions file not found: {pred_file}")
        continue

    pred              = np.load(pred_file, allow_pickle=True)
    preds_transformed = pred["test_predictions_transformed"]
    pt_test           = pred["predict_time"]
    lid_test          = pred["location_id"]
    y_test            = pred["y"]
    shift_all = preds_transformed[:, 0]
    mu_all    = preds_transformed[:, 1]
    sigma_all = preds_transformed[:, 2]

    print(f"  Loaded predictions: {len(pt_test)} samples")

    yr_times = np.array([int(t) // 1000000          for t in unique_times], dtype=np.int32)
    mo_times = np.array([(int(t) % 1000000) // 10000 for t in unique_times], dtype=np.int32)
    train_time_mask = ~((yr_times == yr) & (mo_times == mo))

    obs_pool   = obs_matrix[train_time_mask]
    n_pool     = len(obs_pool)
    print(f"  Obs pool: {obs_pool.shape}  (training times x locations)")
    assert n_pool >= N_MEMBERS, \
        f"Obs pool ({n_pool}) < N_MEMBERS ({N_MEMBERS}) for fold {fold_tag}"
    rng = np.random.default_rng(SEED + fold_idx)

    unique_test_dates = np.unique(pt_test)
    print(f"  Applying SS to {len(unique_test_dates)} test dates ...")

    all_ensemble = []
    all_pt       = []
    all_lid      = []
    all_y        = []

    for fdate in unique_test_dates:
        date_mask     = (pt_test == fdate)
        loc_ids_today = lid_test[date_mask]
        shift_today   = shift_all[date_mask]
        mu_today      = mu_all[date_mask]
        sigma_today   = sigma_all[date_mask]
        y_today       = y_test[date_mask]

        sort_order    = np.argsort(loc_ids_today)
        loc_ids_today = loc_ids_today[sort_order]
        shift_today   = shift_today[sort_order]
        mu_today      = mu_today[sort_order]
        sigma_today   = sigma_today[sort_order]
        y_today       = y_today[sort_order]
        n_locs_today  = len(loc_ids_today)

        raw_samples = np.zeros((n_locs_today, N_MEMBERS), dtype=np.float32)
        for i in range(n_locs_today):
            raw_samples[i] = sample_csgd(
                shift_today[i], mu_today[i], sigma_today[i], N_MEMBERS, rng
            )

        chosen_idx  = rng.choice(n_pool, size=N_MEMBERS, replace=False)
        loc_col_idx  = np.array([loc_id_to_idx[int(lid)] for lid in loc_ids_today])
        obs_template = obs_pool[chosen_idx][:, loc_col_idx]

        shuffled = np.zeros_like(raw_samples)
        for i in range(n_locs_today):
            obs_ranks    = np.argsort(np.argsort(obs_template[:, i]))
            model_sorted = np.sort(raw_samples[i])
            shuffled[i]  = model_sorted[obs_ranks]

        ensemble_out = shuffled.T.astype(np.float32)
        all_ensemble.append(ensemble_out)
        all_pt.append(np.full(n_locs_today, fdate, dtype=np.int64))
        all_lid.append(loc_ids_today)
        all_y.append(y_today)

    fold_ensemble = np.concatenate(all_ensemble, axis=1).astype(np.float32)
    fold_pt       = np.concatenate(all_pt)
    fold_lid      = np.concatenate(all_lid)
    fold_y        = np.concatenate(all_y)

    np.savez_compressed(
        ss_file,
        ensemble     = fold_ensemble,
        predict_time = fold_pt,
        location_id  = fold_lid,
        y            = fold_y,
    )
    print(f"  Saved: {os.path.basename(ss_file)}  ensemble={fold_ensemble.shape}")
    fold_ss_files.append(ss_file)

    del all_ensemble, fold_ensemble
    import gc; gc.collect()


# ============================================================
# MERGE FOLD PREDICTIONS
# ============================================================
all_ensemble     = []
all_predict_time = []
all_location_id  = []
all_y            = []

for fpath in fold_ss_files:
    d = np.load(fpath, allow_pickle=True)
    all_ensemble.append(d["ensemble"])
    all_predict_time.append(d["predict_time"])
    all_location_id.append(d["location_id"])
    all_y.append(d["y"])

all_ensemble     = np.concatenate(all_ensemble,     axis=1)
all_predict_time = np.concatenate(all_predict_time, axis=0)
all_location_id  = np.concatenate(all_location_id,  axis=0)
all_y            = np.concatenate(all_y,             axis=0)

sort_order       = np.argsort(all_predict_time, kind="stable")
all_ensemble     = all_ensemble[:, sort_order]
all_predict_time = all_predict_time[sort_order]
all_location_id  = all_location_id[sort_order]
all_y            = all_y[sort_order]

merged_path = os.path.join(out_root+"ss/", f"all_ss_predictions_lead_{lead_time}.npz")
np.savez_compressed(
    merged_path,
    ensemble     = all_ensemble,
    predict_time = all_predict_time,
    location_id  = all_location_id,
    y            = all_y,
)

print(f"  Total samples  : {len(all_y)}")
print(f"  Ensemble shape : {all_ensemble.shape}")
print(f"  Saved: {merged_path}")
print(f"\nDONE. All outputs saved to: {out_root}")

