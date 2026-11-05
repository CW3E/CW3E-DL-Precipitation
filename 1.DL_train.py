"""Train CSGD precipitation post-processing models using leave-one-month-out validation."""

import os
import math

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras import layers


# ============================================================
# Configuration
# ============================================================

lead_times = [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72,
              78, 84, 90, 96, 102, 108, 114, 120, 126, 132,
              138, 144, 150, 156, 162, 168]
lead_time = lead_times[0]

data_path = (
    f"/cw3e/mead/projects/cwp190/projects/post-processing/"
    f"output/v16.12/lead{lead_time}_all_dataset_minimal.npz"
)
out_root = "/cw3e/mead/projects/cwp190/projects/post-processing/output/v17/v17.0/lead_time_" + str(lead_time) + "/"
os.makedirs(out_root, exist_ok=True)

LOCID_COL = 8
batch_size = 8192
max_epochs = 100
patience = 5
lr = 1e-3
seed = 42

np.random.seed(seed)
tf.random.set_seed(seed)


# ============================================================
# Model and loss function
# ============================================================

def extract_year_month(predict_time_int):
    pt = np.asarray(predict_time_int, dtype=np.int64)
    yr = pt // 1000000
    mo = (pt % 1000000) // 10000
    return yr, mo


def crps_cens(y_true, y_pred):
    shift2 = y_pred[:, 0]
    shift = -shift2 - 0.00049
    mu = y_pred[:, 1] + 0.0005
    sigma = y_pred[:, 2] + 0.0182
    obs = y_true[:, 0]

    shape = K.square(mu / sigma)
    scale = K.square(sigma) / mu
    c_bar = (-1.0 * shift) / scale
    y_bar = (obs - shift) / scale

    F_k_c = tf.math.igamma(shape, 1.0 * c_bar)
    F_kp1_c = tf.math.igamma(shape + 1.0, 1.0 * c_bar)
    F_2k_2c = tf.math.igamma(2.0 * shape, 2.0 * c_bar)
    B_05_kp05 = tf.vectorized_map(
        lambda x: K.exp(tf.math.lbeta([0.5, x])), shape + 0.5
    )
    F_k_y = tf.math.igamma(shape, 1.0 * y_bar)
    F_kp1_y = tf.math.igamma(shape + 1.0, 1.0 * y_bar)

    c1 = y_bar * (2.0 * F_k_y - 1.0)
    c2 = shape * (2.0 * F_kp1_y - 1.0 + K.square(F_k_c) - 2.0 * F_kp1_c * F_k_c)
    c3 = c_bar * K.square(F_k_c)
    c4 = (shape / float(math.pi)) * B_05_kp05 * (1.0 - F_2k_2c)

    CRPS = (c1 - c2 - c3 - c4) * scale
    return K.mean(CRPS)


def build_model(n_locations):
    loc_input = keras.Input(shape=(), dtype=tf.int32, name="loc_input")
    num_input = keras.Input(shape=(8,), dtype=tf.float32, name="num_input")

    emb = layers.Embedding(
        input_dim=n_locations, output_dim=25, name="location_embedding"
    )(loc_input)
    emb = layers.Flatten()(emb)

    x = layers.Concatenate()([emb, num_input])
    x = layers.Dense(64, activation="softplus", name="fc1")(x)
    x = layers.Dense(32, activation="softplus", name="fc2")(x)
    out = layers.Dense(3, activation="softplus", name="out")(x)

    model = keras.Model(inputs=[loc_input, num_input], outputs=out)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=crps_cens,
    )
    return model


class MySequence(keras.utils.Sequence):
    def __init__(self, X_num, X_loc, y, batch_size=8192, shuffle=False, seed=42):
        self.X_num = X_num
        self.X_loc = X_loc
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.y))
        self.rng = np.random.default_rng(seed)
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.indices) / self.batch_size)

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        return (
            {"loc_input": self.X_loc[batch_idx],
             "num_input": self.X_num[batch_idx]},
            self.y[batch_idx],
        )

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.indices)


# ============================================================
# Data loading
# ============================================================

data = np.load(data_path, allow_pickle=True)
feature = data["features"]
y_all = data["y"].astype(np.float32)
pt_all = data["predict_time"]
lid_all = feature[:, LOCID_COL].astype(np.int64)

X_num_all = feature[:, :8].astype(np.float32)
X_loc_all = (lid_all - 1).astype(np.int64)

if y_all.ndim == 1:
    y_all = y_all[:, None]

n_locations = int(X_loc_all.max()) + 1
yr_all, mo_all = extract_year_month(pt_all)
unique_yms = sorted(set(zip(yr_all, mo_all)))
n_months = len(unique_yms)


# ============================================================
# Leave-one-month-out cross-validation
# ============================================================

month_pred_files = []

for fold_idx, (yr, mo) in enumerate(unique_yms):
    fold_tag = f"{yr}_{mo:02d}"
    pred_file = os.path.join(out_root, f"predictions_{fold_tag}.npz")

    test_mask = (yr_all == yr) & (mo_all == mo)
    trainval_mask = ~test_mask
    trainval_idx = np.where(trainval_mask)[0]
    rng_fold = np.random.default_rng(seed + fold_idx)
    val_size = int(0.1 * len(trainval_idx))
    val_idx = rng_fold.choice(len(trainval_idx), val_size, replace=False)
    train_idx = np.setdiff1d(np.arange(len(trainval_idx)), val_idx)

    train_global = trainval_idx[train_idx]
    val_global = trainval_idx[val_idx]
    test_global = np.where(test_mask)[0]

    train_seq = MySequence(
        X_num_all[train_global], X_loc_all[train_global],
        y_all[train_global], batch_size=batch_size, shuffle=True, seed=seed + fold_idx,
    )
    val_seq = MySequence(
        X_num_all[val_global], X_loc_all[val_global],
        y_all[val_global], batch_size=batch_size, shuffle=False,
    )
    test_seq = MySequence(
        X_num_all[test_global], X_loc_all[test_global],
        y_all[test_global], batch_size=batch_size, shuffle=False,
    )

    best_model_path = os.path.join(out_root, f"best_model_{fold_tag}.keras")
    loss_csv_path = os.path.join(out_root, f"loss_log_{fold_tag}.csv")

    keras.backend.clear_session()
    tf.random.set_seed(seed + fold_idx)
    model = build_model(n_locations)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience,
            restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=best_model_path, monitor="val_loss",
            save_best_only=True, verbose=1,
        ),
        keras.callbacks.CSVLogger(filename=loss_csv_path, append=False),
    ]

    history = model.fit(
        train_seq, validation_data=val_seq,
        epochs=max_epochs, callbacks=callbacks, verbose=1,
    )

    best_model = keras.models.load_model(
        best_model_path, custom_objects={"crps_cens": crps_cens}
    )
    preds_raw = best_model.predict(test_seq, verbose=1)

    shift2 = preds_raw[:, 0]
    shift = -shift2 - 0.00049
    mu = preds_raw[:, 1] + 0.0005
    sigma = preds_raw[:, 2] + 0.0182
    preds_transformed = np.stack([shift, mu, sigma], axis=1)

    np.savez_compressed(
        pred_file,
        test_predictions=preds_raw,
        test_predictions_transformed=preds_transformed,
        predict_time=pt_all[test_global],
        location_id=lid_all[test_global],
        y=y_all[test_global, 0].astype(np.float32),
    )
    month_pred_files.append(pred_file)

    del model, best_model, preds_raw, preds_transformed
    del train_seq, val_seq, test_seq
    keras.backend.clear_session()


# ============================================================
# Merge cross-validated predictions
# ============================================================

all_raw = []
all_transformed = []
all_predict_time = []
all_location_id = []
all_y = []

for fpath in month_pred_files:
    d = np.load(fpath, allow_pickle=True)
    all_raw.append(d["test_predictions"])
    all_transformed.append(d["test_predictions_transformed"])
    all_predict_time.append(d["predict_time"])
    all_location_id.append(d["location_id"])
    all_y.append(d["y"])

all_raw = np.concatenate(all_raw, axis=0)
all_transformed = np.concatenate(all_transformed, axis=0)
all_predict_time = np.concatenate(all_predict_time, axis=0)
all_location_id = np.concatenate(all_location_id, axis=0)
all_y = np.concatenate(all_y, axis=0)

sort_order = np.argsort(all_predict_time, kind="stable")
all_raw = all_raw[sort_order]
all_transformed = all_transformed[sort_order]
all_predict_time = all_predict_time[sort_order]
all_location_id = all_location_id[sort_order]
all_y = all_y[sort_order]

merged_path = os.path.join(out_root, "all_predictions_lead_" + str(lead_time) + ".npz")
np.savez_compressed(
    merged_path,
    test_predictions=all_raw,
    test_predictions_transformed=all_transformed,
    predict_time=all_predict_time,
    location_id=all_location_id,
    y=all_y,
)
