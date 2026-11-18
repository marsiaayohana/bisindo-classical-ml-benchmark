# Dataset-A experiments. One '# %%' block = one notebook cell. Run in order.

# %% CELL 0 - common module (run once; writes a_common.py next to the notebook)
COMMON = r"""
# a_common.py - shared helpers for the Dataset-A experiments. No experiment results live here.
import os, json, hashlib, platform, warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

RANDOM_STATE = 42
N_FOLDS = 5
FRAMES_PER_VIDEO = 30
SIGNERS = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7']  # anonymised signer codes, in positional order; the data file's 'signer' column must use them
DATA_CSV = os.environ.get('A_DATA_CSV', 'dataset_paper_5379_with_video(3)(2).csv')
OUT = os.environ.get('A_OUT', 'A_outputs')
os.makedirs(OUT, exist_ok=True)

# Identical to make_classical_pipeline of the exploratory notebook -------------------
def make_classical_pipeline(model_name, use_scaler=True):
    steps = [('imputer', SimpleImputer(strategy='constant', fill_value=0.0))]
    if use_scaler:
        steps.append(('scaler', StandardScaler()))
    if model_name == 'RF':
        steps.append(('clf', RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)))
    elif model_name == 'MLP':
        steps.append(('clf', MLPClassifier(hidden_layer_sizes=(256, 128), activation='relu',
                                           max_iter=500, random_state=RANDOM_STATE)))
    elif model_name == 'SVM':
        steps.append(('clf', SVC(kernel='rbf', random_state=RANDOM_STATE)))
    elif model_name == 'KNN':
        steps.append(('clf', KNeighborsClassifier(n_neighbors=5)))
    return Pipeline(steps)

SCALED = {'RF': False, 'MLP': True, 'SVM': True, 'KNN': True}   # as in original Cell 50
MODELS = ['RF', 'MLP', 'SVM', 'KNN']

def load_A():
    # Load Dataset A (5,379 rows) and validate every structural claim before any experiment.
    d = pd.read_csv(DATA_CSV)
    fc = [c for c in d.columns if c.startswith('f') and c[1:].isdigit()]
    need = {'label', 'signer', 'video', 'old_row'}            # video_no is OPTIONAL (not created or inferred here)
    assert need <= set(d.columns), f'missing columns: {need - set(d.columns)}'
    print(f'schema: {d.shape[1]} columns | 285 features + {sorted(set(d.columns) - set(fc))} | video_no present: {"video_no" in d.columns}')
    assert len(d) == 5379, f'expected 5379 rows, got {len(d)}'
    assert len(fc) == 285, f'expected 285 feature columns, got {len(fc)}'
    assert d['old_row'].is_unique, 'old_row must be unique'
    d = d.sort_values('old_row', kind='stable').reset_index(drop=True)
    assert np.isfinite(d[fc].values).all(), 'non-finite feature values'
    assert d['video'].nunique() == 210 and d['signer'].nunique() == 7 and d['label'].nunique() == 6
    v = (d['old_row'].values % 1050) // 30
    assert (d['signer'].values == np.array([SIGNERS[i // 5] for i in v])).all(), 'signer != positional rule'
    if 'video_no' in d.columns:                                   # checked only if the column really exists
        assert (d['video_no'].values == v % 5 + 1).all(), 'video_no != positional rule'
    CLASS_ORDER = ['Apa Kabar', 'Halo', 'Maaf', 'Sama-sama', 'Terima kasih', 'Tolong']   # class blocks of 1,050 rows (Cell 8 log)
    assert (d['label'].values == np.array([CLASS_ORDER[i] for i in d['old_row'].values // 1050])).all(), 'label != class block of old_row'
    blk = d['old_row'].values // 30
    assert d.groupby('video')['old_row'].apply(lambda s: len(set(s.values // 30))).max() == 1, 'video spans >1 block'
    assert pd.Series(blk).groupby(d['video'].values).first().is_unique, 'two videos share one 30-row block'
    assert d.groupby('video')['signer'].nunique().max() == 1 and d.groupby('video')['label'].nunique().max() == 1, 'video with >1 signer/label'
    assert d.groupby('video').size().max() <= FRAMES_PER_VIDEO
    le = LabelEncoder().fit(d['label'].values)
    y = le.transform(d['label'].values)
    X = d[fc].values.astype(float)
    return d, fc, le, X, y

def delta_features(d, fc):
    # Delta / delta-delta per video, frames ordered by old_row (same rule as in the exploratory notebook).
    # First retained frame of each video = 0 (delta); first two = 0 (delta-delta).
    # Differences are between consecutive RETAINED rows; filtered-out frames can create gaps (reported).
    X = d[fc].fillna(0).values.astype(float)
    delta = np.zeros_like(X); dd = np.zeros_like(X)
    gap = np.zeros(len(d), dtype=bool); has_prev = np.zeros(len(d), dtype=bool)
    orow = d['old_row'].values
    for vid, idx in d.groupby('video', sort=False).indices.items():
        idx = idx[np.argsort(orow[idx], kind='stable')]
        for i, pos in enumerate(idx):
            if i == 0:
                continue
            prev = idx[i - 1]
            has_prev[pos] = True
            gap[pos] = (orow[pos] - orow[prev]) > 1
            delta[pos] = X[pos] - X[prev]
            if i >= 2:
                dd[pos] = delta[pos] - delta[prev]
    return np.hstack([X, delta, dd]), gap, has_prev

def frame_splits(X, y, seed=RANDOM_STATE):
    return list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed).split(X, y))

def grouped_splits(X, y, groups, seed=RANDOM_STATE):
    return list(StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=seed).split(X, y, groups))

def run_cv(model, X, y, splits, rows):
    # Fit a fresh pipeline per fold. Returns (fold dataframe, out-of-fold prediction dataframe).
    fr, op = [], []
    for k, (tr, te) in enumerate(splits, 1):
        pipe = make_classical_pipeline(model, SCALED[model])
        pipe.fit(X[tr], y[tr]); p = pipe.predict(X[te])
        fr.append(dict(model=model, fold=k, n_train=len(tr), n_test=len(te),
                       accuracy=accuracy_score(y[te], p) * 100, macro_f1=f1_score(y[te], p, average='macro') * 100))
        op.append(pd.DataFrame(dict(model=model, fold=k, old_row=rows[te], y_true=y[te], y_pred=p)))
    return pd.DataFrame(fr), pd.concat(op, ignore_index=True)

EVAL_UNIT = {'A_frame_cv': 'frame rows | random stratified folds', 'A_delta_cv': 'frame rows | random stratified folds',
             'A_grouped_cv': 'frame rows | video-grouped folds'}
DL_UNIT = 'video sequences | stratified folds (DL)'

def save_tables(tag, folds, oof):
    folds = folds.copy(); folds['eval_unit'] = EVAL_UNIT[tag]
    folds.to_csv(f'{OUT}/{tag}_folds.csv', index=False)
    oof.to_csv(f'{OUT}/{tag}_oof_predictions.csv', index=False)
    keys = ['eval_unit', 'features', 'model']
    s = folds.groupby(keys, sort=False)['accuracy'].agg(
        mean='mean', sd_ddof0=lambda v: np.std(v), sd_ddof1=lambda v: np.std(v, ddof=1))
    f1 = folds.groupby(keys, sort=False)['macro_f1'].mean().rename('macro_f1_mean')
    s = s.join(f1); s.to_csv(f'{OUT}/{tag}_summary.csv'); return s

def write_environment():
    import sklearn, scipy
    env = dict(python=platform.python_version(), platform=platform.platform(), processor=platform.processor(),
               numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__, scipy=scipy.__version__,
               data_csv=DATA_CSV, data_sha256=hashlib.sha256(open(DATA_CSV, 'rb').read()).hexdigest(),
               random_state=RANDOM_STATE, n_folds=N_FOLDS)
    try:
        import tensorflow as tf; env['tensorflow'] = tf.__version__
    except Exception:
        env['tensorflow'] = 'not installed'
    json.dump(env, open(f'{OUT}/A_environment.json', 'w'), indent=2)
    return env
"""
open('a_common.py', 'w').write(COMMON)
print('a_common.py written. Set DATA_CSV in a_common.py or env var A_DATA_CSV if your file name differs.')

# %% CELL 1 - validate Dataset A and reproduce the already-known anchor values (run FIRST; stop if any check says FAIL)
import importlib, a_common; importlib.reload(a_common)
from a_common import *
d, fc, le, X, y = load_A()
env = write_environment(); print(env)
print('classes (encoder order):', list(le.classes_))
print('per-class rows:', d['label'].value_counts().to_dict())
print('frames/video: min', d.groupby('video').size().min(), 'mean', round(d.groupby('video').size().mean(), 2))

rows = d['old_row'].values
chk = []
# (a) RF frame-level 5-fold, from the earlier exploratory run: 97.95 +- 0.41
f, o = run_cv('RF', X, y, frame_splits(X, y), rows)
chk.append(('RF 5-fold mean 97.95', round(f.accuracy.mean(), 2), abs(f.accuracy.mean() - 97.95) < 0.01))
chk.append(('RF 5-fold SD(ddof0) 0.41', round(np.std(f.accuracy), 2), abs(np.std(f.accuracy) - 0.41) < 0.01))
# (b) MLP frame-level 5-fold, from the earlier exploratory run: 97.53 +- 0.49
f, o = run_cv('MLP', X, y, frame_splits(X, y), rows)
chk.append(('MLP 5-fold mean 97.53', round(f.accuracy.mean(), 2), abs(f.accuracy.mean() - 97.53) < 0.01))
# (c) RF 70:30 from the earlier exploratory run: 1572/1614 correct and this confusion matrix
REPORTED_CM = np.array([[272,0,0,4,4,1],[0,262,0,1,1,4],[1,0,248,6,1,11],
                        [0,1,0,267,0,1],[0,0,0,3,260,0],[0,0,2,0,1,263]])
tr, te = train_test_split(np.arange(len(y)), test_size=0.3, stratify=y, random_state=RANDOM_STATE)
p = make_classical_pipeline('RF', False).fit(X[tr], y[tr]).predict(X[te])
cm = confusion_matrix(y[te], p)
chk.append(('RF 70:30 correct = 1572 of 1614', f'{(p==y[te]).sum()} of {len(te)}', (p == y[te]).sum() == 1572 and len(te) == 1614))
chk.append(('RF 70:30 confusion matrix identical to the reference run', 'identical' if np.array_equal(cm, REPORTED_CM) else 'DIFFERENT', np.array_equal(cm, REPORTED_CM)))
print(pd.DataFrame(chk, columns=['anchor', 'this run', 'PASS?']).to_string(index=False))
print('\nIf any anchor FAILS, do not run the later cells: the data file or pipeline differs from the one behind Cells 170-171.')

# %% CELL 2 - frame-level stratified 5-fold, 4 classical models on the 285 frame features (Table 2, frame rows)
from a_common import *
d, fc, le, X, y = load_A(); rows = d['old_row'].values
splits = frame_splits(X, y)
F, O = [], []
for m in MODELS:
    f, o = run_cv(m, X, y, splits, rows); f.insert(0, 'features', 'frame285'); o.insert(0, 'features', 'frame285')
    F.append(f); O.append(o); print(m, round(f.accuracy.mean(), 2), '+-', round(np.std(f.accuracy), 2))
s = save_tables('A_frame_cv', pd.concat(F, ignore_index=True), pd.concat(O, ignore_index=True)); print(s.round(2))

# %% CELL 3 - delta features (855-d) and frame-level 5-fold for the 4 classical models (Table 2, delta rows)
from a_common import *
d, fc, le, X, y = load_A(); rows = d['old_row'].values
Xd, gap, has_prev = delta_features(d, fc)
assert Xd.shape == (5379, 855)
gs = dict(rows_with_a_previous_frame=int(has_prev.sum()), delta_across_filtered_gap=int(gap.sum()),
          share_of_deltas_across_gap=round(float(gap.sum() / has_prev.sum()), 4))
print('DELTA PROTOCOL: per-video, ordered by old_row, first frame = 0, never across videos.'); print(gs)
pd.Series(gs).to_csv(f'{OUT}/A_delta_gap_stats.csv')
splits = frame_splits(Xd, y)          # same seed and class layout as the frame cell
F, O = [], []
for m in MODELS:
    f, o = run_cv(m, Xd, y, splits, rows); f.insert(0, 'features', 'delta855'); o.insert(0, 'features', 'delta855')
    F.append(f); O.append(o); print(m, round(f.accuracy.mean(), 2), '+-', round(np.std(f.accuracy), 2))
s = save_tables('A_delta_cv', pd.concat(F, ignore_index=True), pd.concat(O, ignore_index=True)); print(s.round(2))

# %% CELL 4 - 80:20 and 70:30 frame-level splits, all 8 classical configurations (Table 3, Table 6, Figure 6 matrix)
from a_common import *
d, fc, le, X, y = load_A(); rows = d['old_row'].values
Xd, _, _ = delta_features(d, fc)
res, preds = [], []
for split, ts in [('80:20', 0.2), ('70:30', 0.3)]:
    tr, te = train_test_split(np.arange(len(y)), test_size=ts, stratify=y, random_state=RANDOM_STATE)
    for fs, XX in [('frame285', X), ('delta855', Xd)]:
        for m in MODELS:
            p = make_classical_pipeline(m, SCALED[m]).fit(XX[tr], y[tr]).predict(XX[te])
            cm = confusion_matrix(y[te], p, labels=range(len(le.classes_)))
            pd.DataFrame(cm, index=le.classes_, columns=le.classes_).to_csv(f'{OUT}/A_cm_{split.replace(":","")}_{fs}_{m}.csv')
            res.append(dict(split=split, features=fs, model=m, n_train=len(tr), n_test=len(te),
                            accuracy=accuracy_score(y[te], p) * 100, macro_f1=f1_score(y[te], p, average='macro') * 100,
                            errors=int((p != y[te]).sum())))
            preds.append(pd.DataFrame(dict(split=split, features=fs, model=m, old_row=rows[te], y_true=y[te], y_pred=p)))
R = pd.DataFrame(res); R.insert(0, 'eval_unit', 'frame rows | random stratified hold-out'); R.to_csv(f'{OUT}/A_split_results.csv', index=False)
pd.concat(preds).to_csv(f'{OUT}/A_split_predictions.csv', index=False)
print(R.round(2).to_string(index=False))
cm = pd.read_csv(f'{OUT}/A_cm_7030_frame285_RF.csv', index_col=0)
print('\nRF(frame) 70:30 confusion matrix:'); print(cm)
off = [(a, b, int(cm.loc[a, b])) for a in cm.index for b in cm.columns if a != b and cm.loc[a, b] > 0]
print('largest off-diagonal cells:', sorted(off, key=lambda t: -t[2])[:6])

# %% CELL 5 - video-grouped StratifiedGroupKFold(5) for the 8 classical configurations (leakage-aware benchmark)
from a_common import *
d, fc, le, X, y = load_A(); rows = d['old_row'].values
Xd, _, _ = delta_features(d, fc); groups = d['video'].values
splits = grouped_splits(X, y, groups)
for tr, te in splits:                      # hard guarantee: no video in both sides
    assert not (set(groups[tr]) & set(groups[te]))
print('videos per test fold:', [len(set(groups[te])) for _, te in splits])
F, O = [], []
for fs, XX in [('frame285', X), ('delta855', Xd)]:
    for m in MODELS:
        f, o = run_cv(m, XX, y, splits, rows); f.insert(0, 'features', fs); o.insert(0, 'features', fs)
        F.append(f); O.append(o); print(fs, m, round(f.accuracy.mean(), 2), '+-', round(np.std(f.accuracy), 2))
s = save_tables('A_grouped_cv', pd.concat(F, ignore_index=True), pd.concat(O, ignore_index=True)); print(s.round(2))
print('\nCompare with the exploratory grouped-CV run (RF frame, grouped): 94.02 +- 1.63. A difference means a different splitter seed/config; report both.')

# %% CELL 6 - 1D-CNN and LSTM on 210 video sequences: stratified 5-fold, fold-internal scaling (Table 2 DL rows)
from a_common import *

def build_sequences(d, fc, T=FRAMES_PER_VIDEO):
    """One sequence per video, frames ordered by old_row, truncated to T. Returns list of (n_i, 285) arrays."""
    orow = d['old_row'].values; Xr = d[fc].fillna(0).values.astype(float)
    seqs, labels, vids, signers = [], [], [], []
    for vid, idx in d.groupby('video', sort=False).indices.items():
        lab, sig = set(d['label'].values[idx]), set(d['signer'].values[idx])
        assert len(lab) == 1, f'video {vid} has {len(lab)} labels: {lab}'
        assert len(sig) == 1, f'video {vid} has {len(sig)} signers: {sig}'
        assert len(set(orow[idx])) == len(idx), f'video {vid} has duplicate old_row values'
        assert len(idx) <= T, f'video {vid} has {len(idx)} > {T} frames: refusing to truncate silently'
        idx = idx[np.argsort(orow[idx], kind='stable')]
        seqs.append(Xr[idx]); labels.append(lab.pop()); vids.append(vid); signers.append(sig.pop())
    return seqs, np.array(labels), np.array(vids), np.array(signers)

def scale_and_pad(fit_seqs, other_seqs, T=FRAMES_PER_VIDEO):
    """StandardScaler is fitted ONLY on the real (non-padded) frames of `fit_seqs`; every group is then transformed
    with it and zero-padded AFTER scaling (padded steps are exactly 0, i.e. the training mean in scaled space).
    NOTE: no masking is applied to the padded steps in the CNN/LSTM (see A_PROTOCOL.md, limitation L1)."""
    train_seqs = fit_seqs
    sc = StandardScaler().fit(np.vstack(train_seqs))
    def pad(sq):
        out = np.zeros((len(sq), T, sq[0].shape[1]), dtype='float32')
        for i, a in enumerate(sq):
            out[i, :len(a)] = sc.transform(a)
        return out
    return pad(train_seqs), [pad(o) for o in other_seqs]

def build_cnn1d(n_timesteps, n_features, n_classes):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, GlobalAveragePooling1D
    from tensorflow.keras.optimizers import Adam
    m = Sequential([Conv1D(64, 3, activation='relu', input_shape=(n_timesteps, n_features)), MaxPooling1D(2), Dropout(0.3),
                    Conv1D(128, 3, activation='relu'), GlobalAveragePooling1D(), Dense(64, activation='relu'),
                    Dropout(0.3), Dense(n_classes, activation='softmax')])
    m.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy']); return m

def build_lstm(n_timesteps, n_features, n_classes):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    m = Sequential([LSTM(128, input_shape=(n_timesteps, n_features)), Dropout(0.3), Dense(64, activation='relu'),
                    Dropout(0.3), Dense(n_classes, activation='softmax')])
    m.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy']); return m

def fit_predict_dl(name, tr_seqs, ytr, te_seqs, n_classes, seed):
    """Same architectures, optimiser, batch size, epochs and early-stopping as original Cells 49-50.
    DIFFERENCE (documented): the 15 % validation set is a STRATIFIED random subset of the training videos
    (the original used Keras validation_split, which takes the LAST 15 % of the training array)."""
    import tensorflow as tf
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.callbacks import EarlyStopping
    tf.keras.utils.set_random_seed(seed)
    # 1) split the training videos FIRST; 2) fit the scaler on real frames of the training subset (itr) only;
    # 3) transform train subset, validation subset and test sequences with that scaler.
    itr, iva = train_test_split(np.arange(len(ytr)), test_size=0.15, stratify=ytr, random_state=RANDOM_STATE)
    Xtr, (Xva, Xte) = scale_and_pad([tr_seqs[i] for i in itr], [[tr_seqs[i] for i in iva], te_seqs])
    ytr_in, yva = ytr[itr], ytr[iva]
    model = {'1D-CNN': build_cnn1d, 'LSTM': build_lstm}[name](Xtr.shape[1], Xtr.shape[2], n_classes)
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    h = model.fit(Xtr, to_categorical(ytr_in, n_classes), validation_data=(Xva, to_categorical(yva, n_classes)),
                  epochs=100, batch_size=16, callbacks=[es], verbose=0)
    p = model.predict(Xte, verbose=0).argmax(1); tf.keras.backend.clear_session()
    return p, len(h.history['loss'])

from a_common import *
d, fc, le, X, y = load_A()
seqs, labels, vids, signers = build_sequences(d, fc)
yv = le.transform(labels); n_classes = len(le.classes_)
nreal = np.array([len(s) for s in seqs]); print('sequences:', len(seqs), '| real frames/video: mean', nreal.mean().round(2), 'min', nreal.min(), 'max', nreal.max())
print('zero-padded timesteps (no masking in the models): %.1f %% of all 210 x 30 steps' % (100 * (1 - nreal.sum() / (len(seqs) * FRAMES_PER_VIDEO))))
pd.DataFrame(dict(video=vids, signer=signers, label=labels, n_real_frames=nreal, n_padded=FRAMES_PER_VIDEO - nreal)).to_csv(f'{OUT}/A_dl_sequence_lengths.csv', index=False)
assert len(seqs) == 210
splits = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=RANDOM_STATE).split(np.zeros(len(yv)), yv))
print('test videos per fold:', [len(te) for _, te in splits])
F, O = [], []
for name in ['1D-CNN', 'LSTM']:
    for k, (tr, te) in enumerate(splits, 1):
        p, ep = fit_predict_dl(name, [seqs[i] for i in tr], yv[tr], [seqs[i] for i in te], n_classes, seed=RANDOM_STATE + k)
        F.append(dict(features='frame285_seq30', model=name, fold=k, n_train=len(tr), n_test=len(te), epochs=ep,
                      accuracy=accuracy_score(yv[te], p) * 100, macro_f1=f1_score(yv[te], p, average='macro') * 100,
                      n_distinct_predicted_classes=len(set(p))))
        O.append(pd.DataFrame(dict(features='frame285_seq30', model=name, fold=k, video=vids[te], y_true=yv[te], y_pred=p)))
        print(name, 'fold', k, round(F[-1]['accuracy'], 2), 'epochs', ep, 'distinct preds', F[-1]['n_distinct_predicted_classes'])
F = pd.DataFrame(F); F['eval_unit'] = DL_UNIT; F.to_csv(f'{OUT}/A_dl_cv_folds.csv', index=False); pd.concat(O).to_csv(f'{OUT}/A_dl_cv_oof_predictions.csv', index=False)
print(F.groupby('model')['accuracy'].agg(mean='mean', sd_ddof0=lambda v: np.std(v), sd_ddof1=lambda v: np.std(v, ddof=1)).round(2))

# %% CELL 7 - 1D-CNN and LSTM: video-level 80:20 and 70:30 hold-out (Table 3 DL rows)
from a_common import *

def build_sequences(d, fc, T=FRAMES_PER_VIDEO):
    """One sequence per video, frames ordered by old_row, truncated to T. Returns list of (n_i, 285) arrays."""
    orow = d['old_row'].values; Xr = d[fc].fillna(0).values.astype(float)
    seqs, labels, vids, signers = [], [], [], []
    for vid, idx in d.groupby('video', sort=False).indices.items():
        lab, sig = set(d['label'].values[idx]), set(d['signer'].values[idx])
        assert len(lab) == 1, f'video {vid} has {len(lab)} labels: {lab}'
        assert len(sig) == 1, f'video {vid} has {len(sig)} signers: {sig}'
        assert len(set(orow[idx])) == len(idx), f'video {vid} has duplicate old_row values'
        assert len(idx) <= T, f'video {vid} has {len(idx)} > {T} frames: refusing to truncate silently'
        idx = idx[np.argsort(orow[idx], kind='stable')]
        seqs.append(Xr[idx]); labels.append(lab.pop()); vids.append(vid); signers.append(sig.pop())
    return seqs, np.array(labels), np.array(vids), np.array(signers)

def scale_and_pad(fit_seqs, other_seqs, T=FRAMES_PER_VIDEO):
    """StandardScaler is fitted ONLY on the real (non-padded) frames of `fit_seqs`; every group is then transformed
    with it and zero-padded AFTER scaling (padded steps are exactly 0, i.e. the training mean in scaled space).
    NOTE: no masking is applied to the padded steps in the CNN/LSTM (see A_PROTOCOL.md, limitation L1)."""
    train_seqs = fit_seqs
    sc = StandardScaler().fit(np.vstack(train_seqs))
    def pad(sq):
        out = np.zeros((len(sq), T, sq[0].shape[1]), dtype='float32')
        for i, a in enumerate(sq):
            out[i, :len(a)] = sc.transform(a)
        return out
    return pad(train_seqs), [pad(o) for o in other_seqs]

def build_cnn1d(n_timesteps, n_features, n_classes):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, GlobalAveragePooling1D
    from tensorflow.keras.optimizers import Adam
    m = Sequential([Conv1D(64, 3, activation='relu', input_shape=(n_timesteps, n_features)), MaxPooling1D(2), Dropout(0.3),
                    Conv1D(128, 3, activation='relu'), GlobalAveragePooling1D(), Dense(64, activation='relu'),
                    Dropout(0.3), Dense(n_classes, activation='softmax')])
    m.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy']); return m

def build_lstm(n_timesteps, n_features, n_classes):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    m = Sequential([LSTM(128, input_shape=(n_timesteps, n_features)), Dropout(0.3), Dense(64, activation='relu'),
                    Dropout(0.3), Dense(n_classes, activation='softmax')])
    m.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy']); return m

def fit_predict_dl(name, tr_seqs, ytr, te_seqs, n_classes, seed):
    """Same architectures, optimiser, batch size, epochs and early-stopping as original Cells 49-50.
    DIFFERENCE (documented): the 15 % validation set is a STRATIFIED random subset of the training videos
    (the original used Keras validation_split, which takes the LAST 15 % of the training array)."""
    import tensorflow as tf
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.callbacks import EarlyStopping
    tf.keras.utils.set_random_seed(seed)
    # 1) split the training videos FIRST; 2) fit the scaler on real frames of the training subset (itr) only;
    # 3) transform train subset, validation subset and test sequences with that scaler.
    itr, iva = train_test_split(np.arange(len(ytr)), test_size=0.15, stratify=ytr, random_state=RANDOM_STATE)
    Xtr, (Xva, Xte) = scale_and_pad([tr_seqs[i] for i in itr], [[tr_seqs[i] for i in iva], te_seqs])
    ytr_in, yva = ytr[itr], ytr[iva]
    model = {'1D-CNN': build_cnn1d, 'LSTM': build_lstm}[name](Xtr.shape[1], Xtr.shape[2], n_classes)
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    h = model.fit(Xtr, to_categorical(ytr_in, n_classes), validation_data=(Xva, to_categorical(yva, n_classes)),
                  epochs=100, batch_size=16, callbacks=[es], verbose=0)
    p = model.predict(Xte, verbose=0).argmax(1); tf.keras.backend.clear_session()
    return p, len(h.history['loss'])

from a_common import *
d, fc, le, X, y = load_A()
seqs, labels, vids, signers = build_sequences(d, fc); yv = le.transform(labels); n_classes = len(le.classes_)
R = []
for split, ts in [('80:20', 0.2), ('70:30', 0.3)]:
    tr, te = train_test_split(np.arange(len(yv)), test_size=ts, stratify=yv, random_state=RANDOM_STATE)
    for name in ['1D-CNN', 'LSTM']:
        p, ep = fit_predict_dl(name, [seqs[i] for i in tr], yv[tr], [seqs[i] for i in te], n_classes, seed=RANDOM_STATE)
        R.append(dict(split=split, model=name, n_train_videos=len(tr), n_test_videos=len(te), epochs=ep,
                      accuracy=accuracy_score(yv[te], p) * 100, macro_f1=f1_score(yv[te], p, average='macro') * 100,
                      n_distinct_predicted_classes=len(set(p))))
        pd.DataFrame(dict(split=split, model=name, video=vids[te], y_true=yv[te], y_pred=p)).to_csv(f'{OUT}/A_dl_split_{split.replace(":","")}_{name}_predictions.csv', index=False)
R = pd.DataFrame(R); R.insert(0, 'eval_unit', 'video sequences | stratified hold-out (DL)'); R.to_csv(f'{OUT}/A_dl_split_results.csv', index=False); print(R.round(2).to_string(index=False))

# %% CELL 8 - statistics from the saved fold scores (Table 4 replacement)
from a_common import *
from scipy.stats import wilcoxon
def load(tag):
    f = pd.read_csv(f'{OUT}/{tag}_folds.csv'); f['name'] = f['model'] + ' (' + f['features'].str.replace('285_seq30', '285').str.extract(r'^(frame|delta)')[0] + ')'
    return f
fr = pd.concat([load('A_frame_cv'), load('A_delta_cv')]); dl = pd.read_csv(f'{OUT}/A_dl_cv_folds.csv'); dl['name'] = dl['model']
allf = pd.concat([fr[['name', 'fold', 'accuracy', 'eval_unit']], dl[['name', 'fold', 'accuracy', 'eval_unit']]])
acc = {n: g.sort_values('fold')['accuracy'].values for n, g in allf.groupby('name', sort=False)}
unit = allf.groupby('name', sort=False)['eval_unit'].first().to_dict()
base = acc['RF (frame)']; rows = []
print('!! DL rows use VIDEO-level folds (42 test videos); classical rows use FRAME-level folds (~1,076 test frames).')
print('!! Their fold units differ, so any DL-vs-classical Wilcoxon/d below is DESCRIPTIVE only (column same_unit_as_baseline = False).\n')
for n, a in acc.items():
    if n == 'RF (frame)': continue
    diff = base - a; sd = diff.std(ddof=1)
    p = wilcoxon(base, a).pvalue if np.any(diff != 0) else 1.0
    rows.append(dict(model=n, eval_unit=unit[n], same_unit_as_baseline=(unit[n] == unit['RF (frame)']), mean=a.mean(), sd_ddof0=np.std(a), sd_ddof1=np.std(a, ddof=1), mean_diff=diff.mean(),
                     wilcoxon_p=p, cohens_d_paired=(diff.mean() / sd if sd > 0 else np.nan)))
T = pd.DataFrame(rows); T.to_csv(f'{OUT}/A_stats_vs_RF_frame.csv', index=False); print(T.round(4).to_string(index=False))
print('\nNOTES: (1) with n = 5 folds the smallest possible two-sided Wilcoxon p is 0.0625, so p < 0.05 is unreachable.')
print('(2) Classical folds are frame-level partitions; DL folds are video-level partitions. Pairing is by fold index only, so treat DL rows as descriptive.')
print('(3) Report which SD convention is used (ddof0 matches the original Table 2).')

# %% CELL 9 - nested CV for tuned-vs-default on Dataset A (Table 10 replacement; removes best_score_ selection bias)
from a_common import *
from sklearn.model_selection import GridSearchCV
d, fc, le, X, y = load_A(); rows = d['old_row'].values
GRIDS = {'RF': {'clf__n_estimators': [50, 100, 200], 'clf__max_depth': [None, 10, 20]},
         'MLP': {'clf__hidden_layer_sizes': [(128, 64), (256, 128)], 'clf__alpha': [0.0001, 0.001]},
         'SVM': {'clf__C': [0.1, 1, 10], 'clf__gamma': ['scale', 0.01]},
         'KNN': {'clf__n_neighbors': [3, 5, 7, 9]}}            # same grids as Cell 136 (C4)
outer = frame_splits(X, y)                                      # identical outer folds to Cell 2
inner = StratifiedKFold(3, shuffle=True, random_state=RANDOM_STATE)
R = []
for m in MODELS:
    for k, (tr, te) in enumerate(outer, 1):
        gs = GridSearchCV(make_classical_pipeline(m, SCALED[m]), GRIDS[m], cv=inner, scoring='accuracy', n_jobs=-1).fit(X[tr], y[tr])
        p = gs.predict(X[te])
        R.append(dict(model=m, fold=k, best_params=str(gs.best_params_), inner_best_score=gs.best_score_ * 100,
                      outer_accuracy=accuracy_score(y[te], p) * 100))
    print(m, 'nested outer acc', round(np.mean([r['outer_accuracy'] for r in R if r['model'] == m]), 2))
R = pd.DataFrame(R); R.to_csv(f'{OUT}/A_nested_tuning_folds.csv', index=False)
default = pd.read_csv(f'{OUT}/A_frame_cv_folds.csv').groupby('model')['accuracy'].mean()
reported = {'RF': 97.97, 'MLP': 98.27, 'SVM': 98.31, 'KNN': 96.28}   # Cell 136 best_score_ (selection-biased), as reported
S = pd.DataFrame({'default_5fold_A': default, 'tuned_best_score_cell136_reported': pd.Series(reported),
                  'tuned_nested_outer': R.groupby('model')['outer_accuracy'].mean()}).loc[MODELS]
S.to_csv(f'{OUT}/A_tuning_comparison.csv'); print(S.round(2))

# %% CELL 10 - collect everything into one manifest (send me A_outputs/ or this printout)
from a_common import *
import glob
tabs = []
for tag in ['A_frame_cv', 'A_delta_cv', 'A_grouped_cv']:
    p = f'{OUT}/{tag}_summary.csv'
    if os.path.exists(p):
        s = pd.read_csv(p); s.insert(0, 'protocol', tag); tabs.append(s)
if os.path.exists(f'{OUT}/A_dl_cv_folds.csv'):
    dl = pd.read_csv(f'{OUT}/A_dl_cv_folds.csv').groupby(['eval_unit', 'features', 'model'], sort=False)['accuracy'].agg(
        mean='mean', sd_ddof0=lambda v: np.std(v), sd_ddof1=lambda v: np.std(v, ddof=1)).reset_index(); dl.insert(0, 'protocol', 'A_dl_cv'); tabs.append(dl)
M = pd.concat(tabs, ignore_index=True); M.to_csv(f'{OUT}/A_manifest_cv_summary.csv', index=False)
for u, g in M.groupby('eval_unit', sort=False):          # never print frame-level and video-level rows in one block
    print('\n#### EVALUATION UNIT:', u); print(g.drop(columns='eval_unit').round(2).to_string(index=False))
for f in ['A_split_results.csv', 'A_dl_split_results.csv', 'A_stats_vs_RF_frame.csv', 'A_tuning_comparison.csv', 'A_delta_gap_stats.csv']:
    if os.path.exists(f'{OUT}/{f}'): print('\n==', f); print(pd.read_csv(f'{OUT}/{f}').round(3).to_string(index=False))
print('\nFiles:', sorted(os.path.basename(x) for x in glob.glob(f'{OUT}/*')))

