"""
train_cnn_raman.py
==================
Standalone Python script to reproduce the CNN classification pipeline
from cnn_raman_classification.ipynb.

Usage:
    python train_cnn_raman.py

Outputs (written to outputs/):
    model/best_model.pt          -- best checkpoint by val-accuracy
    model/final_model.pt         -- weights at end of training
    model/model_config.json      -- architecture / hyperparameter metadata
    logs/training_log.csv        -- per-epoch loss & accuracy
    logs/key_spectral_regions.csv-- top-5 wavenumber regions per class
    logs/saliency_maps.npz       -- raw Integrated-Gradient arrays
    figures/class_distribution.png
    figures/training_curves.png
    figures/confusion_matrix.png
    figures/saliency_<class>.png (one per molecular class)
    figures/saliency_heatmap_all.png
"""

import os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
from scipy.signal import find_peaks

warnings.filterwarnings('ignore')

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

for d in ('outputs/figures', 'outputs/logs', 'outputs/model'):
    os.makedirs(d, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load & prepare data
# ─────────────────────────────────────────────────────────────────────────────

def parse_list(s):
    return [float(v) for v in s.strip('[]').split(', ')]

spectra_df = pd.read_csv(
    'ramanbiolib/db/raman_spectra_db.csv',
    converters={'wavenumbers': parse_list, 'intensity': parse_list}
)
meta_df = pd.read_csv('ramanbiolib/db/metadata_db.csv')

meta_unique = meta_df[['id', 'type']].drop_duplicates(subset='id')
df = spectra_df.merge(meta_unique, on='id')
df['class'] = df['type'].str.split('/').str[0]

print('Full dataset:', df.shape)
print(df['class'].value_counts().to_string(), '\n')

# ── Class distribution plot ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
counts = df['class'].value_counts()
ax.bar(counts.index, counts.values,
       color=plt.cm.tab10(np.linspace(0, 1, len(counts))))
ax.set_xlabel('Molecular Class', fontsize=12)
ax.set_ylabel('Number of Spectra', fontsize=12)
ax.set_title('Class Distribution in ramanbiolib Spectra DB', fontsize=14)
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig('outputs/figures/class_distribution.png', dpi=150)
plt.close()
print('Saved: outputs/figures/class_distribution.png')

# ── Filter to top-6 classes ───────────────────────────────────────────────────
KEEP_CLASSES = ['Proteins', 'Lipids', 'Saccharides',
                'AminoAcids', 'PrimaryMetabolites', 'NucleicAcids']

df_filt = df[df['class'].isin(KEEP_CLASSES)].reset_index(drop=True)
print('Filtered dataset:', df_filt.shape)
print(df_filt['class'].value_counts().to_string(), '\n')

X_raw       = np.array(df_filt['intensity'].tolist(),   dtype=np.float32)
wavenumbers = np.array(df_filt['wavenumbers'].iloc[0],  dtype=np.float32)

le          = LabelEncoder()
y_raw       = le.fit_transform(df_filt['class'])
CLASS_NAMES = list(le.classes_)
N_CLASSES   = len(CLASS_NAMES)
SEQ_LEN     = X_raw.shape[1]

print(f'Spectrum length  : {SEQ_LEN} points')
print(f'Wavenumber range : {wavenumbers[0]:.0f}–{wavenumbers[-1]:.0f} cm⁻¹')
print(f'Classes ({N_CLASSES})     : {CLASS_NAMES}\n')

# ─────────────────────────────────────────────────────────────────────────────
# 2. Data augmentation
# ─────────────────────────────────────────────────────────────────────────────

AUG_FACTOR = 15

def augment_spectra(X, y, factor=AUG_FACTOR, noise_std=0.015,
                    scale_range=(0.85, 1.15)):
    X_aug, y_aug = [X.copy()], [y.copy()]
    rng = np.random.default_rng(SEED)
    for _ in range(factor):
        noise  = rng.normal(0, noise_std, X.shape).astype(np.float32)
        scales = rng.uniform(*scale_range, (X.shape[0], 1)).astype(np.float32)
        X_aug.append(np.clip(X * scales + noise, 0, None))
        y_aug.append(y.copy())
    return np.vstack(X_aug), np.concatenate(y_aug)

X_aug, y_aug = augment_spectra(X_raw, y_raw)
print(f'After augmentation: {X_aug.shape[0]} spectra\n')

# ─────────────────────────────────────────────────────────────────────────────
# 3. Dataset / DataLoader
# ─────────────────────────────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X_aug, y_aug, test_size=0.20, stratify=y_aug, random_state=SEED
)
print(f'Train: {len(X_train)}  Test: {len(X_test)}\n')


class RamanDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):  return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


train_ds = RamanDataset(X_train, y_train)
test_ds  = RamanDataset(X_test,  y_test)

class_counts = np.bincount(y_train)
weights      = 1.0 / class_counts[y_train]
sampler      = WeightedRandomSampler(weights, len(weights), replacement=True)

BATCH        = 32
train_loader = DataLoader(train_ds, batch_size=BATCH, sampler=sampler)
test_loader  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False)

# ─────────────────────────────────────────────────────────────────────────────
# 4. 1D CNN
# ─────────────────────────────────────────────────────────────────────────────

class RamanCNN1D(nn.Module):
    """Three-block 1D CNN.  Input: (batch, 1, L)."""
    def __init__(self, input_len=1351, n_classes=6):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(1,  32, 15, padding=7), nn.BatchNorm1d(32),  nn.ReLU(),
            nn.Conv1d(32, 32, 15, padding=7), nn.BatchNorm1d(32),  nn.ReLU(),
            nn.MaxPool1d(4), nn.Dropout(0.25)
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(32, 64, 11, padding=5), nn.BatchNorm1d(64),  nn.ReLU(),
            nn.Conv1d(64, 64, 11, padding=5), nn.BatchNorm1d(64),  nn.ReLU(),
            nn.MaxPool1d(4), nn.Dropout(0.25)
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(64, 128, 7, padding=3), nn.BatchNorm1d(128), nn.ReLU(),
            nn.MaxPool1d(4), nn.Dropout(0.25)
        )
        dummy = torch.zeros(1, 1, input_len)
        flat  = self._fwd(dummy).shape[1]
        self.classifier = nn.Sequential(
            nn.Linear(flat, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, n_classes)
        )

    def _fwd(self, x):
        return self.block3(self.block2(self.block1(x))).view(x.size(0), -1)

    def forward(self, x):
        return self.classifier(self._fwd(x))


model    = RamanCNN1D(input_len=SEQ_LEN, n_classes=N_CLASSES).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Model parameters: {n_params:,}\n')

# ─────────────────────────────────────────────────────────────────────────────
# 5. Training
# ─────────────────────────────────────────────────────────────────────────────

EPOCHS = 80
LR, WD = 1e-3, 1e-4

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

history      = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
best_val_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    model.train()
    t_loss = t_correct = t_total = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        logits = model(xb)
        loss   = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        t_loss    += loss.item() * len(yb)
        t_correct += (logits.argmax(1) == yb).sum().item()
        t_total   += len(yb)
    scheduler.step()

    model.eval()
    v_loss = v_correct = v_total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(xb)
            loss   = criterion(logits, yb)
            v_loss    += loss.item() * len(yb)
            v_correct += (logits.argmax(1) == yb).sum().item()
            v_total   += len(yb)

    t_acc = t_correct / t_total
    v_acc = v_correct / v_total
    history['train_loss'].append(t_loss / t_total)
    history['train_acc'].append(t_acc)
    history['val_loss'].append(v_loss / v_total)
    history['val_acc'].append(v_acc)

    if v_acc > best_val_acc:
        best_val_acc = v_acc
        torch.save(model.state_dict(), 'outputs/model/best_model.pt')

    if epoch % 10 == 0 or epoch == 1:
        print(f'Epoch {epoch:3d}/{EPOCHS}  '
              f'train_loss={t_loss/t_total:.4f} acc={t_acc:.3f}  '
              f'val_loss={v_loss/v_total:.4f} acc={v_acc:.3f}')

print(f'\nBest val acc: {best_val_acc:.4f}')
pd.DataFrame(history).to_csv('outputs/logs/training_log.csv', index=False)
print('Saved: outputs/logs/training_log.csv')

# ── Training curves ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history['train_loss'], lw=2, label='Train')
axes[0].plot(history['val_loss'],   lw=2, label='Val')
axes[0].set(xlabel='Epoch', ylabel='Loss', title='Cross-Entropy Loss')
axes[0].legend()
axes[1].plot(history['train_acc'], lw=2, label='Train')
axes[1].plot(history['val_acc'],   lw=2, label='Val')
axes[1].set(xlabel='Epoch', ylabel='Accuracy', title='Classification Accuracy')
axes[1].legend()
plt.tight_layout()
plt.savefig('outputs/figures/training_curves.png', dpi=150)
plt.close()
print('Saved: outputs/figures/training_curves.png')

# ─────────────────────────────────────────────────────────────────────────────
# 6. Evaluation
# ─────────────────────────────────────────────────────────────────────────────

model.load_state_dict(torch.load('outputs/model/best_model.pt', map_location=DEVICE))
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        logits = model(xb.to(DEVICE))
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(yb.numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)
test_acc   = accuracy_score(all_labels, all_preds)

print(f'\nTest accuracy: {test_acc:.4f}  ({test_acc*100:.1f}%)\n')
print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))

# ── Confusion matrix ──────────────────────────────────────────────────────────
cm_vals = confusion_matrix(all_labels, all_preds)
fig, ax = plt.subplots(figsize=(8, 7))
ConfusionMatrixDisplay(cm_vals, display_labels=CLASS_NAMES).plot(
    ax=ax, colorbar=False, cmap='Blues')
ax.set_title('Confusion Matrix – Test Set', fontsize=13)
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig('outputs/figures/confusion_matrix.png', dpi=150)
plt.close()
print('Saved: outputs/figures/confusion_matrix.png')

# ─────────────────────────────────────────────────────────────────────────────
# 7. Integrated Gradients
# ─────────────────────────────────────────────────────────────────────────────

def integrated_gradients(model, x, target_class, n_steps=50):
    """
    Compute Integrated Gradients attribution for a single spectrum.
    x : Tensor (1, 1, L)  on DEVICE
    Returns np.ndarray (L,)
    """
    baseline     = torch.zeros_like(x)
    alphas       = torch.linspace(0, 1, n_steps, device=DEVICE)
    interpolated = torch.stack(
        [baseline + a * (x - baseline) for a in alphas]
    ).squeeze(1)            # (n_steps, 1, L)
    interpolated.requires_grad_(True)
    logits = model(interpolated)
    logits[:, target_class].sum().backward()
    avg_grads = interpolated.grad.mean(dim=0)   # (1, L)
    ig = ((x - baseline).squeeze() * avg_grads.squeeze()).detach().cpu().numpy()
    return ig


def class_mean_saliency(model, X_cls, label, n_samples=15, n_steps=50):
    model.eval()
    igs = []
    idx = np.random.choice(len(X_cls), min(n_samples, len(X_cls)), replace=False)
    for i in idx:
        x  = torch.tensor(X_cls[i]).unsqueeze(0).unsqueeze(0).to(DEVICE)
        ig = integrated_gradients(model, x, label, n_steps=n_steps)
        igs.append(np.abs(ig))
    return np.mean(igs, axis=0)


print('\nComputing Integrated Gradient saliency maps...')
saliency_maps  = {}
class_spectra  = {}

for cls_name in CLASS_NAMES:
    cls_idx = le.transform([cls_name])[0]
    mask    = y_raw == cls_idx
    X_cls   = X_raw[mask]
    print(f'  {cls_name} ({len(X_cls)} spectra) ...', end=' ', flush=True)
    saliency_maps[cls_name] = class_mean_saliency(model, X_cls, cls_idx)
    class_spectra[cls_name] = X_cls.mean(axis=0)
    print('done')

# ─────────────────────────────────────────────────────────────────────────────
# 8. Saliency overlay plots
# ─────────────────────────────────────────────────────────────────────────────

COLORS = plt.cm.tab10(np.linspace(0, 1, N_CLASSES))

for i, cls_name in enumerate(CLASS_NAMES):
    sal   = saliency_maps[cls_name]
    spec  = class_spectra[cls_name]
    sal_n = (sal - sal.min()) / (sal.max() - sal.min() + 1e-9)

    fig, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(wavenumbers, spec, color=COLORS[i], lw=1.5, label='Mean spectrum')
    ax1.set_xlabel('Wavenumber (cm⁻¹)', fontsize=12)
    ax1.set_ylabel('Intensity (normalised)', color=COLORS[i], fontsize=12)
    ax1.tick_params(axis='y', labelcolor=COLORS[i])

    ax2 = ax1.twinx()
    ax2.fill_between(wavenumbers, sal_n, alpha=0.35, color='crimson', label='IG saliency')
    ax2.set_ylabel('Normalised |IG| saliency', color='crimson', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='crimson')

    peaks, _ = find_peaks(sal_n, prominence=0.15, distance=20)
    if len(peaks) > 0:
        top_pk = peaks[np.argsort(sal_n[peaks])[::-1][:3]]
        for pk in top_pk:
            ax1.axvline(wavenumbers[pk], color='grey', lw=0.8, ls='--')
            ax1.text(wavenumbers[pk] + 5, spec.max() * 0.9,
                     f'{wavenumbers[pk]:.0f}', fontsize=8, rotation=90)

    l1, n1 = ax1.get_legend_handles_labels()
    l2, n2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, n1 + n2, loc='upper right', fontsize=9)
    ax1.set_title(f'Saliency Map – {cls_name}', fontsize=14)
    plt.tight_layout()
    fp = f'outputs/figures/saliency_{cls_name.lower()}.png'
    plt.savefig(fp, dpi=150)
    plt.close()
    print(f'Saved: {fp}')

# ── Aggregate heatmap ─────────────────────────────────────────────────────────
sal_matrix = np.array([
    (saliency_maps[c] - saliency_maps[c].min()) /
    (saliency_maps[c].max() - saliency_maps[c].min() + 1e-9)
    for c in CLASS_NAMES
])
step  = 10
wn_ds = wavenumbers[::step]
sd_ds = sal_matrix[:, ::step]

fig, ax = plt.subplots(figsize=(14, 4))
im = ax.imshow(sd_ds, aspect='auto', cmap='hot',
               extent=[wn_ds[0], wn_ds[-1], len(CLASS_NAMES)-0.5, -0.5])
ax.set_yticks(range(len(CLASS_NAMES)))
ax.set_yticklabels(CLASS_NAMES, fontsize=11)
ax.set_xlabel('Wavenumber (cm⁻¹)', fontsize=12)
ax.set_title('Integrated-Gradient Saliency Heatmap (all classes)', fontsize=13)
plt.colorbar(im, ax=ax, label='Normalised |IG|')
plt.tight_layout()
plt.savefig('outputs/figures/saliency_heatmap_all.png', dpi=150)
plt.close()
print('Saved: outputs/figures/saliency_heatmap_all.png')

# ─────────────────────────────────────────────────────────────────────────────
# 9. Key spectral regions summary
# ─────────────────────────────────────────────────────────────────────────────

WINDOW       = 20
summary_rows = []

for cls_name in CLASS_NAMES:
    sal   = saliency_maps[cls_name]
    sal_n = (sal - sal.min()) / (sal.max() - sal.min() + 1e-9)
    peaks, _ = find_peaks(sal_n, prominence=0.10, distance=15)
    if len(peaks) == 0:
        peaks = np.array([np.argmax(sal_n)])
    ranked = peaks[np.argsort(sal_n[peaks])[::-1]]
    for pk in ranked[:5]:
        wn = wavenumbers[pk]
        summary_rows.append({
            'class': cls_name,
            'center_cm': int(wn),
            'range': f'{int(wn-WINDOW)}–{int(wn+WINDOW)} cm⁻¹',
            'saliency_score': round(float(sal_n[pk]), 4)
        })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv('outputs/logs/key_spectral_regions.csv', index=False)
print('\nKey spectral regions:')
print(summary_df.to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# 10. Save artefacts
# ─────────────────────────────────────────────────────────────────────────────

torch.save(model.state_dict(), 'outputs/model/final_model.pt')

config = {
    'input_len': SEQ_LEN,
    'n_classes': N_CLASSES,
    'class_names': CLASS_NAMES,
    'wavenumber_range': [int(wavenumbers[0]), int(wavenumbers[-1])],
    'epochs': EPOCHS,
    'best_val_acc': round(best_val_acc, 4),
    'test_acc': round(float(test_acc), 4)
}
with open('outputs/model/model_config.json', 'w') as fh:
    json.dump(config, fh, indent=2)

np.savez('outputs/logs/saliency_maps.npz',
         wavenumbers=wavenumbers,
         class_names=np.array(CLASS_NAMES),
         **{cls: saliency_maps[cls] for cls in CLASS_NAMES})

print('\n=== All artefacts saved ===')
print(f'FINAL TEST ACCURACY: {test_acc*100:.2f}%')
