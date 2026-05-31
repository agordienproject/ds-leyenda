# ============================================================
# COMPARAISON BINAIRE : Notre CNN vs fakv8 vs fakv12
# ============================================================
# Meme tache (Photo / Pas une Photo), meme loss (binary_crossentropy)
# Seule difference : taille d'entree et architecture
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score

# Rechargement des meilleurs checkpoints
model_best     = keras.models.load_model(MODEL_SAVE_PATH,   compile=False)
model_fak_best = keras.models.load_model(FAKV8_SAVE_PATH,   compile=False)
model_fak12    = keras.models.load_model(FAKV12_SAVE_PATH,  compile=False)

def binary_metrics(mdl, ds):
    """Retourne (accuracy, AUC) sur un dataset."""
    y_true_l, y_prob_l = [], []
    for xb, yb in ds:
        probs = mdl.predict(xb, verbose=0)[:, 0]
        y_prob_l.extend(probs.tolist())
        y_true_l.extend(yb.numpy().tolist())
    yt = np.array(y_true_l, dtype=int)
    yp = np.array(y_prob_l)
    acc = accuracy_score(yt, (yp >= 0.5).astype(int))
    auc = roc_auc_score(yt, yp)
    return acc, auc

acc_cnn,   auc_cnn   = binary_metrics(model_best,     test_ds)
acc_fak,   auc_fak   = binary_metrics(model_fak_best, test_ds_fak)
acc_fak12, auc_fak12 = binary_metrics(model_fak12,    test_ds_fak12)

# Tableau recapitulatif
print('=' * 78)
print(f'{"Modele":<30} {"Input":>10} {"Params":>12} {"Accuracy":>10} {"AUC":>8}')
print('-' * 78)
print(f'{"Notre CNN Binaire":<30} {"128x128":>10} {model_best.count_params():>12,} {acc_cnn*100:>9.2f}% {auc_cnn:>8.4f}')
print(f'{"fakv8 Binaire":<30} {"256x256":>10} {model_fak_best.count_params():>12,} {acc_fak*100:>9.2f}% {auc_fak:>8.4f}')
print(f'{"fakv12 Binaire":<30} {"256x256":>10} {model_fak12.count_params():>12,} {acc_fak12*100:>9.2f}% {auc_fak12:>8.4f}')
print('=' * 78)

# --- Graphiques comparatifs ---
models_names = ['Notre CNN\n(128x128)', 'fakv8\n(256x256)', 'fakv12\n(256x256)']
colors_bar   = ['#3498db', '#e74c3c', '#2ecc71']

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Comparaison Binaire — CNN vs fakv8 vs fakv12', fontsize=14, fontweight='bold')

# Accuracy
accs = [acc_cnn*100, acc_fak*100, acc_fak12*100]
axes[0].bar(models_names, accs, color=colors_bar)
for i, v in enumerate(accs):
    axes[0].text(i, v + 0.3, f'{v:.2f}%', ha='center', fontweight='bold')
axes[0].set_ylim(0, 100); axes[0].set_title('Accuracy (%)'); axes[0].set_ylabel('%')

# AUC
aucs = [auc_cnn, auc_fak, auc_fak12]
axes[1].bar(models_names, aucs, color=colors_bar)
for i, v in enumerate(aucs):
    axes[1].text(i, v + 0.005, f'{v:.4f}', ha='center', fontweight='bold')
axes[1].set_ylim(0, 1); axes[1].set_title('AUC (ROC)')

# Params
params = [m.count_params()/1e6 for m in [model_best, model_fak_best, model_fak12]]
axes[2].bar(models_names, params, color=colors_bar)
for i, v in enumerate(params):
    axes[2].text(i, v + 0.01, f'{v:.1f}M', ha='center', fontweight='bold')
axes[2].set_title('Parametres (M)')

for ax in axes:
    ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# --- Comparaison des courbes d'apprentissage ---
_histories_b = [
    (history,       'CNN Binaire',    '#3498db'),
    (history_fak,   'fakv8 Binaire',  '#e74c3c'),
    (history_fak12, 'fakv12 Binaire', '#2ecc71'),
]

fig2, axes2 = plt.subplots(1, 2, figsize=(16, 5))
fig2.suptitle("Courbes d'apprentissage -- Binaire", fontsize=14, fontweight='bold')

for hist, label, color in _histories_b:
    h      = hist.history
    epochs = range(1, len(h['loss']) + 1)
    axes2[0].plot(epochs, h['loss'],         '--', color=color, alpha=0.45, linewidth=1.2)
    axes2[0].plot(epochs, h['val_loss'],     '-',  color=color, label=label, linewidth=2)
    axes2[1].plot(epochs, h['accuracy'],     '--', color=color, alpha=0.45, linewidth=1.2)
    axes2[1].plot(epochs, h['val_accuracy'], '-',  color=color, label=label, linewidth=2)

for ax, title, ylabel in [
    (axes2[0], 'Loss     (-- train,  — val)', 'Loss'),
    (axes2[1], 'Accuracy (-- train,  — val)', 'Accuracy'),
]:
    ax.set_title(title, fontsize=12)
    ax.set_xlabel('Epoch'); ax.set_ylabel(ylabel)
    ax.legend(); ax.grid(alpha=0.3)

plt.tight_layout()
plt.show()
