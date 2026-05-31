# ============================================================
# fakv12 -- Architecture Residuelle (binaire Photo/Pas Photo)
# A coller directement dans une cellule du notebook.
# Prérequis : DATA_DIR, VAL_SPLIT, EPOCHS, PHOTO_CLASS_IDX,
#             tf, keras, layers, models, AUTOTUNE, plot_history
# ============================================================
import json as _json
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# --- Hyperparamètres ---
FAKV12_IMG_SIZE  = (256, 256)
FAKV12_BATCH     = 96
FAKV12_SAVE_PATH = './fakv12_binary_best.h5'
AUTOTUNE_FAK12   = tf.data.AUTOTUNE

# Mixed precision : ~moitie moins de VRAM + Tensor Cores RTX actifs
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# --- Pipeline de données ---
rescale_fak12  = layers.Rescaling(1.0 / 255.0)
data_aug_fak12 = keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.10),
    layers.RandomZoom(0.10),
], name='aug_fakv12')

def preprocess_fak12_train(x, y):
    x = data_aug_fak12(x, training=True)
    x = rescale_fak12(x)
    return x, y

def preprocess_fak12_eval(x, y):
    return rescale_fak12(x), y

def to_binary_fak12(x, y):
    return x, tf.cast(tf.equal(y, PHOTO_CLASS_IDX), tf.int32)

train_ds_fak12_raw = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=VAL_SPLIT, subset='training',
    seed=42, image_size=FAKV12_IMG_SIZE, batch_size=FAKV12_BATCH,
    label_mode='int', shuffle=True,
)
_vt_fak12_1 = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=VAL_SPLIT, subset='validation',
    seed=42, image_size=FAKV12_IMG_SIZE, batch_size=FAKV12_BATCH,
    label_mode='int', shuffle=True,
)
_vt_fak12_2 = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=VAL_SPLIT, subset='validation',
    seed=42, image_size=FAKV12_IMG_SIZE, batch_size=FAKV12_BATCH,
    label_mode='int', shuffle=True,
)

n_vt_fak12        = tf.data.experimental.cardinality(_vt_fak12_1).numpy()
n_val_fak12       = n_vt_fak12 // 2
val_ds_fak12_raw  = _vt_fak12_1.take(n_val_fak12)
test_ds_fak12_raw = _vt_fak12_2.skip(n_val_fak12)

train_ds_fak12 = (
    train_ds_fak12_raw
    .map(preprocess_fak12_train, num_parallel_calls=AUTOTUNE_FAK12)
    .map(to_binary_fak12,        num_parallel_calls=AUTOTUNE_FAK12)
    .prefetch(AUTOTUNE_FAK12)
)
val_ds_fak12 = (
    val_ds_fak12_raw
    .map(preprocess_fak12_eval, num_parallel_calls=AUTOTUNE_FAK12)
    .map(to_binary_fak12,       num_parallel_calls=AUTOTUNE_FAK12)
    .prefetch(AUTOTUNE_FAK12)
)
test_ds_fak12 = (
    test_ds_fak12_raw
    .map(preprocess_fak12_eval, num_parallel_calls=AUTOTUNE_FAK12)
    .map(to_binary_fak12,       num_parallel_calls=AUTOTUNE_FAK12)
    .prefetch(AUTOTUNE_FAK12)
)

print(f'Pipeline fakv12 binaire pret.')
print(f'  Train : {tf.data.experimental.cardinality(train_ds_fak12_raw)} batches')
print(f'  Val   : {n_val_fak12} batches | Test : {n_vt_fak12 - n_val_fak12} batches')

# --- Poids de classes (recalcul local pour eviter conflit de type) ---
_bin_labels = []
for _, _yb in train_ds_fak12_raw.unbatch().map(to_binary_fak12):
    _bin_labels.append(int(_yb.numpy()))
_cw = compute_class_weight('balanced', classes=np.array([0, 1]), y=_bin_labels)
class_weight_fak12 = {0: float(_cw[0]), 1: float(_cw[1])}
print(f'Poids des classes : {class_weight_fak12}')

# --- Architecture résiduelle ---
def residual_block_fak12(x, filters):
    shortcut = x
    x = layers.Conv2D(filters, (3, 3), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, (3, 3), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1, 1), padding='same', use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    x = layers.Add()([shortcut, x])
    x = layers.Activation('relu')(x)
    return x

def build_fakv12_binary(input_shape=(256, 256, 3)):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(2, 2)(x)
    x = residual_block_fak12(x, 64)
    x = layers.MaxPooling2D(2, 2)(x)
    x = residual_block_fak12(x, 128)
    x = layers.MaxPooling2D(2, 2)(x)
    x = residual_block_fak12(x, 256)
    x = layers.MaxPooling2D(2, 2)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.5)(x)
    # dtype float32 obligatoire avec mixed precision
    outputs = layers.Dense(1, activation='sigmoid', dtype='float32', name='photo_probability')(x)
    return keras.Model(inputs=inputs, outputs=outputs, name='fakv12_binary')

model_fak12 = build_fakv12_binary(input_shape=(*FAKV12_IMG_SIZE, 3))
model_fak12.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        keras.metrics.AUC(name='auc'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
    ],
)
model_fak12.summary(line_length=80)
print(f'Parametres : {model_fak12.count_params():,}')

# --- Callbacks ---
callbacks_fak12 = [
    keras.callbacks.ModelCheckpoint(
        filepath=FAKV12_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        save_format='h5',
        verbose=1,
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.5,
        patience=4,
        min_lr=1e-6,
        verbose=1,
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=8,
        restore_best_weights=True,
        verbose=1,
    ),
]

# --- Entraînement ---
print("Entrainement fakv12 (binaire Photo/Pas Photo)...")
print(f'  Sauvegarde : {FAKV12_SAVE_PATH}')
history_fak12 = model_fak12.fit(
    train_ds_fak12,
    epochs=EPOCHS,
    validation_data=val_ds_fak12,
    class_weight=class_weight_fak12,
    callbacks=callbacks_fak12,
    verbose=1,
)

# --- Sauvegarde historique JSON ---
_HIST_FAK12_PATH = './history_fak12.json'
_hist_fak12 = {k: [float(v) for v in vals] for k, vals in history_fak12.history.items()}
with open(_HIST_FAK12_PATH, 'w', encoding='utf-8') as _f:
    _json.dump(_hist_fak12, _f, indent=1)
_n_ep12     = len(next(iter(_hist_fak12.values())))
_best_acc12 = max(_hist_fak12.get('val_accuracy', [0]))
print(f'Historique fakv12 Binaire sauvegarde : {_HIST_FAK12_PATH}')
print(f'  {_n_ep12} epochs | meilleure val_accuracy : {_best_acc12:.4f}')

# --- Courbes ---
print("=== Courbes d'entrainement : fakv12 Binaire ===")
plot_history(history_fak12)
