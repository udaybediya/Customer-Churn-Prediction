import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import keras_tuner as kt
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping


# ---------------- GPU SETUP ----------------
gpus = tf.config.list_physical_devices('GPU')

if gpus:
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("✅ GPU Enabled:", gpus)
    except RuntimeError as e:
        print(e)
else:
    print("❌ GPU not found, using CPU")


# ---------------- LOAD DATA ----------------
df = pd.read_csv("../Data/cleaned.csv")

# ---------------- FEATURES ----------------
categorical = ['Gender', 'Geography']

numerical = [
    'CreditScore', 'Age', 'Tenure', 'Balance',
    'NumOfProducts', 'HasCrCard',
    'IsActiveMember', 'EstimatedSalary'
]

# ---------------- SPLIT X / y ----------------
X = df[numerical + categorical]
y = to_categorical(df["Exited"])

# ---------------- ENCODER ----------------
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

X_cat = encoder.fit_transform(X[categorical])
X_num = X[numerical].values

# combine
X_final = np.hstack((X_num, X_cat))

# ---------------- TRAIN TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, random_state=42
)

# ---------------- SCALER ----------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------- EARLY STOP ----------------
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# ---------------- NEURON OPTIONS ----------------
NEURON_OPTIONS = [2, 4, 8, 16, 32, 64, 128, 256]

# ---------------- MODEL BUILDER ----------------
def build_model(hp):
    model = Sequential()

    for i in range(hp.Int("num_layers", 1, 5)):

        units = hp.Choice(f'units_{i}', NEURON_OPTIONS)

        model.add(Dense(units=units, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(rate=hp.Float(f'dropout_{i}', 0.0, 0.5, step=0.1)))

    # Output layer
    model.add(Dense(2, activation='sigmoid'))

    optimizer = hp.Choice('optimizer', ['adam', 'rmsprop', 'sgd'])

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

# ---------------- TUNER ----------------
tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=1,
    directory='tuner_dir',
    project_name='churn_gpu_final'
)

# ---------------- TUNING ----------------
tuner.search(
    X_train, y_train,
    epochs=20,
    validation_split=0.2,
    batch_size=64,
    callbacks=[early_stop]
)

# ---------------- BEST MODEL ----------------
best_hp = tuner.get_best_hyperparameters(1)[0]
best_model = build_model(best_hp)

print("\n🔥 BEST HYPERPARAMETERS")
print("Optimizer:", best_hp.get('optimizer'))
print("Layers:", best_hp.get('num_layers'))

# ---------------- FINAL TRAINING ----------------
history = best_model.fit(
    X_train, y_train,
    epochs=100,
    validation_data=(X_test, y_test),
    batch_size=64,
    callbacks=[early_stop]
)

# ---------------- EVALUATION ----------------
loss, accuracy = best_model.evaluate(X_test, y_test)
print("\n✅ FINAL TEST ACCURACY:", accuracy)

# ---------------- PLOTS ----------------
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title("Loss Graph")
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title("Accuracy Graph")
plt.show()

# ---------------- CONFUSION MATRIX ----------------
y_pred = best_model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true, y_pred_classes)

sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.show()

print("\n📊 Classification Report:\n")
print(classification_report(y_true, y_pred_classes))

# ---------------- SAVE ALL ----------------
os.makedirs("Model", exist_ok=True)

best_model.save("Model/churn_model.keras")
joblib.dump(scaler, "Model/scaler.pkl")
joblib.dump(encoder, "Model/encoder.pkl")

print("✅ Model + Scaler + Encoder saved successfully")