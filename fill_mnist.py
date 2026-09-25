import json
from pathlib import Path

original = next(
    p for p in Path(".").glob("MNIST_EDA_Opdracht_Responsible_AI*.ipynb")
    if "Uitgewerkt" not in p.name
)
nb = json.loads(original.read_text(encoding="utf-8-sig"))

def fill(index, code):
    nb["cells"][index]["source"] = code.strip().splitlines(True)

# 2.1 Pandas
fill(8, """
print(df.shape)
display(df.head())
display(df["label"].value_counts().sort_index())
""")

# 3. Sanity check
fill(11, """
dataset_shape = df.shape
labels = df["label"].unique()
missing_values = df.isna().sum().sum()
min_pixel = df[pixel_cols].min().min()
max_pixel = df[pixel_cols].max().max()

print("Vorm:", dataset_shape)
print("Labels:", sorted(labels))
print("Ontbrekende waarden:", missing_values)
print("Pixelrange:", min_pixel, "t/m", max_pixel)
""")

# 3.1 Steekproef
fill(13, 'display(df.sample(n=5, random_state=42))')

# 4. Train/test split
fill(15, """
X_all = df[pixel_cols]
y_all = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all,
    test_size=0.20,
    random_state=42,
    stratify=y_all
)
print("Train:", X_train.shape, y_train.shape)
print("Test:", X_test.shape, y_test.shape)
""")

fill(17, """
train_percentage = y_train.value_counts(normalize=True).sort_index()
test_percentage = y_test.value_counts(normalize=True).sort_index()

split_check = pd.DataFrame({
    "train_%": (train_percentage * 100).round(2),
    "test_%": (test_percentage * 100).round(2)
})
display(split_check)
""")

# 5. EDA
fill(20, """
train_df = X_train.copy()
train_df["label"] = y_train
display(train_df.head())
""")

def replace(index, substitutions):
    source = "".join(nb["cells"][index]["source"])
    for old, new in substitutions.items():
        if old not in source:
            raise ValueError(f"Cel {index}: niet gevonden: {old}")
        source = source.replace(old, new)
    nb["cells"][index]["source"] = source.splitlines(True)

replace(22, {
    "label_counts = ...":
    'label_counts = train_df["label"].value_counts().sort_index()'
})

replace(24, {
    "most_common_label = ...": "most_common_label = label_counts.idxmax()",
    "least_common_label = ...": "least_common_label = label_counts.idxmin()"
})

replace(27, {
    "sample = ...": "sample = train_df.sample(n=10, random_state=42)"
})

replace(30, {
    'train_eda["mean_intensity"] = ...':
    'train_eda["mean_intensity"] = X_train.mean(axis=1)',
    'train_eda["active_pixels"] = ...':
    'train_eda["active_pixels"] = (X_train > 0).sum(axis=1)',
    'train_eda["bright_pixels"] = ...':
    'train_eda["bright_pixels"] = (X_train > 128).sum(axis=1)'
})

replace(34, {
    "mean_active_by_label = ...":
    'mean_active_by_label = train_eda.groupby("label")["active_pixels"].mean()'
})

replace(36, {
    "mean_digits = ...":
    'mean_digits = train_df.groupby("label")[pixel_cols].mean()'
})

fill(38, """
missing_pixels = X_train.isna().sum().sum()
min_pixel = X_train.min().min()
max_pixel = X_train.max().max()
duplicate_images = X_train.duplicated().sum()

print("Ontbrekende pixelwaarden:", missing_pixels)
print("Pixelrange:", min_pixel, "t/m", max_pixel)
print("Exacte dubbele afbeeldingen:", duplicate_images)
""")

# 8. Preprocessing
fill(41, """
X_train_scaled = X_train.astype("float32") / 255.0
X_test_scaled = X_test.astype("float32") / 255.0

print("Train min:", X_train_scaled.min().min())
print("Train max:", X_train_scaled.max().max())
print("Datatype:", X_train_scaled.dtypes.iloc[0])
""")

# 10. Validation set
replace(46, {
    "test_size=...": "test_size=0.20",
    "random_state=...": "random_state=42",
    "stratify=...": "stratify=y_train"
})

fill(48, """
train_images = X_model_train.to_numpy()
val_images = X_val.to_numpy()
test_images = X_test_scaled.to_numpy()

train_labels = y_model_train.to_numpy()
val_labels = y_val.to_numpy()
test_labels = y_test.to_numpy()

print(train_images.shape, val_images.shape, test_images.shape)
""")

# 11. Baselines
fill(50, """
random_baseline = 1 / 10
majority_baseline = y_model_train.value_counts(normalize=True).max()

print(f"Random baseline: {random_baseline:.2%}")
print(f"Majority baseline: {majority_baseline:.2%}")
""")

# 12. Neural network
fill(52, """
import keras
from keras import layers

keras.utils.set_random_seed(42)

model = keras.Sequential([
    keras.Input(shape=(784,)),
    layers.Dense(512, activation="relu"),
    layers.Dense(10, activation="softmax")
])

model.summary()
""")

# 13. Compile
fill(55, """
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)
""")

# 14. Training
fill(57, """
history = model.fit(
    train_images,
    train_labels,
    epochs=10,
    batch_size=128,
    validation_data=(val_images, val_labels)
)
""")

replace(59, {
    "history_df = ...": "history_df = pd.DataFrame(history.history)"
})

# 16. Evaluatie
replace(62, {
    "test_loss, test_accuracy = ...":
    "test_loss, test_accuracy = model.evaluate(test_images, test_labels)"
})

# 17. Voorspellingen
replace(64, {
    "predictions = ...": "predictions = model.predict(test_images)",
    "y_pred = ...": "y_pred = predictions.argmax(axis=1)"
})

# Voeg een machineleesbare samenvatting toe
nb["cells"].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": """
import json
from sklearn.metrics import confusion_matrix

summary = {
    "shape": list(df.shape),
    "missing": int(missing_values),
    "min_pixel": int(min_pixel),
    "max_pixel": int(max_pixel),
    "duplicates": int(duplicate_images),
    "most_common": int(most_common_label),
    "least_common": int(least_common_label),
    "min_active": int(mean_active_by_label.idxmin()),
    "max_active": int(mean_active_by_label.idxmax()),
    "train_accuracy": float(history.history["accuracy"][-1]),
    "val_accuracy": float(history.history["val_accuracy"][-1]),
    "val_history": [float(v) for v in history.history["val_accuracy"]],
    "test_accuracy": float(test_accuracy),
    "majority_baseline": float(majority_baseline),
    "confusions": [
        [int(i), int(j), int(cm[i,j])]
        for i in range(10) for j in range(10) if i != j
    ]
}
print("RESULT_JSON:" + json.dumps(summary))
""".strip().splitlines(True)
})

nb["metadata"]["kernelspec"] = {
    "display_name": "Python (.venv)",
    "language": "python",
    "name": "python3"
}

Path("MNIST_EDA_Uitgewerkt.ipynb").write_text(
    json.dumps(nb, ensure_ascii=False, indent=1),
    encoding="utf-8"
)

print("Alle programmeeropdrachten zijn ingevuld.")
