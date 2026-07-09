# Animal-trophic-level-classification
"""
Trophic Level Classification using Supervised Learning
Classes: Carnivore, Omnivore, Herbivore, Frugivore, Insectivore

Model: Random Forest Classifier (also compares with Decision Tree & KNN)
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------------------------------------------------
# 1. DATASET
# Features engineered from real animal traits
# teeth_type: 0=canine-dominant, 1=molar-dominant, 2=mixed, 3=beak, 4=piercing/none
# gut_length: 0=short, 1=medium, 2=long
# eye_position: 0=forward, 1=side
# hunts_prey: 0=no, 1=yes
# habitat: 0=ground, 1=arboreal, 2=aquatic, 3=aerial
# body_size: 0=small, 1=medium, 2=large
# ---------------------------------------------------------

data = [
    # animal            teeth  gut  eye  hunts  habitat  size  label
    ["Lion",              0,    0,   0,    1,      0,     2, "Carnivore"],
    ["Tiger",             0,    0,   0,    1,      0,     2, "Carnivore"],
    ["Wolf",              0,    0,   0,    1,      0,     1, "Carnivore"],
    ["Crocodile",         4,    0,   0,    1,      2,     2, "Carnivore"],
    ["Shark",             4,    0,   0,    1,      2,     2, "Carnivore"],
    ["Eagle",             4,    0,   0,    1,      3,     1, "Carnivore"],

    ["Bear",              2,    1,   0,    1,      0,     2, "Omnivore"],
    ["Pig",               2,    1,   1,    0,      0,     1, "Omnivore"],
    ["Human",             2,    1,   0,    0,      0,     1, "Omnivore"],
    ["Chimpanzee",        2,    1,   0,    0,      1,     1, "Omnivore"],
    ["Crow",              2,    1,   0,    0,      3,     0, "Omnivore"],
    ["Raccoon",           2,    1,   0,    0,      0,     0, "Omnivore"],

    ["Cow",               1,    2,   1,    0,      0,     2, "Herbivore"],
    ["Deer",              1,    2,   1,    0,      0,     1, "Herbivore"],
    ["Elephant",          1,    2,   1,    0,      0,     2, "Herbivore"],
    ["Rabbit",            1,    2,   1,    0,      0,     0, "Herbivore"],
    ["Horse",             1,    2,   1,    0,      0,     2, "Herbivore"],
    ["Giraffe",           1,    2,   1,    0,      0,     2, "Herbivore"],

    ["Fruit Bat",         1,    0,   0,    0,      1,     0, "Frugivore"],
    ["Toucan",            3,    0,   0,    0,      1,     0, "Frugivore"],
    ["Orangutan",         2,    0,   0,    0,      1,     1, "Frugivore"],
    ["Spider Monkey",     2,    0,   0,    0,      1,     0, "Frugivore"],
    ["Fruit Pigeon",      3,    0,   0,    0,      1,     0, "Frugivore"],

    ["Anteater",          4,    0,   0,    1,      0,     1, "Insectivore"],
    ["Pangolin",          4,    0,   0,    1,      0,     0, "Insectivore"],
    ["Hedgehog",          2,    0,   0,    1,      0,     0, "Insectivore"],
    ["Chameleon",         4,    0,   0,    1,      1,     0, "Insectivore"],
    ["Swallow (bird)",    3,    0,   0,    1,      3,     0, "Insectivore"],
]

columns = ["animal", "teeth_type", "gut_length", "eye_position",
           "hunts_prey", "habitat", "body_size", "trophic_level"]

df = pd.DataFrame(data, columns=columns)

# ---------------------------------------------------------
# 2. FEATURES & LABELS
# ---------------------------------------------------------
X = df.drop(columns=["animal", "trophic_level"])
y = df["trophic_level"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ---------------------------------------------------------
# 3. TRAIN/TEST SPLIT
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

# ---------------------------------------------------------
# 4. TRAIN MODELS
# ---------------------------------------------------------
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "KNN (k=3)": KNeighborsClassifier(n_neighbors=3),
}

print("=" * 55)
print("TROPHIC LEVEL CLASSIFICATION - MODEL COMPARISON")
print("=" * 55)

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\n--- {name} ---")
    print(f"Accuracy: {acc:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, preds, target_names=le.classes_, zero_division=0))

# ---------------------------------------------------------
# 5. FEATURE IMPORTANCE (Random Forest)
# ---------------------------------------------------------
rf = models["Random Forest"]
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\n" + "=" * 55)
print("FEATURE IMPORTANCE (Random Forest)")
print("=" * 55)
print(importances)

# ---------------------------------------------------------
# 6. PREDICT A NEW / UNSEEN ANIMAL
# ---------------------------------------------------------
# Example: A Panda -> tricky case, taxonomically a carnivore but eats bamboo
new_animal = pd.DataFrame([{
    "teeth_type": 2,     # mixed
    "gut_length": 1,     # medium
    "eye_position": 0,   # forward (predator lineage)
    "hunts_prey": 0,     # doesn't hunt
    "habitat": 0,        # ground
    "body_size": 2,      # large
}])

prediction = rf.predict(new_animal)
predicted_label = le.inverse_transform(prediction)
print("\n" + "=" * 55)
print("PREDICTION FOR NEW ANIMAL (Panda-like features)")
print("=" * 55)
print(f"Predicted trophic level: {predicted_label[0]}")
