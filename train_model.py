import os
import cv2
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans

data = []
labels = []

path = "PetImages"
categories = ["Cat", "Dog"]

for category in categories:
    folder = os.path.join(path, category)
    label = categories.index(category)

    for img in os.listdir(folder)[:2000]:
        try:
            img_path = os.path.join(folder, img)
            image = cv2.imread(img_path)
            image = cv2.resize(image, (64, 64))
            image = cv2.flatten(image) if hasattr(cv2, 'flatten') else image.flatten()

            data.append(image)
            labels.append(label)
        except:
            pass

data = np.array(data) / 255.0
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

models = {
    "svm": SVC(),
    "rf": RandomForestClassifier(),
    "lr": LogisticRegression(max_iter=1000),
    "kmeans": KMeans(n_clusters=2)
}

os.makedirs("models", exist_ok=True)

for name, model in models.items():
    model.fit(X_train, y_train)
    with open(f"models/{name}.pkl", "wb") as f:
        pickle.dump(model, f)

print("Models trained and saved")
