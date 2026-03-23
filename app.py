from flask import Flask, request, render_template_string
import pickle
import numpy as np
import cv2

app = Flask(__name__)

models = {
    "svm": pickle.load(open("models/svm.pkl", "rb")),
    "rf": pickle.load(open("models/rf.pkl", "rb")),
    "lr": pickle.load(open("models/lr.pkl", "rb")),
    "kmeans": pickle.load(open("models/kmeans.pkl", "rb"))
}

HTML = """
<!DOCTYPE html>
<html>
<head>
<title>Cat vs Dog Classifier</title>
<style>
body {
    font-family: Arial;
    background: linear-gradient(120deg, #1e3c72, #2a5298);
    color: white;
    text-align: center;
    padding-top: 50px;
}
.container {
    background: white;
    color: black;
    width: 400px;
    margin: auto;
    padding: 30px;
    border-radius: 15px;
}
input, select, button {
    margin: 10px;
    padding: 10px;
    width: 90%;
}
button {
    background: #2a5298;
    color: white;
    border: none;
    cursor: pointer;
}
</style>
</head>
<body>

<div class="container">
<h2>🐶🐱 Cat vs Dog Classifier</h2>

<form method="POST" enctype="multipart/form-data">
<input type="file" name="file" required>
<select name="model">
<option value="svm">SVM</option>
<option value="rf">Random Forest</option>
<option value="lr">Logistic Regression</option>
<option value="kmeans">K-Means</option>
</select>
<button type="submit">Predict</button>
</form>

{% if result %}
<h3>Prediction: {{result}}</h3>
{% endif %}

</div>

</body>
</html>
"""

def preprocess(img):
    img = cv2.resize(img, (64, 64))
    img = img.flatten() / 255.0
    return img.reshape(1, -1)

@app.route("/", methods=["GET", "POST"])
def home():
    result = None

    if request.method == "POST":
        file = request.files["file"]
        model_name = request.form["model"]

        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        img = preprocess(img)

        model = models[model_name]
        pred = model.predict(img)[0]

        result = "Dog 🐶" if pred == 1 else "Cat 🐱"

    return render_template_string(HTML, result=result)

if __name__ == "__main__":
    app.run(debug=True)
