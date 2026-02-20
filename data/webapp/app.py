from flask import Flask, request, render_template
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model("../models/cnn_weights.h5")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["media"]
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (128,128)) / 255.0
        prediction = model.predict(np.expand_dims(img, axis=0))[0][0]
        result = "Fake" if prediction > 0.5 else "Real"
        return f"Result: {result} (Confidence: {prediction:.2f})"
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
