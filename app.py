from flask import Flask, request, jsonify
import librosa
import numpy as np
import tensorflow as tf

app = Flask(__name__)
model = tf.keras.models.load_model("music_genre_model.h5")  # Pre-trained model

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    y, sr = librosa.load(file, duration=30)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)

    prediction = model.predict(np.expand_dims(mfccs, axis=0))
    genre = np.argmax(prediction)

    return jsonify({"genre": genre})

if __name__ == "__main__":
    app.run(debug=True)
