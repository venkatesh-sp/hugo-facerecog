from flask import Flask, request, jsonify, render_template
from urllib.request import urlopen
import requests
import json
import face_recognition
import numpy as np
import base64
from PIL import Image
import io

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html")
    elif request.method == "POST":
        data = request.form

        with urlopen(data.get("image_data")) as response:
            image_data = response.read()

        with open(f"{data.get('name')}.png", "wb") as file_name:
            file_name.write(image_data)
        loaded_face = face_recognition.load_image_file(f"{file_name.name}")
        known_embedding = face_recognition.face_encodings(loaded_face, model="small")[0]
        np.save(f"{data.get('name')}", known_embedding)
        return jsonify({"status": "success"})


@app.route("/check/", methods=["GET", "POST"])
def match_user():
    if request.method == "GET":
        return render_template("check.html")
    elif request.method == "POST":
        data = request.form
        # img_decode = base64.b64decode(data.get("image_data"))

        img = np.fromstring(data.get("image_data"), dtype=np.uint8)
        img = img.reshape(-1, -1, 3)
        print(img.shape)
        # Get face locations extracted from frame
        # face_location = face_recognition.face_locations(img)
        # convert the face image to encoding
        face_encoding = face_recognition.face_encodings(img)[0]

        known_embedding = np.load(f"{data.get('name')}.npy")
        # Compare the face encoding extracted with the stored encodings for this user
        match = face_recognition.compare_faces(
            [known_embedding], face_encoding, tolerance=0.4
        )[0]
        return jsonify({"status": match})


if __name__ == "__main__":
    app.run(debug=True)

