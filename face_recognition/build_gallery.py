import face_recognition
import os
import pickle

ASSETS_DIR = "assets"
ENCODINGS_FILE = "encodings.pkl"

known_encodings = []
known_names = []

for file in os.listdir(ASSETS_DIR):
    if file.lower().endswith((".png", ".jpg", ".jpeg")):
        path = os.path.join(ASSETS_DIR, file)
        print(f"[INFO] Processing {file}...")

        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)

        if encodings:
            known_encodings.append(encodings[0])
            known_names.append(os.path.splitext(file)[0])
        else:
            print(f"[WARNING] No face found in {file}, skipping!")

with open(ENCODINGS_FILE, "wb") as f:
    pickle.dump({"encodings": known_encodings, "names": known_names}, f)

print("✅ All encodings saved to encodings.pkl")
