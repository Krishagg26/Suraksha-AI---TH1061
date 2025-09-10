import face_recognition
import cv2
import pickle

with open("encodings.pkl", "rb") as f:
    data = pickle.load(f)

known_encodings = data["encodings"]
known_names = data["names"]

person_details = {
    "Aman": {"phone": "9876543210", "aadhaar": "1234-5678-9012"},
    "Anuj": {"phone": "9123456789", "aadhaar": "2345-6789-0123"},
    "Krish": {"phone": "9111111111", "aadhaar": "3456-7890-1234"},
    "Pihu": {"phone": "9000000000", "aadhaar": "4567-8901-2345"},
    "somya": {"phone": "9999999999", "aadhaar": "5678-9012-3456"},
}

ref_images = {}
for name in known_names:
    img_path = f"assets/{name}.png"
    ref = cv2.imread(img_path)
    if ref is not None:
        ref = cv2.resize(ref, (80, 80))
        ref_images[name] = ref

image_path = "assets/Crowd.png"
image = cv2.imread(image_path)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

boxes = face_recognition.face_locations(rgb)
encodings = face_recognition.face_encodings(rgb, boxes)

for (box, encoding) in zip(boxes, encodings):
    matches = face_recognition.compare_faces(known_encodings, encoding)
    name = "Unknown"

    if True in matches:
        match_index = matches.index(True)
        name = known_names[match_index]

    top, right, bottom, left = box
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

    if name in ref_images:
        ref = ref_images[name]
        h, w, _ = ref.shape
        y1, y2 = top - h - 10, top - 10
        x1, x2 = left, left + w
        if y1 >= 0 and x2 <= image.shape[1]:
            image[y1:y2, x1:x2] = ref

    font = cv2.FONT_HERSHEY_SIMPLEX
    label = name
    (font_width, font_height), baseline = cv2.getTextSize(label, font, 1.2, 2)
    cv2.rectangle(image,
                  (left, bottom + 5),
                  (left + font_width, bottom + 5 + font_height + baseline),
                  (0, 0, 0),
                  cv2.FILLED)
    cv2.putText(image, label, (left, bottom + font_height + 5),
                font, 1.2, (255, 255, 255), 2)

    if name in person_details:
        details = person_details[name]
        cv2.putText(image, f"Ph: {details['phone']}", (left, bottom + font_height + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(image, f"Aadhaar: {details['aadhaar']}", (left, bottom + font_height + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

cv2.imshow("Recognized Crowd", image)

while True:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
