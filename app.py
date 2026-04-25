from flask import Flask, render_template, Response, request
from ultralytics import YOLO
from dotenv import load_dotenv
from email.message import EmailMessage
import cv2
import os
import time
import smtplib

load_dotenv()

app = Flask(__name__)

EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASS = os.getenv("EMAIL_PASS")
RECEIVER_EMAIL = os.getenv("RECEIVER_EMAIL")
THRESHOLD = int(os.getenv("THRESHOLD", 3))

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = YOLO("yolov8n.pt")
camera = cv2.VideoCapture(0)

last_email_time = 0
EMAIL_COOLDOWN = 60


def send_email_alert(count):
    global last_email_time

    if time.time() - last_email_time < EMAIL_COOLDOWN:
        return

    msg = EmailMessage()
    msg["Subject"] = "People Count Alert"
    msg["From"] = EMAIL_USER
    msg["To"] = RECEIVER_EMAIL
    msg.set_content(f"ALERT! People count exceeded.\nCurrent Count: {count}")

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(EMAIL_USER, EMAIL_PASS)
            smtp.send_message(msg)

        print("Email alert sent")
        last_email_time = time.time()

    except Exception as e:
        print("Email Error:", e)


def detect_people(frame):
    results = model(frame, verbose=False)
    people_count = 0

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])

            if class_id == 0:
                people_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                cv2.putText(
                    frame,
                    "Person",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

    return frame, people_count


def generate_frames():
    while True:
        success, frame = camera.read()

        if not success:
            break

        frame, people_count = detect_people(frame)

        status = "SAFE"

        if people_count > THRESHOLD:
            status = "ALERT"
            send_email_alert(people_count)

        color = (0, 255, 0) if status == "SAFE" else (0, 0, 255)

        cv2.putText(
            frame,
            f"People Count: {people_count} | Status: {status}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2
        )

        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"


@app.route("/", methods=["GET", "POST"])
def index():
    image_path = None
    image_count = None
    image_status = None

    if request.method == "POST":
        file = request.files["image"]

        if file:
            input_path = os.path.join(UPLOAD_FOLDER, "input.jpg")
            output_path = os.path.join(UPLOAD_FOLDER, "output.jpg")

            file.save(input_path)

            image = cv2.imread(input_path)
            image, image_count = detect_people(image)

            image_status = "SAFE"

            if image_count > THRESHOLD:
                image_status = "ALERT"
                send_email_alert(image_count)

            cv2.putText(
                image,
                f"People Count: {image_count} | Status: {image_status}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0) if image_status == "SAFE" else (0, 0, 255),
                2
            )

            cv2.imwrite(output_path, image)
            image_path = output_path

    return render_template(
        "index.html",
        threshold=THRESHOLD,
        image_path=image_path,
        image_count=image_count,
        image_status=image_status
    )


@app.route("/video")
def video():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)