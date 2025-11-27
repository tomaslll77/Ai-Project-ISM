import cv2
import torch
import numpy as np
from torchvision import transforms
from model import LitRN50
import time

# ==========================================================
# CONFIG
# ==========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FACE_SIZE = (200, 200)              # Kad butu toks pat inputas kaip modelyje
SMOOTHING_ALPHA = 0.20              # EMA smoothing - labai sunku padaryti gera decisiona, kiekvienas sujudejimas daro itaka rezultatui.
STABILITY_TOLERANCE = 2.0           # Jei predictina daznai +-2metai leidzia lockint
STABLE_FRAMES_REQUIRED = 20         # kiek frames turetu buti stabilus
MIN_FRAMES_BEFORE_FORCE = 20        # kad iskart neuzlockintu amziaus
MAX_SCAN_TIME = 10                  # kad per ilgai negalvotu, jei uncertain
RESET_TIMEOUT = 2.0                 # kai nera veido, resettina

AGE_THRESHOLD_NO_ID = 25            # amziaus threshold 25
AGE_MIN = 0
AGE_MAX = 100

print("DEVICE =", DEVICE)
print("Loading model checkpoint...")

model = LitRN50.load_from_checkpoint("final_resnet50_model.ckpt", strict=False)
model.to(DEVICE)
model.eval()

print("Model loaded successfully!")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def predict_age(face_rgb):
    face_resized = cv2.resize(face_rgb, FACE_SIZE)
    tensor = transform(face_resized).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        age = model(tensor).item()

    return float(np.clip(age, AGE_MIN, AGE_MAX))
## Webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Webcam not found!")
    exit()

print("Starting webcam. Press 'q' to quit.")

ema_age = None
stable_frames = 0
locked_age = None
last_seen = time.time()
locked = False

frame_counter = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_h, frame_w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 4, minSize=(60, 60))

    if len(faces) == 0:
        if locked and (time.time() - last_seen > RESET_TIMEOUT):
            locked = False
            ema_age = None
            stable_frames = 0
            frame_counter = 0
            start_time = time.time()
        cv2.putText(frame, "No face detected",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 0, 255), 2)
        cv2.imshow("AI Age Estimation", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue
    last_seen = time.time()
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    cv2.rectangle(frame, (x, y), (x+w, y+h),
                  (0, 255, 0), 2)
    # atstumas - kai tam tikrame atstume yra veidas, geriau predictina
    face_ratio = h / frame_h

    if face_ratio > 0.55:
        cv2.putText(frame, "Move Back",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 0, 255), 2)
        ema_age = None
        stable_frames = 0
        cv2.imshow("AI Age Estimation", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    if face_ratio < 0.20:
        cv2.putText(frame, "Move Closer",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 0, 255), 2)
        ema_age = None
        stable_frames = 0
        cv2.imshow("AI Age Estimation", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    cv2.putText(frame, "Good Distance",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 0), 2)
    face_rgb = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)

    if locked:
        cv2.putText(frame, f"Age: {locked_age:.1f}",
                    (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2)
        cv2.imshow("AI Age Estimation", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    age = predict_age(face_rgb)
    if ema_age is None:
        ema_age = age
    else:
        ema_age = SMOOTHING_ALPHA * age + (1 - SMOOTHING_ALPHA) * ema_age

    smoothed_age = ema_age

    cv2.putText(frame, f"Scanning... {smoothed_age:.1f}",
                (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (255, 255, 0), 2)
    frame_counter += 1
    elapsed = time.time() - start_time

    if abs(age - smoothed_age) < STABILITY_TOLERANCE:
        stable_frames += 1
    else:
        stable_frames = max(0, stable_frames - 1)

    lock_now = (
        stable_frames >= STABLE_FRAMES_REQUIRED or
        (frame_counter >= MIN_FRAMES_BEFORE_FORCE and elapsed >= MAX_SCAN_TIME)
    )
    if lock_now:
        locked = True
        locked_age = smoothed_age

        if locked_age >= AGE_THRESHOLD_NO_ID:
            decision = "Verified: No ID Required"
        else:
            decision = "ID Check Required"

        # perduoda i main appsa predictiona
        print("LOCKED AGE:", locked_age)
        print("RESULT:", decision)

        cv2.putText(frame, decision,
                    (x, y + h + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 255), 2)

    cv2.imshow("AI Age Estimation", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
