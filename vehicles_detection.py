import cv2
import numpy as np
import collections
import time
import sounddevice as sd
from ultralytics import YOLO

SAMPLE_RATE = 44100
CHUNK = 8192
MIN_FREQ = 800
MAX_FREQ = 1500
VOLUME_THRESHOLD = 0.01

HISTORY_LENGTH = max(5, int((SAMPLE_RATE / CHUNK) * 2.0))
REQUIRED_HITS = int(HISTORY_LENGTH * 0.60)

audio_history = collections.deque(maxlen=HISTORY_LENGTH)
siren_audio_active = False
last_audio_time = 0.0


def audio_callback(indata, frames, time_info, status):
    global siren_audio_active, last_audio_time
    if status:
        pass

    audio_data = indata[:, 0]
    rms = np.sqrt(np.mean(np.square(audio_data)))

    is_siren_now = False

    if rms > VOLUME_THRESHOLD:
        fft_data = np.fft.rfft(audio_data)
        fft_freqs = np.fft.rfftfreq(len(audio_data), 1.0 / SAMPLE_RATE)
        peak_freq = fft_freqs[np.argmax(np.abs(fft_data))]

        if MIN_FREQ <= peak_freq <= MAX_FREQ:
            is_siren_now = True

    audio_history.append(is_siren_now)

    if len(audio_history) == HISTORY_LENGTH and sum(audio_history) >= REQUIRED_HITS:
        siren_audio_active = True
        last_audio_time = time.time()
        print("🔊 (Στο παρασκήνιο): Ανιχνεύθηκε Ήχος Σειρήνας!")


audio_stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback, blocksize=CHUNK)
audio_stream.start()

model = YOLO('vehiclesv2.pt')
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Camera not found!")
    exit()

for i in range(30):
    cap.read()
time.sleep(1)

ret, frame = cap.read()
if not ret or frame is None:
    print("Error: Camera did not capture anything!")
    exit()

points = []

def select_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

        if 1 < len(points) <= 4:
            cv2.line(frame, points[-2], points[-1], (255, 0, 0), 2)
            if len(points) == 4:
                cv2.line(frame, points[3], points[0], (255, 0, 0), 2)

        elif 5 < len(points) <= 8:
            cv2.line(frame, points[-2], points[-1], (0, 255, 0), 2)
            if len(points) == 8:
                cv2.line(frame, points[7], points[4], (0, 255, 0), 2)

        cv2.imshow("Calibration - Click 8 Points", frame)


cv2.imshow("Calibration - Click 8 Points", frame)
cv2.setMouseCallback("Calibration - Click 8 Points", select_points)

print("--- ΔΙΑΔΙΚΑΣΙΑ ΒΑΘΜΟΝΟΜΗΣΗΣ ---")
print("1. Κάνε 4 κλικ για να κυκλώσεις την ΟΥΡΑ 1.")
print("2. Κάνε άλλα 4 κλικ για να κυκλώσεις την ΟΥΡΑ 2.")

while len(points) < 8:
    cv2.waitKey(1)

cv2.destroyWindow("Calibration - Click 8 Points")

poly_q1 = np.array(points[0:4], np.int32)
poly_q2 = np.array(points[4:8], np.int32)

print("Οι 2 δρόμοι ορίστηκαν επιτυχώς! Ξεκινάει η ανίχνευση...")

history_q1 = collections.deque(maxlen=30)
history_q2 = collections.deque(maxlen=30)

blink_start_time_q1 = 0.0
blink_start_time_q2 = 0.0

emergency_q1_until = 0.0
emergency_q2_until = 0.0
EMERGENCY_HOLD_TIME = 3.0

visual_hold_q1 = 0.0
visual_hold_q2 = 0.0
VISUAL_HOLD_TIME = 1.5

while True:
    ret, frame = cap.read()
    if not ret: break

    results = model(frame, verbose=False)
    score_q1 = 0.0
    score_q2 = 0.0
    emergency_q1 = 0
    emergency_q2 = 0

    if time.time() - last_audio_time > 2.0:
        siren_audio_active = False

    if siren_audio_active:
        cv2.putText(frame, "AUDIO SIREN DETECTED!", (180, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 3)

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red_1 = np.array([0, 150, 180])
    upper_red_1 = np.array([10, 255, 255])
    lower_red_2 = np.array([170, 150, 180])
    upper_red_2 = np.array([180, 255, 255])
    mask_red = cv2.bitwise_or(cv2.inRange(hsv_frame, lower_red_1, upper_red_1),
                              cv2.inRange(hsv_frame, lower_red_2, upper_red_2))
    lower_blue = np.array([95, 120, 210])
    upper_blue = np.array([140, 255, 255])
    mask_blue = cv2.inRange(hsv_frame, lower_blue, upper_blue)
    siren_mask = cv2.bitwise_or(mask_red, mask_blue)
    cv2.polylines(frame, [poly_q1], isClosed=True, color=(255, 0, 0), thickness=2)
    cv2.polylines(frame, [poly_q2], isClosed=True, color=(0, 255, 0), thickness=2)
    mask_poly_q1 = np.zeros(frame.shape[:2], dtype=np.uint8)
    mask_poly_q2 = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask_poly_q1, [poly_q1], 255)
    cv2.fillPoly(mask_poly_q2, [poly_q2], 255)

    pixels_q1 = cv2.countNonZero(cv2.bitwise_and(siren_mask, siren_mask, mask=mask_poly_q1))
    pixels_q2 = cv2.countNonZero(cv2.bitwise_and(siren_mask, siren_mask, mask=mask_poly_q2))

    history_q1.append(pixels_q1)
    history_q2.append(pixels_q2)

    if len(history_q1) == 30:
        max_p1 = max(history_q1)
        min_p1 = min(history_q1)

        MIN_SIREN_AREA_CONFIRM = 50

        if max_p1 > MIN_SIREN_AREA_CONFIRM:
            amplitude_p1 = max_p1 - min_p1
            threshold_amplitude_q1 = max_p1 * 0.70

            if amplitude_p1 > threshold_amplitude_q1:
                mean_p1 = sum(history_q1) / 30
                crossings_q1 = 0
                for i in range(1, 30):
                    if (history_q1[i - 1] > mean_p1) != (history_q1[i] > mean_p1):
                        crossings_q1 += 1

                if crossings_q1 >= 6:
                    visual_hold_q1 = time.time() + VISUAL_HOLD_TIME

    is_visual_active_q1 = time.time() < visual_hold_q1

    if is_visual_active_q1:
        cv2.putText(frame, "VISUAL SIREN Q1!", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if is_visual_active_q1 and siren_audio_active:
        if blink_start_time_q1 == 0.0:
            blink_start_time_q1 = time.time()
        elif (time.time() - blink_start_time_q1) >= 0.5:
            emergency_q1_until = time.time() + EMERGENCY_HOLD_TIME
    else:
        blink_start_time_q1 = 0.0

    if time.time() < emergency_q1_until:
        emergency_q1 = 1
        cv2.putText(frame, "EMERGENCY Q1 CONFIRMED!", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

    if len(history_q2) == 30:
        max_p2 = max(history_q2)
        min_p2 = min(history_q2)

        if max_p2 > MIN_SIREN_AREA_CONFIRM:
            amplitude_p2 = max_p2 - min_p2
            threshold_amplitude_q2 = max_p2 * 0.70

            if amplitude_p2 > threshold_amplitude_q2:
                mean_p2 = sum(history_q2) / 30
                crossings_q2 = 0
                for i in range(1, 30):
                    if (history_q2[i - 1] > mean_p2) != (history_q2[i] > mean_p2):
                        crossings_q2 += 1

                if crossings_q2 >= 6:
                    visual_hold_q2 = time.time() + VISUAL_HOLD_TIME

    is_visual_active_q2 = time.time() < visual_hold_q2

    if is_visual_active_q2:
        cv2.putText(frame, "VISUAL SIREN Q2!", (350, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if is_visual_active_q2 and siren_audio_active:
        if blink_start_time_q2 == 0.0:
            blink_start_time_q2 = time.time()
        elif (time.time() - blink_start_time_q2) >= 0.5:
            emergency_q2_until = time.time() + EMERGENCY_HOLD_TIME
    else:
        blink_start_time_q2 = 0.0

    if time.time() < emergency_q2_until:
        emergency_q2 = 1
        cv2.putText(frame, "EMERGENCY Q2 CONFIRMED!", (350, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]

            weight = 1.0
            if class_name == 'truck':
                weight = 2.0
            elif class_name == 'bus':
                weight = 5.0

            if cv2.pointPolygonTest(poly_q1, (cx, cy), False) >= 0:
                if class_name != 'ambulance': score_q1 += weight
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"{class_name} ({weight})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 0, 0), 2)

            elif cv2.pointPolygonTest(poly_q2, (cx, cy), False) >= 0:
                if class_name != 'ambulance': score_q2 += weight
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{class_name} ({weight})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 2)

    with open("queue_data.txt", "w") as f:
        f.write(f"{score_q1},{score_q2},{emergency_q1},{emergency_q2}")

    cv2.putText(frame, f"Lane 1 PCE: {score_q1}", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(frame, f"Lane 2 PCE: {score_q2}", (350, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Traffic Camera", frame)
    cv2.imshow("Siren Mask Debug", siren_mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

audio_stream.stop()
audio_stream.close()
cap.release()
cv2.destroyAllWindows()
