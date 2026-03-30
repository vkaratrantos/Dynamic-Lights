import cv2
import numpy as np
import collections
import time
from ultralytics import YOLO

# Αφαιρέσαμε τα χρώματα. Τώρα ορίζουμε το ελάχιστο μέγεθος του φωτός (σε pixels)
MIN_SIREN_AREA = 30  # Το κατέβασα λίγο γιατί ο πυρήνας ενός LED φαίνεται μικρός στην κάμερα

model = YOLO('vehicles.pt')

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("ΣΦΑΛΜΑ: Η κάμερα δεν βρέθηκε!")
    exit()

# Ζέσταμα κάμερας (Warm-up)
for i in range(30):
    cap.read()
time.sleep(1)

ret, frame = cap.read()
if not ret or frame is None:
    print("ΣΦΑΛΜΑ: Η κάμερα δεν τράβηξε εικόνα.")
    exit()

# --- CALIBRATION MODE (8 ΚΛΙΚ ΓΙΑ 2 ΔΡΟΜΟΥΣ) ---
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
print("1. Κάνε 4 κλικ για να κυκλώσεις την ΟΥΡΑ 1 (π.χ. τον οριζόντιο δρόμο).")
print("2. Στη συνέχεια, κάνε άλλα 4 κλικ για να κυκλώσεις την ΟΥΡΑ 2 (π.χ. τον κάθετο δρόμο).")

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

while True:
    ret, frame = cap.read()
    if not ret: break

    results = model(frame, verbose=False)
    score_q1 = 0.0
    score_q2 = 0.0
    emergency_q1 = 0
    emergency_q2 = 0

    # Μετατρέπουμε την εικόνα σε ασπρόμαυρη (Grayscale)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Κρατάμε ΜΟΝΟ τα πολύ φωτεινά σημεία (τιμές από 230 έως 255)
    # Αυτό θα πιάσει οποιοδήποτε έντονο λαμπάκι (κόκκινο, μπλε, κίτρινο, λευκό)
    _, siren_mask = cv2.threshold(gray_frame, 100, 255, cv2.THRESH_BINARY)

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

    # --- Έλεγχος Έκτακτης Ανάγκης στην Ουρά 1 ---
    is_blinking_q1 = False
    if len(history_q1) == 30 and max(history_q1) > MIN_SIREN_AREA and min(history_q1) < (MIN_SIREN_AREA // 2):
        is_blinking_q1 = True

    if is_blinking_q1:
        if blink_start_time_q1 == 0.0:
            blink_start_time_q1 = time.time()
        elif (time.time() - blink_start_time_q1) >= 2.0:
            emergency_q1 = 1
            cv2.putText(frame, "EMERGENCY Q1 CONFIRMED!", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
    else:
        blink_start_time_q1 = 0.0

    # --- Έλεγχος Έκτακτης Ανάγκης στην Ουρά 2 ---
    is_blinking_q2 = False
    if len(history_q2) == 30 and max(history_q2) > MIN_SIREN_AREA and min(history_q2) < (MIN_SIREN_AREA // 2):
        is_blinking_q2 = True

    if is_blinking_q2:
        if blink_start_time_q2 == 0.0:
            blink_start_time_q2 = time.time()
        elif (time.time() - blink_start_time_q2) >= 2.0:
            emergency_q2 = 1
            cv2.putText(frame, "EMERGENCY Q2 CONFIRMED!", (350, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
    else:
        blink_start_time_q2 = 0.0

    # --- Υπολογισμός Οχημάτων με YOLO ---
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            if cv2.pointPolygonTest(poly_q1, (cx, cy), False) >= 0:
                score_q1 += 1.0
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            elif cv2.pointPolygonTest(poly_q2, (cx, cy), False) >= 0:
                score_q2 += 1.0
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    with open("queue_data.txt", "w") as f:
        f.write(f"{score_q1},{score_q2},{emergency_q1},{emergency_q2}")

    cv2.putText(frame, f"Q1 VEHICLES: {score_q1}", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(frame, f"Q2 VEHICLES: {score_q2}", (350, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Traffic Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
