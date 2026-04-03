import cv2
import numpy as np
import collections
import time
import sounddevice as sd  # ΝΕΟ: Βιβλιοθήκη για το μικρόφωνο
from ultralytics import YOLO

# --- ΡΥΘΜΙΣΕΙΣ ΗΧΟΥ (ΣΕΙΡΗΝΑΣ) ---
SAMPLE_RATE = 44100
AUDIO_THRESHOLD = 20.0  # ΡΥΘΜΙΣΕ ΤΟ: Ευαισθησία μικροφώνου (αυξομείωσε το στην πράξη)
siren_audio_active = False
last_audio_time = 0.0


# Συνάρτηση που τρέχει συνεχώς στο παρασκήνιο και "ακούει"
def audio_callback(indata, frames, time_info, status):
    global siren_audio_active, last_audio_time
    if status:
        pass  # Αγνόησε μικρά errors του buffer

    # Μετασχηματισμός Fourier για να βρούμε τις συχνότητες του ήχου
    fft_mags = np.abs(np.fft.rfft(indata[:, 0]))
    freqs = np.fft.rfftfreq(len(indata[:, 0]), 1 / SAMPLE_RATE)

    # Οι σειρήνες συνήθως "παίζουν" μεταξύ 500Hz και 1500Hz
    valid_indices = np.where((freqs > 500) & (freqs < 1500))
    if len(valid_indices[0]) > 0:
        max_mag = np.max(fft_mags[valid_indices])
        if max_mag > AUDIO_THRESHOLD:
            siren_audio_active = True
            last_audio_time = time.time()


# Ξεκινάμε την ακρόαση του μικροφώνου
audio_stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback)
audio_stream.start()

# --- ΑΡΧΙΚΟΠΟΙΗΣΗ ΟΡΑΣΗΣ ---
MIN_SIREN_AREA = 150
model = YOLO('vehiclesv2.pt')
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Camera not found!")
    exit()

# (Warm-up)
for i in range(30):
    cap.read()
time.sleep(1)

ret, frame = cap.read()
if not ret or frame is None:
    print("Error: Camera did not capture anything!")
    exit()

# --- CALIBRATION MODE ---
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

    # --- ΕΛΕΓΧΟΣ ΗΧΟΥ (Σβήνει το flag αν περάσουν 2 δευτερόλεπτα χωρίς ήχο) ---
    if time.time() - last_audio_time > 2.0:
        siren_audio_active = False

    if siren_audio_active:
        cv2.putText(frame, "AUDIO SIREN DETECTED!", (180, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 3)

    # --- ΟΠΤΙΚΗ ΑΝΙΧΝΕΥΣΗ ΦΑΡΟΥ ---
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red_1 = np.array([0, 180, 180])
    upper_red_1 = np.array([10, 255, 255])
    lower_red_2 = np.array([170, 180, 180])
    upper_red_2 = np.array([180, 255, 255])

    mask_red_1 = cv2.inRange(hsv_frame, lower_red_1, upper_red_1)
    mask_red_2 = cv2.inRange(hsv_frame, lower_red_2, upper_red_2)
    mask_red = cv2.bitwise_or(mask_red_1, mask_red_2)

    lower_blue = np.array([100, 180, 255])
    upper_blue = np.array([140, 255, 255])
    mask_blue = cv2.inRange(hsv_frame, lower_blue, upper_blue)

    siren_mask = cv2.bitwise_or(mask_red, mask_blue)

    contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours_blue:
        area = cv2.contourArea(cnt)
        if area > MIN_SIREN_AREA:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (255, 255, 0), 3)
            cv2.putText(frame, "BLUE BEACON", (bx, by - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

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

    # --- ΕΞΥΠΝΟΣ ΕΛΕΓΧΟΣ ΟΥΡΑΣ 1 (SENSOR FUSION: Φάρος + Σειρήνα) ---
    is_blinking_q1 = False
    if len(history_q1) == 30:
        flashes_q1 = 0
        is_on_q1 = False
        for pixels in history_q1:
            if pixels > MIN_SIREN_AREA and not is_on_q1:
                is_on_q1 = True
                flashes_q1 += 1
            elif pixels < (MIN_SIREN_AREA // 2) and is_on_q1:
                is_on_q1 = False
        if flashes_q1 >= 2:
            is_blinking_q1 = True

    if is_blinking_q1 and siren_audio_active:  # <-- ΕΔΩ ΜΠΑΙΝΕΙ Ο ΗΧΟΣ (AND)
        if blink_start_time_q1 == 0.0:
            blink_start_time_q1 = time.time()
        elif (time.time() - blink_start_time_q1) >= 1.0:  # Μειώθηκε ο χρόνος αναμονής
            emergency_q1 = 1
            cv2.putText(frame, "EMERGENCY Q1 CONFIRMED!", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
    else:
        blink_start_time_q1 = 0.0

    # --- ΕΞΥΠΝΟΣ ΕΛΕΓΧΟΣ ΟΥΡΑΣ 2 (SENSOR FUSION: Φάρος + Σειρήνα) ---
    is_blinking_q2 = False
    if len(history_q2) == 30:
        flashes_q2 = 0
        is_on_q2 = False
        for pixels in history_q2:
            if pixels > MIN_SIREN_AREA and not is_on_q2:
                is_on_q2 = True
                flashes_q2 += 1
            elif pixels < (MIN_SIREN_AREA // 2) and is_on_q2:
                is_on_q2 = False
        if flashes_q2 >= 2:
            is_blinking_q2 = True

    if is_blinking_q2 and siren_audio_active:  # <-- ΕΔΩ ΜΠΑΙΝΕΙ Ο ΗΧΟΣ (AND)
        if blink_start_time_q2 == 0.0:
            blink_start_time_q2 = time.time()
        elif (time.time() - blink_start_time_q2) >= 1.0:  # Μειώθηκε ο χρόνος αναμονής
            emergency_q2 = 1
            cv2.putText(frame, "EMERGENCY Q2 CONFIRMED!", (350, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
    else:
        blink_start_time_q2 = 0.0

    # --- ΥΠΟΛΟΓΙΣΜΟΣ ΒΑΡΟΥΣ (WEIGHTED VEHICLES) ---
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            cls_id = int(box.cls[0])
            class_name = model.names[cls_id]

            # Καθορισμός δυναμικού βάρους
            weight = 1.0
            if class_name == 'truck':
                weight = 2.0
            elif class_name == 'bus':
                weight = 5.0

            # Ελέγχουμε αν το όχημα ανήκει στο Q1
            if cv2.pointPolygonTest(poly_q1, (cx, cy), False) >= 0:
                if class_name != 'ambulance':  # Δεν μετράμε το ασθενοφόρο στην ουρά
                    score_q1 += weight
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"{class_name} ({weight})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 0, 0), 2)

            # Ελέγχουμε αν το όχημα ανήκει στο Q2
            elif cv2.pointPolygonTest(poly_q2, (cx, cy), False) >= 0:
                if class_name != 'ambulance':  # Δεν μετράμε το ασθενοφόρο στην ουρά
                    score_q2 += weight
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{class_name} ({weight})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 2)

    with open("queue_data.txt", "w") as f:
        f.write(f"{score_q1},{score_q2},{emergency_q1},{emergency_q2}")

    cv2.putText(frame, f"Q1 SCORE: {score_q1}", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(frame, f"Q2 SCORE: {score_q2}", (350, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Traffic Camera", frame)
    cv2.imshow("Siren Mask Debug", siren_mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Τερματισμός streams
audio_stream.stop()
audio_stream.close()
cap.release()
cv2.destroyAllWindows()
