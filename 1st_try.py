import cv2
import numpy as np
import collections
import time
from ultralytics import YOLO

# Φορτώνουμε το μοντέλο YOLO
model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(0)

# Περιοχές Ενδιαφέροντος (ROIs) - Τα προσαρμόζεις στη μακέτα σου
ROI_QUEUE_1 = (50, 100, 250, 400)
ROI_QUEUE_2 = (350, 100, 550, 400)

LOWER_BLUE = np.array([100, 150, 150])
UPPER_BLUE = np.array([140, 255, 255])
MIN_SIREN_AREA = 50

# Αυξήσαμε το ιστορικό στα 30 frames (περίπου 1 δευτερόλεπτο)
# για να καταλαβαίνει καλύτερα το ρυθμό του αναβοσβήσματος.
history_q1 = collections.deque(maxlen=30)
history_q2 = collections.deque(maxlen=30)

# Μεταβλητές για την αποθήκευση του χρόνου που ξεκίνησε το αναβόσβημα
blink_start_time_q1 = 0.0
blink_start_time_q2 = 0.0

while True:
    ret, frame = cap.read()
    if not ret: break

    results = model(frame, classes=[2, 3, 5, 7], verbose=False)
    score_q1 = 0.0
    score_q2 = 0.0

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    siren_mask = cv2.inRange(hsv_frame, LOWER_BLUE, UPPER_BLUE)

    emergency_q1 = 0
    emergency_q2 = 0

    cv2.rectangle(frame, (ROI_QUEUE_1[0], ROI_QUEUE_1[1]), (ROI_QUEUE_1[2], ROI_QUEUE_1[3]), (255, 0, 0), 2)
    cv2.rectangle(frame, (ROI_QUEUE_2[0], ROI_QUEUE_2[1]), (ROI_QUEUE_2[2], ROI_QUEUE_2[3]), (0, 255, 0), 2)

    pixels_q1 = cv2.countNonZero(siren_mask[ROI_QUEUE_1[1]:ROI_QUEUE_1[3], ROI_QUEUE_1[0]:ROI_QUEUE_1[2]])
    pixels_q2 = cv2.countNonZero(siren_mask[ROI_QUEUE_2[1]:ROI_QUEUE_2[3], ROI_QUEUE_2[0]:ROI_QUEUE_2[2]])

    history_q1.append(pixels_q1)
    history_q2.append(pixels_q2)

    # --- ΛΟΓΙΚΗ ΓΙΑ ΤΗΝ ΟΥΡΑ 1 ---
    is_blinking_q1 = False
    if len(history_q1) == 30:
        if max(history_q1) > MIN_SIREN_AREA and min(history_q1) < (MIN_SIREN_AREA // 2):
            is_blinking_q1 = True

    if is_blinking_q1:
        if blink_start_time_q1 == 0.0:
            blink_start_time_q1 = time.time()  # Ξεκινάει το χρονόμετρο!
        else:
            elapsed_time = time.time() - blink_start_time_q1
            if elapsed_time >= 2.0:  # Έχουν περάσει 2 δευτερόλεπτα σταθερού αναβοσβήσματος!
                emergency_q1 = 1
                cv2.putText(frame, "EMERGENCY Q1 CONFIRMED!", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
            else:
                # Δείχνουμε την αντίστροφη μέτρηση στην οθόνη για να βλέπεις τι γίνεται
                cv2.putText(frame, f"Detecting Q1... {2.0 - elapsed_time:.1f}s", (50, 80), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 165, 255), 2)
    else:
        blink_start_time_q1 = 0.0  # Το αντικείμενο έφυγε ή σταμάτησε να αναβοσβήνει, οπότε μηδενίζουμε

    # --- ΛΟΓΙΚΗ ΓΙΑ ΤΗΝ ΟΥΡΑ 2 ---
    is_blinking_q2 = False
    if len(history_q2) == 30:
        if max(history_q2) > MIN_SIREN_AREA and min(history_q2) < (MIN_SIREN_AREA // 2):
            is_blinking_q2 = True

    if is_blinking_q2:
        if blink_start_time_q2 == 0.0:
            blink_start_time_q2 = time.time()
        else:
            elapsed_time = time.time() - blink_start_time_q2
            if elapsed_time >= 2.0:
                emergency_q2 = 1
                cv2.putText(frame, "EMERGENCY Q2 CONFIRMED!", (350, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
            else:
                cv2.putText(frame, f"Detecting Q2... {2.0 - elapsed_time:.1f}s", (350, 80), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 165, 255), 2)
    else:
        blink_start_time_q2 = 0.0

    # Υπολογισμός YOLO
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            if ROI_QUEUE_1[0] < cx < ROI_QUEUE_1[2] and ROI_QUEUE_1[1] < cy < ROI_QUEUE_1[3]:
                score_q1 += 1.0
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            elif ROI_QUEUE_2[0] < cx < ROI_QUEUE_2[2] and ROI_QUEUE_2[1] < cy < ROI_QUEUE_2[3]:
                score_q2 += 1.0
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    with open("queue_data.txt", "w") as f:
        f.write(f"{score_q1},{score_q2},{emergency_q1},{emergency_q2}")

    cv2.putText(frame, f"Q1 Score: {score_q1}", (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    cv2.putText(frame, f"Q2 Score: {score_q2}", (350, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Traffic Camera - AI & Emergency Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
