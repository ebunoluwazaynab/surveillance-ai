import cv2
from ultralytics import YOLO

# 1. Load the YOLOv8 model (n = nano, the fastest version for laptops)
model = YOLO('yolov8n.pt')

# 2. Open the laptop camera
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # Capture frame-by-frame
    success, frame = cap.read()

    if success:
        # 3. Run YOLOv8 inference on the frame
        # 'predict' identifies items; 'show=True' draws the boxes automatically
        results = model.predict(frame, conf=0.5, show=False)

        # 4. Process results to "flag" items in the console
        annotated_frame = results[0].plot() # This creates the image with boxes
       
        # Check what was found
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            label = model.names[class_id]
            print(f"Flagged: {label} detected.")

        # 5. Display the live feed
        cv2.imshow("Ultralytics Real-Time Monitoring", annotated_frame)

        # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release hardware
cap.release()
cv2.destroyAllWindows()