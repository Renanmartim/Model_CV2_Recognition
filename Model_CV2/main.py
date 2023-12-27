import cv2
import numpy as np

# Load YOLO model
net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")

# Load COCO names
with open('coco.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Open video capture (0 for default camera or provide a video file path)
cap = cv2.VideoCapture('Rush Hour Traffic with motorcycle in Ho Chi Minh city - Vietnam.mp4')

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("Video stream ended or cannot be opened.")
        break

    height, width = frame.shape[:2]

    # Prepare the frame for YOLO
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)

    # Get output layer names
    output_layers = net.getUnconnectedOutLayersNames()

    # Forward pass
    outs = net.forward(output_layers)

    # Initialize lists for class IDs, confidences, and bounding boxes
    class_ids = []
    confidences = []
    boxes = []

    # Loop through each detection
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            # Ensure that class_id is within the valid range
            if 0 <= class_id < len(classes):
                label = str(classes[class_id])

                # Append information to lists
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # Apply non-maximum suppression to remove overlapping bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Check if any indices are available
    if len(indices) > 0:
        indices = indices.flatten()

        for i in indices:
            i = int(i)

            # Check if the index is within the valid range
            if 0 <= i < len(class_ids) and 0 <= i < len(confidences):
                box = boxes[i]
                x, y, w, h = box
                label = str(classes[class_ids[i]])
                confidence = confidences[i]

                # Draw rectangle and label
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f'{label} {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the result
    cv2.imshow('Object Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()
