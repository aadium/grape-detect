import cv2
import numpy as np

# Load the YOLO v3 network
net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yv3_grapes.weights')

# Load the classes
classes = []
with open('obj.names', 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Open a connection to the webcam
# cap = cv2.VideoCapture('testingVideos/grapev3.mp4')
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()
    
    grapesCount = 0

    # Create a blob from the frame
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    
    # Set the input blob for the network
    net.setInput(blob)
    
    # Forward pass through the network
    outs = net.forward(net.getUnconnectedOutLayersNames())
    
    # Process the outputs
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
    
            if confidence > 0.5:
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                width = int(detection[2] * frame.shape[1])
                height = int(detection[3] * frame.shape[0])
                left = int(center_x - width/2)
                top = int(center_y - height/2)
    
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
    
    # Apply non-maximum suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    # Draw bounding boxes and labels on the frame
    for i in indices:
        box = boxes[i]
        left, top, width, height = box
        class_id = class_ids[i]
        label = classes[class_id]
        confidence = confidences[i]
    
        cv2.rectangle(frame, (left, top), (left + width, top + height), (0, 255, 0), 2)
        cv2.putText(frame, f'{label}: {confidence:.2f}', (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        grapesCount += 1
    
    print(grapesCount)
    
    # Display the frame
    cv2.imshow('Object Detection', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture and destroy windows
cap.release()
cv2.destroyAllWindows()