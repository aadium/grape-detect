import cv2
import numpy as np

class StreamCamera:
    def __init__(self, camera_index):
        self.camera_index = camera_index
        self.running = False
        self.net = None
        self.classes = []

    def load_model(self):
        # Load the Tiny YOLO v3 network
        self.net = cv2.dnn.readNetFromDarknet('yolov3.cfg', 'yv3_grapes.weights')

        # Load the classes
        with open('obj.names', 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]

    def start_stream(self):
        self.load_model()

        self.cap = cv2.VideoCapture(self.camera_index)

        self.running = True
        while self.running:
            _, frame = self.cap.read()

            # Resize the frame to a smaller size
            small_frame = cv2.resize(frame, (500, 300))

            # Convert the resized frame to a blob
            blob = cv2.dnn.blobFromImage(small_frame, 1/255.0, (416, 416), swapRB=True, crop=False)

            # Set the blob as the input to the network
            self.net.setInput(blob)

            # Run forward pass to get output layer predictions
            outs = self.net.forward(self.net.getUnconnectedOutLayersNames())

            # Process the output layer predictions
            detections = self.process_predictions(outs, frame)

            # Display the frame with detections
            cv2.imshow('Object Detection', frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def process_predictions(self, outs, frame):
        height, width = frame.shape[:2]

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        detections = []
        for i in indices:
            box = boxes[i]
            x, y, w, h = box
            label = self.classes[class_ids[i]]
            confidence = confidences[i]

            detections.append({
                'label': label,
                'confidence': confidence,
                'box': (x, y, w, h)
            })

            # Draw the bounding box and label on the frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f'{label}: {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return detections

    def stop_stream(self):
        self.running = False

def main():
    stream_camera = StreamCamera(0)
    stream_camera.start_stream()

if __name__ == "__main__":
    main()
