import cv2
import numpy as np

class StreamCamera:
    def __init__(self, camera_index):
        self.camera_index = camera_index
        self.running = False
        self.net = None
        self.classes = []
        self.detected_grapes = set()

    def load_model(self):
        # Load the YOLO v3 network
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
            
            detections = self.detect_objects(frame)
            frame = self.draw_detections(frame, detections)

            cv2.imshow('Object Detection', frame)
            if cv2.waitKey(1) == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

        yield_estimate = self.calculate_yield_estimate()
        print('Grapes Yield Estimate:', yield_estimate)

    def detect_objects(self, frame):
        
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.net.getUnconnectedOutLayersNames())

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

        return detections

    def draw_detections(self, frame, detections):
        for detection in detections:
            label = detection['label']
            confidence = detection['confidence']
            box = detection['box']

            x, y, w, h = box
            grape_id = f'{x}_{y}'

            # Check if grape has already been detected
            if grape_id not in self.detected_grapes:
                self.detected_grapes.add(grape_id)  # Add grape to set of detected grapes
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(frame, ((x + int(w / 2)), (y + int(h / 2))), 5, (0, 0, 255), -1)
                cv2.putText(frame, f'{label}: {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return frame

    def calculate_yield_estimate(self):
        yield_estimate = len(self.detected_grapes)
        return yield_estimate

    def stop_stream(self):
        self.running = False

def main():
    stream_camera = StreamCamera(0)
    stream_camera.start_stream()

if __name__ == "__main__":
    main()
