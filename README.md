# grape-detect
https://drive.google.com/file/d/1wz0pACVMTV9x1HwatQSxKj8wSWG5yOcQ/view?usp=sharing
To perform object detection, run the yolov3.py file.

<H2>Description</H2>
This code defines a Python class called StreamCamera that performs real-time object detection on a video stream from a camera. The class uses the YOLOv3 (You Only Look Once) deep learning model to detect grapes in the video frames. It keeps track of the number of grapes detected and calculates a yield estimate based on the count of detected grapes.

Here is a summary of the code:
<ul>
<li>The code imports the necessary libraries: cv2 for computer vision operations and numpy for numerical computations.

<li>The StreamCamera class is defined, which has the following attributes:
<ul>
<li>camera_index: An index representing the camera source to capture the video stream.
<li>running: A boolean variable indicating whether the video stream is running.
<li>net: The YOLOv3 neural network model.
<li>classes: A list of classes that the model can detect.
<li>detected_grapes: A set to store the unique identifiers of detected grapes.
</ul>
<li>The StreamCamera class provides the following methods:
<ul>
<li>load_model(): Loads the YOLOv3 network from configuration and weight files and reads the classes from a file.
<li>start_stream(): Starts the video stream, continuously reads frames from the camera, detects grapes in each frame, and displays the frames with bounding boxes and labels. It also calculates the yield estimate.
<li>detect_objects(frame): Performs grape detection on a given frame using YOLOv3. It processes the frame through the network and extracts the bounding boxes, class labels, and confidence scores of the detected grapes.
<li>draw_detections(frame, detections): Draws bounding boxes, labels, and circles around the detected grapes on the input frame. It also adds the unique identifier of each detected grape to the set of detected_grapes.
<li>calculate_yield_estimate(): Calculates the yield estimate by counting the number of unique detected grapes.
<li>stop_stream(): Stops the video stream by setting the running attribute to False.
<li>The main() function creates an instance of StreamCamera with a camera index of 0 (typically representing the default camera) and starts the video stream by calling the start_stream() method.
</ul>
<li>The script is executed if it is run directly (i.e., not imported as a module) by calling the main() function.
</ul>
In summary, this code sets up a real-time grape detection system using YOLOv3 and estimates the yield of grapes based on the detected count in the video stream.
