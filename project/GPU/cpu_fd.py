
import cv2
import dlib
import numpy as np
import time

# Load OpenCV Deep Learning Face Detector (CPU Version)
model_path = "/home/risheek/AdvCompArch2/Adv_Comp_Arch2/project/GPU/"

net = cv2.dnn.readNetFromCaffe(
    model_path + "deploy.prototxt",
    model_path + "res10_300x300_ssd_iter_140000_fp16.caffemodel"
)

# ✅ Force CPU execution
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)  # Use OpenCV's CPU backend
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)       # Force model to run on CPU

# Load dlib's landmark predictor
predictor = dlib.shape_predictor("./CPU_TEST/shape_predictor_68_face_landmarks.dat")  # Download separately

# Load image
image_path = "./test1.jpg"  # Change this to your image path
image = cv2.imread(image_path)

# Convert image to blob for CPU-based face detection
start_time = time.time()
blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0))
net.setInput(blob)
detections = net.forward()
cpu_face_detection_time = time.time() - start_time
print(f"CPU Face Detection Time: {cpu_face_detection_time:.4f} seconds")

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Benchmark Landmark Detection (CPU)
start_time = time.time()
h, w = image.shape[:2]
for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > 0.5:  # Confidence threshold
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        x, y, x1, y1 = box.astype("int")
        face_rect = dlib.rectangle(x, y, x1, y1)
        
        # Get facial landmarks
        landmarks = predictor(gray, face_rect)
        
        for j in range(68):  # dlib provides 68 facial landmarks
            lx, ly = landmarks.part(j).x, landmarks.part(j).y
            cv2.circle(image, (lx, ly), 2, (0, 255, 0), -1)  # Draw marker

cpu_landmark_detection_time = time.time() - start_time
print(f"CPU Landmark Detection Time: {cpu_landmark_detection_time:.4f} seconds")

# Show output
cv2.imshow("Facial Features - CPU", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

