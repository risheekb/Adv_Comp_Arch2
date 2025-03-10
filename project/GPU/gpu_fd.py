import cv2
import dlib
import numpy as np
import time
import pynvml  # NVIDIA GPU monitoring library

# Initialize NVIDIA Management Library to fetch CUDA core info
pynvml.nvmlInit()
device_count = pynvml.nvmlDeviceGetCount()

# CUDA cores per SM for various architectures
CUDA_CORES_PER_SM = {
    "Kepler": 192,
    "Maxwell": 128,
    "Pascal": 64,
    "Volta": 64,
    "Turing": 64,
    "Ampere": 128,
    "Hopper": 128
}

def get_cuda_cores(handle):
    """Estimate CUDA cores based on GPU architecture."""
    try:
        major, minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
        sm_count = major * minor  # Approximate number of multiprocessors
        gpu_name = pynvml.nvmlDeviceGetName(handle)  # No .decode() needed!

        # Identify architecture based on GPU name
        for arch, cores_per_sm in CUDA_CORES_PER_SM.items():
            if arch in gpu_name:
                return sm_count * cores_per_sm  # Estimate total CUDA cores

        return sm_count * 64  # Default assumption (Pascal-based)

    except pynvml.NVMLError as e:
        print(f"Error fetching CUDA cores: {e}")
        return "Unknown"

# Iterate through each GPU and print CUDA core details
for i in range(device_count):
    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
    gpu_name = pynvml.nvmlDeviceGetName(handle)  # No .decode() needed!
    total_cuda_cores = get_cuda_cores(handle)
    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu  # GPU usage in %

    print(f"GPU {i}: {gpu_name}")
    print(f"Total CUDA Cores (Estimated): {total_cuda_cores}")
    print(f"CUDA Cores Utilized: ~{int(total_cuda_cores * (utilization / 100))} cores")
    print("=" * 40)

# Load CUDA-accelerated OpenCV Deep Learning Face Detector
model_path = "/home/risheek/AdvCompArch2/Adv_Comp_Arch2/project/GPU/"

net = cv2.dnn.readNetFromCaffe(
    model_path + "deploy.prototxt",
    model_path + "res10_300x300_ssd_iter_140000_fp16.caffemodel"
)

#net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000_fp16.caffemodel")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Load dlib's landmark predictor
predictor = dlib.shape_predictor("./CPU_TEST/shape_predictor_68_face_landmarks.dat")  # Download separately

# Load image
image_path = "./CPU_TEST/testset/data_326.jpg"  # Change this to your image path
#image_path = "./test1.jpg"  # Change this to your image path
image = cv2.imread(image_path)

# Convert image to blob for CUDA-based face detection
start_time = time.time()
blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0))
net.setInput(blob)
detections = net.forward()
cuda_face_detection_time = time.time() - start_time
print(f"CUDA Face Detection Time: {cuda_face_detection_time:.4f} seconds")

# Convert image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Benchmark Landmark Detection (Still on CPU)
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

landmark_detection_time = time.time() - start_time
print(f"CUDA Landmark Detection Time (Still on CPU): {landmark_detection_time:.4f} seconds")

# Show output
cv2.imshow("Facial Features - CUDA", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Shutdown NVML
pynvml.nvmlShutdown()

