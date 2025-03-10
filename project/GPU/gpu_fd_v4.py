import cv2
import dlib
import numpy as np
import time
import glob
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
        sm_count = major * minor 
        gpu_name = pynvml.nvmlDeviceGetName(handle)

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
    gpu_name = pynvml.nvmlDeviceGetName(handle)
    total_cuda_cores = get_cuda_cores(handle)
    utilization_before = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu  

    print(f"GPU {i}: {gpu_name}")
    print(f"Total CUDA Cores (Estimated): {total_cuda_cores}")
    print(f"Initial CUDA Cores Utilized: ~{int(total_cuda_cores * (utilization_before / 100))} cores")
    print("=" * 40)

# Load CUDA-accelerated OpenCV Deep Learning Face Detector
model_path = "/home/risheek/AdvCompArch2/Adv_Comp_Arch2/project/GPU/"

net = cv2.dnn.readNetFromCaffe(
    model_path + "deploy.prototxt",
    model_path + "res10_300x300_ssd_iter_140000_fp16.caffemodel"
)

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Load dlib's landmark predictor
predictor = dlib.shape_predictor("./CPU_TEST/shape_predictor_68_face_landmarks.dat")  # Download separately

# Load multiple images for batch processing
image_paths = glob.glob("./CPU_TEST/testset/*.jpg")  # Load all images from a folder
images = [cv2.imread(img) for img in image_paths if cv2.imread(img) is not None]

# Resize images to prevent high memory usage
images = [cv2.resize(img, (1280, 720)) for img in images]  # Resize to 720p to save VRAM

# Dynamically set batch size based on available GPU memory
batch_size = 2  # Adjust based on memory availability
num_images = len(images)

print(f"Processing {num_images} images in batches of {batch_size}...")

# Process images in mini-batches
total_start_time = time.time()

for i in range(0, num_images, batch_size):
    batch_images = images[i: i + batch_size]  # Take a subset of images
    blobs = cv2.dnn.blobFromImages(batch_images, scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0))

    net.setInput(blobs)
    
    # Start GPU timer
    start_time = time.time()
    detections = net.forward()  # Run detection on batch
    batch_detection_time = time.time() - start_time

    print(f"Batch {i // batch_size + 1}: Face Detection Time: {batch_detection_time:.4f} sec")

    # Process each image's detection
    for idx, image in enumerate(batch_images):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = image.shape[:2]

        if detections.shape[0] > idx:
            for j in range(detections.shape[2]):  # Loop over detected faces
                box = detections[0, 0, j, 3:7] * np.array([w, h, w, h])
                x, y, x1, y1 = box.astype("int")
                face_rect = dlib.rectangle(x, y, x1, y1)

                # Get facial landmarks (Still on CPU)
                landmarks = predictor(gray, face_rect)

                # Draw landmarks on image
                for k in range(68):  # dlib provides 68 facial landmarks
                    lx, ly = landmarks.part(k).x, landmarks.part(k).y
                    cv2.circle(image, (lx, ly), 2, (0, 255, 0), -1)  # Draw marker

        # Show the image briefly
        #cv2.imshow("Facial Features - CUDA", image)
        #cv2.waitKey(500)  # Show each image for 500ms

# Monitor CUDA utilization after finishing
for i in range(device_count):
    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
    utilization_after = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu  

    utilization_diff = utilization_after - utilization_before
    estimated_cores_used = int(total_cuda_cores * (utilization_diff / 100))

    print("\n=== Post-Task GPU Stats ===")
    print(f"GPU {i}: {gpu_name}")
    print(f"Final CUDA Cores Utilized: ~{int(total_cuda_cores * (utilization_after / 100))} cores")
    print(f"Increase in CUDA Usage During Task: {utilization_diff}%")
    print(f"Estimated Cores Used for Task: ~{estimated_cores_used} cores")
    print("=" * 40)

# Total processing time
total_time = time.time() - total_start_time
print(f"\nTotal Processing Time for {num_images} images: {total_time:.4f} sec")

cv2.destroyAllWindows()

# Shutdown NVML
pynvml.nvmlShutdown()

