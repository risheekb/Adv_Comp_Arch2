import cv2
import dlib
import numpy as np
import time
import glob
import psutil  # For CPU core usage monitoring

# Get the number of CPU cores
total_cpu_cores = psutil.cpu_count(logical=True)

# Load OpenCV Deep Learning Face Detector (CPU)
model_path = "/home/risheek/AdvCompArch2/Adv_Comp_Arch2/project/GPU/"

net = cv2.dnn.readNetFromCaffe(
    model_path + "deploy.prototxt",
    model_path + "res10_300x300_ssd_iter_140000_fp16.caffemodel"
)

# **Force OpenCV to use CPU (Disable OpenCL & CUDA)**
cv2.ocl.setUseOpenCL(False)

# Load dlib's landmark predictor (Ensure dlib is using CPU)
predictor = dlib.shape_predictor("./CPU_TEST/shape_predictor_68_face_landmarks.dat")  # Download separately

# Load multiple images for batch processing
image_paths = glob.glob("./CPU_TEST/testset/*.jpg")  # Load all images from a folder
images = [cv2.imread(img) for img in image_paths if cv2.imread(img) is not None]

# Resize images to prevent high memory usage
images = [cv2.resize(img, (1280, 720)) for img in images]  # Resize to 720p for efficiency

# Set batch size
batch_size = min(2, len(images))  # Ensure batch size doesn't exceed total images
num_images = len(images)

print(f"Processing {num_images} images in batches of {batch_size} on CPU...")

# Monitor CPU utilization before starting
cpu_usage_before = psutil.cpu_percent(interval=1, percpu=True)

# Process images in mini-batches
total_start_time = time.time()

for i in range(0, num_images, batch_size):
    batch_images = images[i : i + batch_size]  # Take a subset of images
    blobs = cv2.dnn.blobFromImages(batch_images, scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0))

    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)  # Force CPU backend
    net.setInput(blobs)

    # Start CPU timer
    start_time = time.time()
    detections = net.forward()  # Run detection on batch (CPU mode)
    batch_detection_time = time.time() - start_time

    print(f"Batch {i // batch_size + 1}: Face Detection Time (CPU): {batch_detection_time:.4f} sec")

    # Process each image's detection
    for idx, image in enumerate(batch_images):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = image.shape[:2]

        # Ensure we don't access out-of-bounds indices
        if detections.shape[0] > idx:
            for j in range(detections.shape[2]):  # Loop over detected faces
                confidence = detections[0, 0, j, 2]  # Ensure correct indexing
                if confidence > 0.5:  # Confidence threshold
                    box = detections[0, 0, j, 3:7] * np.array([w, h, w, h])
                    x, y, x1, y1 = box.astype("int")
                    face_rect = dlib.rectangle(x, y, x1, y1)

                    # Get facial landmarks (Force dlib to use CPU)
                    start_time = time.time()
                    landmarks = predictor(gray, face_rect)  # dlib runs on CPU by default
                    landmark_detection_time = time.time() - start_time

                    print(f"Image {i + idx + 1}: Landmark Detection Time (CPU): {landmark_detection_time:.4f} sec")

                    # Draw landmarks on image
                    for k in range(68):  # dlib provides 68 facial landmarks
                        lx, ly = landmarks.part(k).x, landmarks.part(k).y
                        cv2.circle(image, (lx, ly), 2, (0, 255, 0), -1)  # Draw marker

        else:
            print(f"⚠️ No detections for image {i + idx + 1}, skipping landmark detection.")

        # Show output per image (Optional)
        #cv2.imshow(f"Facial Features - CPU Image {i + idx + 1}", image)
        #cv2.waitKey(500)  # Show each image for 500ms before moving to the next

# Monitor CPU utilization after finishing
cpu_usage_after = psutil.cpu_percent(interval=1, percpu=True)

# Calculate CPU utilization change
core_utilization_differences = [after - before for before, after in zip(cpu_usage_before, cpu_usage_after)]
avg_cpu_usage = sum(core_utilization_differences) / total_cpu_cores

# Total processing time
total_time = time.time() - total_start_time
print(f"\nTotal Processing Time for {num_images} images (CPU): {total_time:.4f} sec")

# Display CPU core utilization
print("\n=== CPU Utilization Stats ===")
print(f"Total CPU Cores: {total_cpu_cores}")
print(f"Average CPU Usage Increase: {avg_cpu_usage:.2f}%")
print("Per-Core Usage:")
for core_id, usage in enumerate(core_utilization_differences):
    print(f"Core {core_id}: {usage:.2f}% usage")

cv2.destroyAllWindows()

