#code to test 330 images and face detected/ time taken/FPS/CPU USAGE / MEMORY USGAE
import cv2
import dlib
import numpy as np
import os
import time
import psutil

# Load dlib's face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def load_pts_file(pts_path):
    """Load .pts file and extract (x, y) coordinates correctly."""
    with open(pts_path, "r") as f:
        lines = f.readlines()

    landmarks = []
    start_reading = False  

    for line in lines:
        line = line.strip()
        if line == "{":
            start_reading = True  
            continue
        elif line == "}":
            break  
        if start_reading:
            x, y = map(float, line.split())
            landmarks.append((x, y))

    return np.array(landmarks)

def extract_landmarks(image_paths, pts_paths):
    total_start_time = time.time()
    
    for img_path, pts_path in zip(image_paths, pts_paths):
        if not os.path.exists(img_path) or not os.path.exists(pts_path):
            print(f" Skipping missing file: '{img_path}' or '{pts_path}'")
            continue

        image = cv2.imread(img_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        load_pts_file(pts_path)  # Load ground truth landmarks
        
        cpu_before = psutil.cpu_percent()
        mem_before = psutil.virtual_memory().used / (1024 * 1024)  

        start_time = time.time()
        faces = detector(gray)
        elapsed_time = time.time() - start_time

        cpu_after = psutil.cpu_percent()
        mem_after = psutil.virtual_memory().used / (1024 * 1024)  
        mem_usage_diff = mem_after - mem_before  

        # Compute FPS
        fps = 1 / elapsed_time if elapsed_time > 0 else 0

        # Display Execution Details
        print(f"\n Processed: {img_path}")
        print(f" - Faces Detected: {len(faces)}")
        print(f" - Time Taken: {elapsed_time:.4f} sec")
        print(f" - FPS: {fps:.2f} frames/sec")
        print(f" - CPU Usage: Before = {cpu_before:.1f}%, After = {cpu_after:.1f}%")
        print(f" - Memory Usage Change: {mem_usage_diff:.2f} MB")

    total_elapsed_time = time.time() - total_start_time
    print(f"\n Total Execution Time: {total_elapsed_time:.4f} sec")
    print(f" Overall FPS: {len(image_paths) / total_elapsed_time:.2f} frames/sec")

# Define paths for all 330 images in 'testset' folder
image_list = [f"testset/data_{i}.jpg" for i in range(1, 331)]
pts_list = [f"testset/data_{i}.pts" for i in range(1, 331)]

# Run facial landmark extraction
extract_landmarks(image_list, pts_list)
