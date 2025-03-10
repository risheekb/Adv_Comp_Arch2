#code to test 21 images and face detected/ time taken/FPS/CPU USAGE / MEMORY USGAE

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
            print(f"Error: Missing file '{img_path}' or '{pts_path}'. Skipping.")
            continue

        image = cv2.imread(img_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        load_pts_file(pts_path)
        
        cpu_before = psutil.cpu_percent()
        mem_before = psutil.virtual_memory().used / (1024 * 1024)  

        start_time = time.time()
        faces = detector(gray)
        elapsed_time = time.time() - start_time

        cpu_after = psutil.cpu_percent()
        mem_after = psutil.virtual_memory().used / (1024 * 1024)  
        mem_usage_diff = mem_after - mem_before  

        # Calculate CPU change
        cpu_change = cpu_after - cpu_before
        cpu_change_text = " (CPU load decreased after processing)" if cpu_change < 0 else " (CPU load increased during processing)"

        # Calculate Memory usage change
        memory_change_text = f"Increased by {abs(mem_usage_diff):.2f} MB" if mem_usage_diff > 0 else f"Decreased by {abs(mem_usage_diff):.2f} MB (Some memory was freed after processing.)"

        print(f"\nProcessed: {img_path}")
        print(f" - Faces Detected: {len(faces)}")
        print(f" - Time Taken: {elapsed_time:.4f} sec")
        print(f" - FPS: {1 / elapsed_time:.2f} frames/sec")
        print(f" - CPU Usage: Before detection = {cpu_before:.1f}%, After detection = {cpu_after:.1f}%{cpu_change_text}")
        print(f" - Memory Usage: {memory_change_text}")

        if len(faces) == 0:
            continue

        for i, face in enumerate(faces):
            landmarks = predictor(gray, face)
            predicted_landmarks = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)]

            for x, y in predicted_landmarks:
                cv2.circle(image, (x, y), 2, (0, 255, 0), -1)  

        cv2.imshow(f"Landmarks - {os.path.basename(img_path)}", image)

    total_elapsed_time = time.time() - total_start_time
    print(f"\nTotal Execution Time: {total_elapsed_time:.4f} sec")
    print(f"Overall FPS: {len(image_paths) / total_elapsed_time:.2f} frames/sec")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Define file paths
image_list = [f"dataset/face{i}.jpg" for i in range(1, 22)]
pts_list = [f"dataset/face{i}.pts" for i in range(1, 22)]

# Run facial landmark extraction
extract_landmarks(image_list, pts_list)
