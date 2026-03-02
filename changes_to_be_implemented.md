# AI Classroom System — Changes To Be Implemented

> This document is a living reference. Each section captures a user query, followed by structured improvement suggestions. New entries are added at the bottom.

---

## Table of Contents

1. [Feature Request — Upload Video Recording for Attendance](#1-feature-request--upload-video-recording-for-attendance)
2. [Performance Issue — YOLOv8 CPU Lag](#2-performance-issue--yolov8-cpu-lag)
3. [Training & Accuracy Improvements (Images & Encodings)](#3-training--accuracy-improvements-images--encodings)

---

## 1. Feature Request — Upload Video Recording for Attendance

### User Query
> *"Is it possible to add a feature where upload the recording of class and then the attendance is marked?"*

---

### Implementation Plan

#### 1.1 Video Upload Interface
- **Task**: Add a file upload UI in the web interface that accepts video files (e.g., MP4, AVI).
- **Impact**: Allows users to upload pre-recorded classes instead of relying solely on live webcam feeds.

#### 1.2 Frame Extraction
- **Task**: Use OpenCV (`cv2.VideoCapture`) securely on the backend to open the uploaded video file and read it frame-by-frame.
- **Impact**: Translates video files into processable image frames.

#### 1.3 Optimized Processing (Frame Sampling)
- **Problem**: Processing every frame of a long video is extremely computationally expensive and slow.
- **Fix**: Skip frames. Sample the video by extracting and processing one frame every 2 to 5 seconds.
- **Impact**: Speeds up processing of video files significantly.

#### 1.4 Background Processing & Progress Bar
- **Problem**: Processing a 1-hour video will take time and freeze the web page if done synchronously.
- **Fix**: Run the video face recognition as a background task. Implement a progress bar or loading spinner on the frontend.
- **Impact**: Enhances user experience by preventing timeouts and frozen pages.

#### 1.5 Adjust Upload Limits
- **Problem**: Class recordings can be very large (hundreds of MBs to GBs). Default web framework upload limits will reject them.
- **Fix**: Increase the maximum upload size limit in the web framework (e.g., Flask/FastAPI configuration).
- **Impact**: Enables the upload and processing of typical full-length class recordings.

---

## 2. Performance Issue — YOLOv8 CPU Lag

*(Note: Multi-threading background processing has already been implemented, reducing main UI lag. The following are further suggested optimizations.)*

### 2.1 Drop PyTorch for ONNX (OpenCV DNN)
- **Problem**: Running a raw `.pt` PyTorch model in Python carries heavy execution overhead, especially on CPUs lacking CUDA processing.
- **Fix**: Export `yolov8n-face.pt` to `.onnx` format and run inference via OpenCV's `cv2.dnn` module or `ONNXRuntime`.
- **Expected Impact**: Removes PyTorch runtime overhead, potentially doubling bounding-box generation FPS on weaker hardware.

### 2.2 Aggressive Frame Skipping
- **Problem**: The AI threads still consume heavy resources by calculating at maximum rate.
- **Fix**: Instead of processing as fast as the loop runs, change the logic to process every 10th or 15th frame. Given PyTorch's heavy footprint, the AI only needs to look at the crowd once every 0.5 seconds anyway.

---

## 3. Training & Accuracy Improvements (Images & Encodings)

### User Query
> *"how exactly is the model traing for face recog done here and what model is used does adding more students and images improve accuracy... suggest improvements or changes for improving accuracy and performance and training with images"*

---

### 3.1 Accuracy Improvements

#### 3.1.1 Increase Jitters During Encoding
- **Problem**: Default encoding maps the face just once, which might miss slight structural variations.
- **Fix**: Update `encodings = face_recognition.face_encodings(image, num_jitters=5)` in `encode_known_faces()`. This randomly distorts the image 5 times to create a more robust "average" face map.
- **Expected Impact**: Makes live matching much more accurate against different angles in the live feed. (Increases initial processing time, but not live detection time).

#### 3.1.2 Ensure Clean Training Data (Single Face Policy)
- **Problem**: If a student uploads a group photo, the system might encode the wrong face or crash the encoder.
- **Fix**: Add a validation step before encoding to ensure `len(face_locations(image)) == 1`. If multiple faces are detected, skip the image, log a warning, and move the image to a `rejected_images` folder.
- **Expected Impact**: Prevents database corruption with bad/ambiguous face encodings.

#### 3.1.3 Store Multiple Variable Images Per Student
- **Problem**: The system struggles if the live classroom lighting drastically differs from the single reference photo.
- **Fix**: Add 3-5 images per student under varying lighting conditions, angles, and with/without accessories (e.g., glasses). The underlying architecture natively supports multiple encodings matching to a single name.
- **Expected Impact**: The most significant and immediate way to boost accuracy without making architectural code changes.

#### 3.1.4 Artificial Data Augmentation (Brightness)
- **Problem**: Relying on students to provide perfectly lit photos is unrealistic.
- **Fix**: Programmatically lower and raise the brightness of the provided image using OpenCV inside `encode_known_faces()`, extracting encodings for each artificial variation.
- **Expected Impact**: Generates extreme robustness against dark classroom corners and brightly lit windows.

---

### 3.2 Performance Improvements

#### 3.2.1 Shrink YOLO Scan Frame Smaller
- **Problem**: YOLOv8 is running on a 50% downscaled frame, but it's efficient enough to scan even smaller frames.
- **Fix**: Change downscale factor to 25% (`fx=0.25, fy=0.25`) for the YOLO pass, then scale bounding box coordinates back up by a factor of 4 to map to the 100% resolution frame for the heavy dlib encoder.
- **Expected Impact**: Exponentially reduces PyTorch CPU workload, drastically improving background AI thread speed.

---

*Add new query sections below this line as they arise.*
