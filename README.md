# FakeSeeker: A Deepfake Image Detection Using EfficientNet

Welcome to FakeSeeker! This application is designed to help you detect deepfake images and videos using advanced AI.

## Table of Contents
1.  [Getting Started (Installation)](#getting-started-installation)
2.  [User Manual](#user-manual)
    *   [2.1 Home Page](#21-home-page)
    *   [2.2 Uploading and Scanning Media (Image/Video)](#22-uploading-and-scanning-media-imagevideo)
    *   [2.3 Real-Time Detection](#23-real-time-detection)
        *   [2.3.1 Camera Detection](#231-camera-detection)
        *   [2.3.2 Screen Monitoring](#232-screen-monitoring)
    *   [2.4 Viewing Scan History](#24-viewing-scan-history)
    *   [2.5 Understanding Scan Reports](#25-understanding-scan-reports)
    *   [2.6 Sidebar Navigation](#26-sidebar-navigation)
    *   [2.7 Action Buttons (Back/Exit)](#27-action-buttons-backexit)
    *   [2.8 Automatic Updates](#28-automatic-updates)
3.  [Troubleshooting](#troubleshooting)
4.  [About & Contact (Optional)](#4-about--contact-optional)

---

## 1. Getting Started (Installation)

This guide will help you download and run FakeSeeker on your computer.

**What You Need:**

*   A Windows 10/11 computer.
*   An internet connection (for the initial download and for model updates).
*   A tool to extract `.7z` files, such as [7-Zip (free)](https://www.7-zip.org/) or WinRAR.

**Steps:**

1.  **Download the Application:**
    *   Go to the [**Releases Page**](https://github.com/liftlobby/Fakeseeker/releases) on GitHub.
    *   Find the latest release (e.g., v1.3.0 or newer).
    *   Under the "Assets" section for that release, download the `.7z` file (e.g., `FakeSeeker_vX.Y.Z.7z`).
    *   *(Alternative Google Drive Link, if you still want to provide it, but GitHub Releases is standard for projects like this): [Application Download](https://drive.google.com/file/d/16-YbifOXobMEVKaTZSEAauFXlrTN3s07/view?usp=sharing)*
    *   The download size is under 2GB, so it may take some time depending on your internet speed. Wait for the download to complete.

2.  **Extract the Application:**
    *   Locate the downloaded `FakeSeeker.7z` file (usually in your "Downloads" folder).
    *   Right-click on the `.7z` file.
    *   If you have 7-Zip installed: Select "7-Zip" -> "Extract to FakeSeeker\\".
    *   If you have WinRAR: Select "Extract to FakeSeeker\\".
    *   This will create a new folder named `FakeSeeker`.

3.  **Run FakeSeeker:**
    *   Open the folder that was just created.
    *   Inside this folder, find the file named `FakeSeeker.exe`.
    *   Double-click on **`FakeSeeker.exe`** to start the application.
    *   **Windows Security Note:** The first time you run it, Windows Defender SmartScreen might show a blue window saying "Windows protected your PC".
        *   Click on "**More info**".
        *   Then click on the "**Run anyway**" button that appears. This is common for new applications not yet widely recognized by Microsoft.

---

## 2. User Manual

### 2.1 Home Page
![Home Page](https://github.com/liftlobby/Fakeseeker/images/Interfaces/homepage.png)
The Home Page is your starting point. It provides:
*   A welcome message.
*   Three main action buttons to access FakeSeeker's core features.
*   Brief instructions on how to use the application.

### 2.2 Uploading and Scanning Media (Image/Video)
![Upload Page](https://github.com/liftlobby/Fakeseeker/images/Interfaces/uploadpage.png)
This feature allows you to analyze static image or video files stored on your computer.

1.  **Navigate:** From the Home Page or Sidebar, click "Upload Image/Video".
2.  **Select File:**
    *   Click the "**Select File**" button.
    *   A file dialog will open. Choose the image (`.jpg`, `.jpeg`, `.png`) or video (`.mp4`, `.avi`, `.mov`) file you want to analyze.
    *   A preview of the selected image or the first frame of the video will appear.
3.  **Start Scan:**
    *   Once a file is selected and previewed, the "**Start Scan**" button will become active. Click it to begin the analysis.
    *   The status label below the preview will show the progress (e.g., "Extracting faces...", "Analyzing face X/N...").
4.  **Cancel Scan (Optional):**
    *   While the scan is in progress, the "**Cancel Scan**" button will be active. Click this if you wish to stop the current analysis. The scan will be halted, and no results will be saved for that scan.
5.  **View Results:**
    *   Once the scan is complete, you will automatically be taken to the **Detailed Scan Report** page (see section 2.5).
    *   If you chose to save the scan, it will also appear in your Scan History.

### 2.3 Real-Time Detection
![Real-Time Page](https://github.com/liftlobby/Fakeseeker/images/Interfaces/realtime.png)
FakeSeeker can analyze faces live from your computer's camera or by monitoring your screen.

#### 2.3.1 Camera Detection
1.  **Navigate:** From the Home Page or Sidebar, click "Real-Time Detection".
2.  **Turn Camera On:**
    *   Click the "**Turn Camera On**" button.
    *   If prompted by your system, allow FakeSeeker to access your camera.
    *   Your camera feed should appear in the video display area. The button will change to "Turn Camera Off".
3.  **Start Detection:**
    *   Once the camera feed is active, click the "**Start Detection**" button.
    *   FakeSeeker will start analyzing faces in the camera feed. Detected faces will have bounding boxes drawn around them, along with a "REAL" or "FAKE" label and a confidence score.
4.  **Stop Detection:**
    *   Click the "**Stop Detection**" button to stop the analysis. The camera feed will remain active.
    *   A summary of the detection session (if faces were analyzed) will be saved to your Scan History.
5.  **Turn Camera Off:** Click "Turn Camera Off" to release the camera.

#### 2.3.2 Screen Monitoring
1.  **Navigate:** From the Home Page or Sidebar, click "Real-Time Detection".
2.  **Start Screen Monitor:**
    *   Ensure your camera is off.
    *   Click the "**Start Screen Monitor**" button.
    *   The main FakeSeeker window will hide, and a small **Floating Toolbar** will appear (usually at the top of your screen).
3.  **Start Detection (from Toolbar):**
    *   On the Floating Toolbar, click "**Start Detection**".
    *   FakeSeeker will now analyze faces visible anywhere on your screen. Bounding boxes and predictions will be overlaid directly on your screen content.
    *   *Note: This feature can be resource-intensive.*
4.  **Stop Detection (from Toolbar):**
    *   Click "**Stop Detection**" on the Floating Toolbar.
    *   A summary of the detection session will be saved to your Scan History.
5.  **Show Status (from Toolbar):** Click "**Show Status**" to see a summary of the current real-time session's findings (faces analyzed, real/fake counts).
6.  **Return to Main Window:**
    *   Click "**Show Main Window**" on the Floating Toolbar. This will stop screen monitoring, close the toolbar, and restore the main FakeSeeker application window.
    *   Alternatively, closing the toolbar via its 'X' button will also stop monitoring and show the main window.

### 2.4 Viewing Scan History
![History Page](https://github.com/liftlobby/Fakeseeker/images/Interfaces/historypage.png)
Keep track of all your past analyses.
1.  **Navigate:** From the Home Page or Sidebar, click "View Scan History".
2.  **Browse History:**
    *   Scans are displayed as cards, sorted with the most recent at the top.
    *   Each card shows a summary: timestamp, type of scan, overall result, average fake probability (if applicable), and a thumbnail.
3.  **Actions per Scan:**
    *   **Details:** Click the "**Details**" button on a card to view the full Detailed Scan Report for that scan.
    *   **Delete:** Click the "**Delete**" button to remove that scan entry and its associated thumbnails from your history. You will be asked to confirm.

### 2.5 Understanding Scan Reports
![Report Page](https://github.com/liftlobby/Fakeseeker/images/Interfaces/reportpage.png)
The Detailed Scan Report provides an in-depth look at the analysis.
*   **Detection Summary:** Shows the overall status ("Potential Deepfake" or "Likely Real") and the average fake probability calculated from all analyzed faces, compared against the optimal threshold.
*   **File Details (for uploaded files):** Information like file size, type, resolution, and modification dates.
*   **Real-Time Session Info (for real-time scans):** Summary statistics like total faces processed, number of real/fake detections during the session.
*   **Detected Faces:** Thumbnails of all faces extracted and analyzed for the scan.
    *   For "Scanned" media or "Real-Time Detailed" reports, each face thumbnail will have its individual "REAL"/"FAKE" prediction and probability score displayed beneath it.
    *   For "Real-Time Summary" reports, only representative face thumbnails are shown without individual predictions.

### 2.6 Sidebar Navigation
![Sidebar](https://github.com/liftlobby/Fakeseeker/images/Interfaces/sidebar.png)
The sidebar on the left (visible on most pages except Home) provides quick navigation:
*   **Hover:** Move your mouse over the sidebar to expand it and see button labels.
*   **Click:** Click an icon or label to navigate to the corresponding page (Upload, Real-Time, History, Home).

### 2.7 Action Buttons (Back/Exit)
Located at the very bottom of the application window:
*   **Back to Home/Back:** On functional pages (Upload, Real-Time, History, Report), a button will allow you to navigate back to the Home page or the previous relevant page.
    *   When leaving a *Detailed Report* that was generated from a fresh scan (not from history), you'll be asked if you want to save the result to history before navigating away.
*   **Exit:** Closes the FakeSeeker application. Your scan history is automatically saved.

### 2.8 Automatic Updates
FakeSeeker will automatically check for new versions of the detection model and application itself when it starts (if an internet connection is available).
*   If an update is found, you will be prompted to download it.
*   Downloaded models and thresholds are stored in your user data directory.
*   It's recommended to restart FakeSeeker after an update to ensure the new files are loaded.

---

## 3. Troubleshooting

*   **"App Didn't Start" / Errors on Launch:**
    *   Ensure you extracted the *entire* `FakeSeeker` folder from the `.7z` archive and are running `FakeSeeker.exe` from *inside* that extracted folder. Do not run it directly from the archive viewer or move only the `.exe` file.
    *   Check the `fakeseeker.log` file (located in `C:\Users\[YourUsername]\AppData\Local\ChuaKaiZen_UTHM\FakeSeeker\logs` or a similar path shown in the console on first error) for detailed error messages.
*   **"Cannot Open .7z File":**
    *   You need an extraction tool like [7-Zip (free)](https://www.7-zip.org/) or WinRAR. Windows does not open `.7z` files by default.
*   **Antivirus Warning:**
    *   Some antivirus programs might flag `FakeSeeker.exe` because it's a new application. This is often a false positive. If you downloaded from the official GitHub Releases page, you may need to temporarily adjust your antivirus settings or add an exception for `FakeSeeker.exe`. *Proceed with caution and ensure you trust the download source.*
*   **Model/Threshold Errors on First Run (or after update fails):**
    *   FakeSeeker tries to download the latest detection model and threshold on first run or when an update is available.
    *   Ensure you have a stable internet connection.
    *   If it fails, the application might use a bundled default model or show an error if no model can be loaded. Restarting the application with an active internet connection usually resolves this. Check the logs for specific download errors.
*   **Video Rotation Issues:**
    *   If videos taken on a phone (especially portrait videos) appear sideways and faces are not detected, FakeSeeker attempts to use `ffprobe` (part of FFmpeg) to correct orientation.
    *   **For this to work, FFmpeg must be installed on your system and its `bin` directory (containing `ffprobe.exe`) must be added to your system's PATH environment variable.** If FFmpeg is not installed or not in PATH, rotation correction will not occur.
*   **"No Faces Detected":**
    *   Ensure the image/video contains clear, reasonably sized faces.
    *   For videos, faces might only be present in some parts. FakeSeeker samples frames; try a different video if issues persist.
    *   Check if the media is heavily compressed or very low resolution, which can hinder face detection.
    *   For rotated videos, ensure the rotation fix (requiring FFmpeg) is working.

---