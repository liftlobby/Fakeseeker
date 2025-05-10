# Getting Started with FakeSeeker (Easy Installation)

Welcome to FakeSeeker! This guide will help you download and run the application on your computer using a direct download link. No technical skills are required!

## What You Need:

- A Windows computer (This guide assumes Windows. Instructions for macOS or Linux may differ slightly).
- An internet connection (for the initial download).
- A tool to extract .7z files, such as 7-Zip (free and recommended) or WinRAR. (Many computers do not have built-in support for .7z files like they do for .zip files.)

## Steps:

1. Download the Application:

- Click on this link to download the application file:
[Application Download](https://drive.google.com/file/d/1UMtDyzLD4PV7QkngxoANNTZ9qqvvDJPO/view?usp=sharing) or you may find it on RELEASES page. [Release page](https://github.com/liftlobby/Fakeseeker/releases/tag/1.2.0)

- Your web browser will open the link. Depending on your Google Drive link settings, you might see a preview or a direct download prompt.

- Look for a "Download" button or icon (often a downward arrow). Click it.

- If prompted, choose a location to save the file. Your browser's "Downloads" folder is the default location.

- The file you download will be named something like FakeSeeker_vX.Y.Z.7z (the version number X.Y.Z might be different). The download size is under 2GB, so it may take some time depending on your internet speed. Wait for the download to complete.

2. Extract the Application:

- Find the downloaded FakeSeeker_vX.Y.Z.7z file (usually in your "Downloads" folder).

- Right-click on the .7z file.

- Select "Extract All...".

- If you have 7-Zip installed, you should see a "7-Zip" option in the context menu. Hover over it and select "Extract to FakeSeeker_vX.Y.Z\". Alternatively, you can choose "Extract Here" if you prefer the files in the current folder, or "Extract files..." to choose a specific location.

- This will create a new folder named FakeSeeker (or similar) in the location you chose.

3. Run FakeSeeker:

- Open the FakeSeeker folder that was just created.

- Inside this folder, find the file named FakeSeeker.exe.

- Double-click on **FakeSeeker.exe** to start the application.

- Windows Security Note: The first time you run it, Windows Defender SmartScreen might pop up a blue window saying "Windows protected your PC".

- Click on "More info".

- Then click on the "Run anyway" button that appears. This is normal for applications downloaded from the internet that aren't yet widely recognized by Microsoft.

4. Using the Application:

- The FakeSeeker window should now open.

- Use the buttons on the home screen or the sidebar (if visible) to:

- Upload Image/Video: Select a file from your computer to scan.

- Real-Time Detection: Use your camera or monitor your screen.

- View Scan History: See results from previous scans.

- Click the buttons to explore the different features.

5. Troubleshooting:

- "App Didn't Start": Make sure you extracted the entire FakeSeeker folder from the .7z file and are running the .exe from inside that extracted folder. Don't just pull the .exe out by itself or run it directly from within the archive viewer.

- Cannot Open .7z File: If you cannot open the downloaded file, you likely need to install an extraction tool like 7-Zip (free).

- Antivirus Warning: Some antivirus programs might flag the .exe because it's a newly created application. This is often a false positive. If you trust the source, you may need to temporarily disable your antivirus or add an exception for FakeSeeker.exe. Proceed with caution if you are unsure.

- Model/Threshold Errors on First Run: The first time you run the app, it might try to download the latest detection model if you have an internet connection. If this fails, you might see an error. Please ensure you are connected to the internet and try restarting the application.

6. Enjoy using FakeSeeker!

## Acknowledgements
1. This project utilizes models and code adapted from or inspired by:
- EfficientNet-PyTorch
- Facenet-PyTorch
2. Thanks to the developers of OpenCV, Pillow, MSS, Requests, and Appdirs libraries.
