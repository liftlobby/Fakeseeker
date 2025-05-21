import tkinter as tk
from tkinter import ttk

class HomePage(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent) # Corrected super call
        self.controller = controller

        main_container = ttk.Frame(self, padding="20")
        main_container.pack(fill=tk.BOTH, expand=True)

        header = ttk.Label(main_container,
                            text="Welcome to FakeSeeker",
                            style='Header.TLabel') # Uses the larger style
        header.pack(pady=(20, 5), anchor=tk.CENTER) # Ensure header is centered

        subheader = ttk.Label(main_container,
                            text="Deepfake Detection Using EfficientNet",
                            style='Normal.TLabel') # Uses the larger default TLabel style
        subheader.pack(pady=(0, 20), anchor=tk.CENTER) # Ensure subheader is centered

        button_frame = ttk.Frame(main_container)
        button_frame.pack(pady=20, fill=tk.X, padx=50, anchor=tk.CENTER)

        # Give equal weight to each column so they share space
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)
        button_frame.columnconfigure(2, weight=1)

        # Buttons now call controller methods to switch pages
        upload_btn = ttk.Button(button_frame,
                                text="Upload Image/Video",
                                command=self.controller.show_upload_page,
                                image=self.controller.upload_icon,
                                compound=tk.TOP,
                                style="Home.TButton")
        upload_btn.grid(row=0, column=0, padx=10, pady=10, sticky="ewns")

        realtime_btn = ttk.Button(button_frame,
                                  text="Real-Time Detection",
                                  command=self.controller.show_realtime_page,
                                  image=self.controller.realtime_icon,
                                  compound=tk.TOP,
                                  style="Home.TButton")
        realtime_btn.grid(row=0, column=1, padx=10, pady=10, sticky="ewns")

        history_btn = ttk.Button(button_frame,
                                 text="View Scan History",
                                 command=self.controller.show_history_page,
                                 image=self.controller.results_icon,
                                 compound=tk.TOP,
                                 style="Home.TButton")
        history_btn.grid(row=0, column=2, padx=10, pady=10, sticky="ewns")

        instructions_frame = ttk.LabelFrame(main_container, text="How to Use FakeSeeker", padding="15", style="Instructions.TLabelframe")
        instructions_frame.pack(pady=(20, 10), padx=50, fill=tk.X, expand=False, anchor=tk.N) # Anchor North

        instructions_text = [
            "1. Upload Image/Video: Select a local image or video file to analyze for deepfakes.",
            "2. Real-Time Detection: Use your camera or monitor your screen for live deepfake analysis.",
            "3. View Scan History: Review past scan results and detailed reports.",
            "   - After uploading a file, click 'Start Scan' to begin analysis.",
            "   - In Real-Time mode, click 'Start Detection' once your camera/screen feed is active.",
            "   - Results will indicate 'REAL' or 'FAKE' with a confidence score."
        ]

        for instruction in instructions_text:
            # Use a specific style for instruction labels if desired, e.g., 'Instruction.TLabel'
            instr_label = ttk.Label(instructions_frame, text=instruction, wraplength=700, justify=tk.LEFT, style="Instruction.TLabel")
            instr_label.pack(anchor=tk.W, pady=3)