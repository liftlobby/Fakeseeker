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
        header.pack(pady=20, anchor=tk.CENTER) # Ensure header is centered

        subheader = ttk.Label(main_container,
                            text="Deepfake Detection Using EfficientNet",
                            style='Normal.TLabel') # Uses the larger default TLabel style
        subheader.pack(pady=10, anchor=tk.CENTER) # Ensure subheader is centered

        button_frame = ttk.Frame(main_container)
        button_frame.pack(pady=30, fill=tk.X, padx=50, anchor=tk.CENTER)

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
                                style="TButton")
        upload_btn.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        realtime_btn = ttk.Button(button_frame,
                                  text="Real-Time Detection",
                                  command=self.controller.show_realtime_page,
                                  image=self.controller.realtime_icon,
                                  compound=tk.TOP,
                                  style="TButton")
        realtime_btn.grid(row=0, column=1, padx=10, pady=10, sticky="ew")

        history_btn = ttk.Button(button_frame,
                                 text="View Scan History",
                                 command=self.controller.show_history_page,
                                 image=self.controller.results_icon,
                                 compound=tk.TOP,
                                 style="TButton")
        history_btn.grid(row=0, column=2, padx=10, pady=10, sticky="ew")