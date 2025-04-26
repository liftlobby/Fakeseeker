import tkinter as tk
from tkinter import ttk
# from .base_page import BasePage

# class RealtimePage(BasePage):
class RealtimePage(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        container = ttk.Frame(self, padding="20")
        container.pack(fill=tk.BOTH, expand=True)

        header = ttk.Label(container, text="Real-Time Detection", style='Header.TLabel')
        header.pack(pady=20)

        control_frame = ttk.Frame(container)
        control_frame.pack(pady=10)

        # Store references to buttons for state updates
        self.camera_toggle_btn = ttk.Button(
            control_frame, text="Turn Camera On",
            command=self.controller.toggle_camera, width=20)
        self.camera_toggle_btn.pack(side=tk.LEFT, padx=5)

        self.screen_toggle_btn = ttk.Button(
            control_frame, text="Start Screen Monitor",
            command=self.controller.toggle_screen_monitoring, width=20)
        self.screen_toggle_btn.pack(side=tk.LEFT, padx=5)

        self.start_detection_btn = ttk.Button(
            control_frame, text="Start Detection",
            command=self.controller.start_detection, width=20, state='disabled')
        self.start_detection_btn.pack(side=tk.LEFT, padx=5)

        self.stop_detection_btn = ttk.Button(
            control_frame, text="Stop Detection",
            command=self.controller.stop_detection, width=20, state='disabled')
        self.stop_detection_btn.pack(side=tk.LEFT, padx=5)

        # Video frame and label (controller needs reference to update it)
        video_frame = ttk.Frame(container, borderwidth=2, relief="solid")
        video_frame.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)

        self.video_label = ttk.Label(video_frame) # Controller will access this
        self.video_label.pack(pady=10)

        # Back button (goes home)
        # back_btn = ttk.Button(container, text="Back", command=self.controller.show_home_page)
        # back_btn.pack(pady=20)

    def update_button_states(self):
        """Updates button states based on controller state."""
        cam_text = "Turn Camera Off" if self.controller.camera_on else "Turn Camera On"
        screen_text = "Stop Screen Monitor" if self.controller.screen_monitoring else "Start Screen Monitor"

        start_enabled = (self.controller.camera_on or self.controller.screen_monitoring) and not self.controller.detection_active
        stop_enabled = self.controller.detection_active

        cam_toggle_enabled = not self.controller.screen_monitoring
        screen_toggle_enabled = not self.controller.camera_on

        # Update main control buttons
        if hasattr(self, 'start_detection_btn') and self.start_detection_btn.winfo_exists():
            self.start_detection_btn.config(state='normal' if start_enabled else 'disabled')
        if hasattr(self, 'stop_detection_btn') and self.stop_detection_btn.winfo_exists():
            self.stop_detection_btn.config(state='normal' if stop_enabled else 'disabled')

        # Update toggle buttons
        if hasattr(self, 'camera_toggle_btn') and self.camera_toggle_btn.winfo_exists():
            self.camera_toggle_btn.config(text=cam_text, state='normal' if cam_toggle_enabled else 'disabled')
        if hasattr(self, 'screen_toggle_btn') and self.screen_toggle_btn.winfo_exists():
            self.screen_toggle_btn.config(text=screen_text, state='normal' if screen_toggle_enabled else 'disabled')