import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
# from .base_page import BasePage

# class UploadPage(BasePage):
class UploadPage(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        container = ttk.Frame(self, padding="20")
        container.pack(fill=tk.BOTH, expand=True)

        header = ttk.Label(container, text="Upload Media for Analysis", style='Header.TLabel')
        header.pack(pady=20)

        # Preview frame - managed internally by this class now
        self.preview_frame = ttk.Frame(container)
        self.preview_frame.pack(pady=20)
        self.preview_label = ttk.Label(self.preview_frame)
        self.preview_label.pack()

        # Button frame
        button_frame = ttk.Frame(container)
        button_frame.pack(pady=20)

        self.status_label = ttk.Label(container, text="", style='Detail.TLabel')
        self.status_label.pack(pady=10)

        # Buttons call controller methods
        upload_btn = ttk.Button(button_frame, text="Select File", command=self.controller.upload_file)
        upload_btn.grid(row=0, column=0, padx=10)

        # Controller will enable/disable this via a method in this class if needed,
        # or directly if reference is kept by controller. Let's add a method.
        self.scan_btn = ttk.Button(button_frame, text="Start Scan",
                                 command=self.controller.start_scan, state=tk.DISABLED)
        self.scan_btn.grid(row=0, column=1, padx=10)

    def update_preview(self, file_path):
        """Updates the preview label. Called by the controller."""
        try:
            if not file_path:
                 self.preview_label.configure(image='')
                 self.preview_label.image = None
                 return

            if file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                image = Image.open(file_path)
                image.thumbnail((400, 400))
                photo = ImageTk.PhotoImage(image)
                self.preview_label.configure(image=photo)
                self.preview_label.image = photo # Keep reference
            elif file_path.lower().endswith(('.mp4', '.avi', '.mov')):
                cap = cv2.VideoCapture(file_path)
                ret, frame = cap.read()
                cap.release()
                if ret:
                    cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(cv2_im)
                    img.thumbnail((400, 400))
                    photo = ImageTk.PhotoImage(img)
                    self.preview_label.configure(image=photo)
                    self.preview_label.image = photo
                else:
                    self.preview_label.configure(text="Preview N/A")
            else:
                 self.preview_label.configure(text="Preview N/A")

        except Exception as e:
            messagebox.showerror("Preview Error", f"Failed to load preview: {str(e)}")
            self.preview_label.configure(image='', text="Preview Error")
            self.preview_label.image = None

    def set_scan_button_state(self, state):
        """Allows controller to enable/disable scan button."""
        if hasattr(self, 'scan_btn') and self.scan_btn.winfo_exists():
            self.scan_btn.config(state=state)