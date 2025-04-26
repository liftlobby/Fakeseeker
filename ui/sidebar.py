import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageDraw
import os
import logging
import sys

logger = logging.getLogger(__name__)

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
        # logger.debug(f"Accessing bundled resource from _MEIPASS: {base_path}") # Logger might not be ready here
    except Exception:
        # If not bundled, use the script's directory
        base_path = os.path.abspath(os.path.dirname(__file__))
        # logger.debug(f"Accessing resource from script path: {base_path}")
    path = os.path.join(base_path, relative_path)
    return path

class Sidebar(ttk.Frame):
    def __init__(self, parent, controller):
        logger.debug("Sidebar init start")
        super().__init__(parent)
        self.controller = controller

        button_container = ttk.Frame(self)
        button_container.pack(expand=True, fill=tk.Y, pady=(150, 150)) # Adjust padding

        icon_size = (32, 32) # Define size
        # Define relative paths
        icon_paths_rel = {
            "Home": "home.png",
            "Upload": "upload.png",
            "Real-Time": "realtime.png",
            "History": "results.png",
        }
        self.sidebar_icons = {}

        default_icon_img = Image.new("RGBA", icon_size, (200, 200, 200, 255)) # Create fallback base image
        default_icon = ImageTk.PhotoImage(default_icon_img) # Keep default TK image reference
        # Load icons
        for name, fname in icon_paths_rel.items():
            # Use resource_path to get full path
            full_path = resource_path(os.path.join('images', fname))
            icon = self._resize_icon(full_path, icon_size)
            self.sidebar_icons[name] = icon if icon else default_icon
            if icon is None:
                logger.warning(f"Using fallback icon for button: {name}")
        
        # Ensure Home icon fallback works if needed (uses default_icon reference)
        if self.sidebar_icons.get("Home") is default_icon:
            logger.warning("Using fallback icon for Home button.")

        sidebar_button_data = [
            ('Upload Image/Video', self.sidebar_icons.get('Upload'), self.controller.show_upload_page),
            ('Real-Time Detection', self.sidebar_icons.get('Real-Time'), self.controller.show_realtime_page),
            ('View Scan History', self.sidebar_icons.get('History'), self.controller.show_history_page),
            ('Home', self.sidebar_icons.get('Home'), self.controller.show_home_page)
        ]

        self.sidebar_buttons = []
        for idx, (text, icon, command) in enumerate(sidebar_button_data):
            # Icon should be a valid PhotoImage by now (loaded or default)
            btn = ttk.Button(button_container, text='', image=icon,
                             compound=tk.LEFT, command=command, width=3)
            btn.image = icon # Keep reference
            btn.pack(fill=tk.X, pady=5, padx=3)
            self.sidebar_buttons.append((btn, text))

        self.bind('<Enter>', self.expand_sidebar)
        self.bind('<Leave>', self.collapse_sidebar)
        self.collapse_sidebar(None) # Start collapsed
        logger.debug("Sidebar init end")


    def _resize_icon(self, icon_path, size):
        try:
            if os.path.exists(icon_path):
                icon = Image.open(icon_path).convert("RGBA")
                icon = icon.resize(size, Image.Resampling.LANCZOS)
                return ImageTk.PhotoImage(icon)
            else:
                logger.warning(f"Icon not found: {icon_path}")
                return None
        except Exception as e:
            logger.error(f"Error resizing icon {icon_path}: {e}", exc_info=True)
            return None

    def expand_sidebar(self, event):
        logger.debug("Sidebar init start")
        for btn, text in self.sidebar_buttons:
            if btn.winfo_exists():
                btn.config(text=text, width=20)
        self.config(width=180)


    def collapse_sidebar(self, event):
        logger.debug("Sidebar init end")
        for btn, _ in self.sidebar_buttons:
            if btn.winfo_exists():
                btn.config(text='', width=3)
        self.config(width=50)