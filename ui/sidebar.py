import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageDraw
import os
import logging

logger = logging.getLogger(__name__)

class Sidebar(ttk.Frame):
    def __init__(self, parent, controller):
        logger.debug("Sidebar init start")
        super().__init__(parent)
        self.controller = controller

        self.images_dir = controller.images_dir
        self.base_dir = controller.base_dir

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
        for name, fname in icon_paths_rel.items():
            # Use resource_path to get full path
            full_path = resource_path(os.path.join('images', fname))
            self.sidebar_icons[name] = self._resize_icon(full_path, icon_size)

        sidebar_button_data = [
            ('Upload Image/Video', self.sidebar_icons['Upload'], self.controller.show_upload_page),
            ('Real-Time Detection', self.sidebar_icons['Real-Time'], self.controller.show_realtime_page),
            ('View Scan History', self.sidebar_icons['History'], self.controller.show_history_page),
            ('Home', self.sidebar_icons['Home'], self.controller.show_home_page) # Add a Home button?
        ]

        self.sidebar_buttons = []
        default_icon_img = Image.new("RGBA", icon_size, (200, 200, 200, 255))
        default_icon = ImageTk.PhotoImage(default_icon_img)

        # Load Home icon if it exists
        home_icon_path = os.path.join(self.images_dir, 'home.png') # Assuming home.png exists
        home_icon = self._resize_icon(home_icon_path, icon_size)
        if not home_icon:
            # Basic fallback shape if home.png is missing
            draw = ImageDraw.Draw(default_icon_img)
            # Corrected polygon points for a simple house shape
            half_w = icon_size[0] // 2
            half_h = icon_size[1] // 2
            draw.polygon([
                (half_w, 2), # Top point
                (2, half_h - 2), # Top-left roof
                (icon_size[0] - 2, half_h - 2), # Top-right roof
            ], outline='gray', fill=None)
            draw.rectangle([
                (4, half_h - 1),
                (icon_size[0] - 4, icon_size[1] - 2)
            ], outline='gray', fill=None)
            default_icon = ImageTk.PhotoImage(default_icon_img)
            home_icon = default_icon # Use fallback


        sidebar_button_data[3] = ('Home', home_icon, self.controller.show_home_page)


        for idx, (text, icon, command) in enumerate(sidebar_button_data):
            # Ensure icon is valid PhotoImage
            current_icon = icon if icon else default_icon
            if not isinstance(current_icon, ImageTk.PhotoImage):
                 current_icon = default_icon # Fallback if resize failed

            btn = ttk.Button(button_container, text='', image=current_icon,
                             compound=tk.LEFT, command=command, width=3)
            btn.image = current_icon # Keep reference
            btn.pack(fill=tk.X, pady=5, padx=3)
            self.sidebar_buttons.append((btn, text))

        self.bind('<Enter>', self.expand_sidebar)
        self.bind('<Leave>', self.collapse_sidebar)
        self.collapse_sidebar(None) # Start collapsed

    def _resize_icon(self, icon_path, size):
        try:
            if os.path.exists(icon_path):
                icon = Image.open(icon_path).convert("RGBA")
                icon = icon.resize(size, Image.Resampling.LANCZOS)
                return ImageTk.PhotoImage(icon)
            else:
                logger.warning(f"Icon not found: {icon_path}") # Now logger exists
                return None
        except Exception as e:
            logger.error(f"Error resizing icon {icon_path}: {e}", exc_info=True) # Now logger exists
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