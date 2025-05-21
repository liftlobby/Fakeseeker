import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os
import logging
import sys

logger = logging.getLogger(__name__)

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except AttributeError: # More specific exception for _MEIPASS not existing
        # If not bundled, __file__ is the path to sidebar.py (inside ui folder)
        script_dir = os.path.abspath(os.path.dirname(__file__))
        # Go one level up from 'ui' to get to 'fakeseeker_app' (the project root)
        base_path = os.path.dirname(script_dir)
    except Exception as e: # Catch other potential errors during MEIPASS access
        logger.error(f"Error determining base_path, falling back: {e}")
        script_dir = os.path.abspath(os.path.dirname(__file__))
        base_path = os.path.dirname(script_dir) # Fallback again

    return os.path.join(base_path, relative_path)

class Sidebar(ttk.Frame):
    def __init__(self, parent, controller):
        logger.debug("Sidebar init start")
        super().__init__(parent)
        self.controller = controller

        self._expand_job = None
        self._collapse_job = None
        self._debounce_time = 200 # Milliseconds for debouncing

        self.collapsed_width = 50
        self.expanded_width = 180
        self.animation_duration = 200  # milliseconds for the whole animation
        self.animation_steps = 10      # number of steps in the animation
        self.current_animation_job = None
        self.is_expanded = False # Track current state

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

        default_pil_img = Image.new("RGBA", icon_size, (200, 200, 200, 128)) # Slightly transparent gray
        self.default_icon_photoimage = ImageTk.PhotoImage(default_pil_img) # Keep default TK image reference
        
        # Load icons
        for name, fname in icon_paths_rel.items():
            full_path = resource_path(os.path.join('images', fname))
            icon_photo = self._resize_icon(full_path, icon_size)
            self.sidebar_icons[name] = icon_photo if icon_photo else self.default_icon_photoimage
            if icon_photo is None: # Log if specific icon failed, not just if using fallback
                logger.warning(f"Fallback icon used for sidebar button: {name} (Path: {full_path})")

        sidebar_button_data = [
            ('Upload Image/Video', self.sidebar_icons.get('Upload'), self.controller.show_upload_page),
            ('Real-Time Detection', self.sidebar_icons.get('Real-Time'), self.controller.show_realtime_page),
            ('View Scan History', self.sidebar_icons.get('History'), self.controller.show_history_page),
            ('Home', self.sidebar_icons.get('Home'), self.controller.show_home_page)
        ]

        self.sidebar_buttons = []

        for text, icon, command in sidebar_button_data:
            btn = ttk.Button(button_container, text='', image=icon,
                             compound=tk.LEFT, command=command, width=3, style="Sidebar.TButton") # Added a potential style
            btn.image = icon # Keep reference
            btn.pack(fill=tk.X, pady=5, padx=3)
            self.sidebar_buttons.append((btn, text))

        self.bind('<Enter>', self._schedule_expand)
        self.bind('<Leave>', self._schedule_collapse)

        self.config(width=self.collapsed_width) # Initial state
        self._update_button_text_visibility() # Set initial button text based on collapsed stat
        logger.debug("Sidebar init end")

    def _schedule_expand(self, event=None): # event is passed by bind but not used by target
        if self._collapse_job:
            self.after_cancel(self._collapse_job)
            self._collapse_job = None
        if self._expand_job:
            self.after_cancel(self._expand_job)
        self._expand_job = self.after(self._debounce_time, self.expand_sidebar)

    def _schedule_collapse(self, event=None): # event is passed by bind but not used by target
        if self._expand_job:
            self.after_cancel(self._expand_job)
            self._expand_job = None
        if self._collapse_job:
            self.after_cancel(self.collapse_job)
        self._collapse_job = self.after(self._debounce_time, self.collapse_sidebar)

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

    def _schedule_start_animation_expand(self, event=None):
        if self._collapse_job: # If a collapse animation was scheduled, cancel it
            self.after_cancel(self._collapse_job)
            self._collapse_job = None
        if self.current_animation_job: # If an animation is already running, cancel it
             self.after_cancel(self.current_animation_job)
             self.current_animation_job = None

        if self._expand_job: # If an expand trigger was already scheduled, cancel it
            self.after_cancel(self._expand_job)
        # Schedule the start of the expand animation
        self._expand_job = self.after(self._trigger_debounce_time, lambda: self._animate_sidebar(True))

    def _schedule_start_animation_collapse(self, event=None):
        if self._expand_job: # If an expand animation was scheduled, cancel it
            self.after_cancel(self._expand_job)
            self._expand_job = None
        if self.current_animation_job: # If an animation is already running, cancel it
             self.after_cancel(self.current_animation_job)
             self.current_animation_job = None

        if self._collapse_job: # If a collapse trigger was already scheduled, cancel it
            self.after_cancel(self._collapse_job)
        # Schedule the start of the collapse animation
        self._collapse_job = self.after(self._trigger_debounce_time, lambda: self._animate_sidebar(False))


    def _animate_sidebar(self, expand, current_step=0):
        # Cancel any pending trigger jobs
        if self._expand_job: self.after_cancel(self._expand_job); self._expand_job = None
        if self._collapse_job: self.after_cancel(self._collapse_job); self._collapse_job = None

        # If an animation is already running in the opposite direction, stop it.
        # Or, if we are already in the target state, do nothing.
        if self.current_animation_job and self.is_expanded == expand:
            # Already animating towards or in the target state, could refine this logic
            # For now, if we are asked to expand and already expanded (or animating to expand), do nothing.
            return
        if self.is_expanded == expand and current_step == 0: # Already in target state
             return

        # If starting a new animation, cancel any existing one
        if current_step == 0 and self.current_animation_job:
            self.after_cancel(self.current_animation_job)
            self.current_animation_job = None

        start_width = self.winfo_width() # Get current actual width
        target_width = self.expanded_width if expand else self.collapsed_width

        if current_step <= self.animation_steps:
            # Linear interpolation for width
            progress = current_step / self.animation_steps
            new_width = int(start_width + (target_width - start_width) * progress)
            
            # If very close to target, just snap to it to finish
            if abs(new_width - target_width) < abs(target_width - start_width)/self.animation_steps :
                 new_width = target_width

            if self.winfo_exists():
                self.config(width=new_width)

            # Update button text visibility (show text only when mostly expanded)
            if expand and new_width > (self.collapsed_width + (self.expanded_width - self.collapsed_width) * 0.75):
                self._update_button_text_visibility(show_text=True)
            elif not expand and new_width < (self.collapsed_width + (self.expanded_width - self.collapsed_width) * 0.25):
                self._update_button_text_visibility(show_text=False)
            elif not expand and new_width == self.collapsed_width: # Ensure text is hidden when fully collapsed
                 self._update_button_text_visibility(show_text=False)


            delay_per_step = self.animation_duration // self.animation_steps
            self.current_animation_job = self.after(
                delay_per_step,
                lambda: self._animate_sidebar(expand, current_step + 1)
            )
        else:
            # Animation finished, ensure final state
            if self.winfo_exists():
                self.config(width=target_width)
            self.is_expanded = expand
            self._update_button_text_visibility(show_text=self.is_expanded)
            self.current_animation_job = None
            logger.debug(f"Sidebar animation finished. Expanded: {self.is_expanded}")

    def _update_button_text_visibility(self, show_text=None):
        """Shows or hides text on buttons based on show_text flag or self.is_expanded."""
        if show_text is None:
            should_show_text = self.is_expanded
        else:
            should_show_text = show_text

        for btn, text in self.sidebar_buttons:
            if btn.winfo_exists():
                btn.config(text=text if should_show_text else '', compound=tk.LEFT)
                # Adjust button width if text is shown/hidden
                # This part might need fine-tuning for your TButton style
                btn.config(width=20 if should_show_text else 3)

    def expand_sidebar(self):
        logger.debug("Sidebar expanding")
        for btn, text in self.sidebar_buttons:
            if btn.winfo_exists(): # Good practice to check if widget exists
                btn.config(text=text, width=20)
        if self.winfo_exists(): # Check if sidebar itself exists
            self.config(width=180)
        self._expand_job = None

    def collapse_sidebar(self):
        logger.debug("Sidebar collapsing")
        for btn, _ in self.sidebar_buttons:
            if btn.winfo_exists():
                btn.config(text='', width=3)
        if self.winfo_exists():
            self.config(width=50)
        self._collapse_job = None 