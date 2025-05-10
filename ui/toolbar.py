# --- toolbar.py ---
import tkinter as tk
from tkinter import ttk
import logging # Add logging

logger = logging.getLogger(__name__) # Get logger

class FloatingToolbar(tk.Toplevel):
    def __init__(self, parent, app_controller, **kwargs):
        logger.debug("Initializing FloatingToolbar")
        super().__init__(parent, **kwargs)
        self.app = app_controller # Reference to FakeSeekerApp instance

        self.title("Screen Monitor Controls") # Updated title
        self.geometry("350x150") # Keep adjusted size or make configurable
        self.attributes('-topmost', True)
        self.resizable(True, True)
        self.protocol("WM_DELETE_WINDOW", self.handle_close) # Use internal method

        # --- Frame for ALL buttons ---
        button_frame = ttk.Frame(self)
        # Adjust padding if needed now that video is gone
        button_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5, padx=5, expand=True)
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)
        button_frame.columnconfigure(2, weight=1)
        button_frame.columnconfigure(3, weight=1)

        # --- Buttons ---
        self.start_btn = ttk.Button(button_frame, text="Start Detection", command=self.app.start_detection, style="Toolbar.TButton")
        self.start_btn.grid(row=0, column=0, columnspan=2, padx=2, pady=2, sticky="ew") # Span 2 columns

        self.stop_btn = ttk.Button(button_frame, text="Stop Detection", command=self.app.stop_detection, style="Toolbar.TButton")
        self.stop_btn.grid(row=0, column=2, columnspan=2, padx=2, pady=2, sticky="ew") # Span 2 columns

        self.notify_btn = ttk.Button(button_frame, text="Show Status", command=self.app.show_notifications, style="Toolbar.TButton")
        self.notify_btn.grid(row=1, column=0, columnspan=2, padx=2, pady=2, sticky="ew") # Row 1

        self.show_main_btn = ttk.Button(button_frame, text="Show Main Window", command=self.handle_close, style="Toolbar.TButton")
        self.show_main_btn.grid(row=1, column=2, columnspan=2, padx=2, pady=2, sticky="ew") # Row 1

        self.update_button_states() # Update initial state based on app controller

    def update_button_states(self):
        """Updates button states based on the main app's state."""
        logger.debug("Toolbar updating button states...")
        try:
            start_state = 'disabled' if self.app.detection_active else 'normal'
            stop_state = 'normal' if self.app.detection_active else 'disabled'
            # Notification button can always be active when toolbar is visible
            notify_state = 'normal'
            show_main_state = 'normal' # Always allow returning to main

            if hasattr(self, 'start_btn') and self.start_btn.winfo_exists():
                self.start_btn.config(state=start_state)
            if hasattr(self, 'stop_btn') and self.stop_btn.winfo_exists():
                self.stop_btn.config(state=stop_state)
            if hasattr(self, 'notify_btn') and self.notify_btn.winfo_exists():
                self.notify_btn.config(state=notify_state)
            if hasattr(self, 'show_main_btn') and self.show_main_btn.winfo_exists():
                 self.show_main_btn.config(state=show_main_state)

        except tk.TclError:
            logger.warning("TclError during toolbar button update (likely closing).")
        except Exception as e:
            logger.error(f"Unexpected error updating toolbar buttons: {e}", exc_info=True)


    def handle_close(self):
        """Handles closing the toolbar window (via X or Show Main button)."""
        logger.info("Toolbar handle_close called.")
        # This method should trigger the main app to stop monitoring AND show itself
        self.app.handle_toolbar_close() # Delegate responsibility to main app