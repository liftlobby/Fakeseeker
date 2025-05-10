import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os
import json
from datetime import datetime
import sys
import logging

logger = logging.getLogger(__name__)

class HistoryPage(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.thumbnail_cache = {} # Cache PhotoImage objects
        self.logger = logger

        # --- Basic Layout ---
        header = ttk.Label(self, text="Scan History", style="Header.TLabel")
        header.pack(pady=10)

        # Frame to contain the canvas and scrollbar
        scroll_container = ttk.Frame(self)
        scroll_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        self.canvas = tk.Canvas(scroll_container, highlightthickness=0)
        scrollbar = ttk.Scrollbar(scroll_container, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        # This frame holds the actual content (history cards)
        self.scrollable_frame = ttk.Frame(self.canvas) # No specific style, will inherit from parent or be default
        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        # Bind events for scrolling
        self.scrollable_frame.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        self.canvas.bind("<MouseWheel>", self._on_mousewheel) # For direct canvas focus
        self.canvas.bind("<Button-4>", self._on_mousewheel)   # Linux scroll up
        self.canvas.bind("<Button-5>", self._on_mousewheel)   # Linux scroll down

        # Bind to the scrollable_frame to ensure scrolling when mouse is over content
        self.scrollable_frame.bind("<Enter>", self._bind_mousewheel_globally)
        self.scrollable_frame.bind("<Leave>", self._unbind_mousewheel_globally)

    def _bind_mousewheel_globally(self, event):
        """Bind mousewheel events globally when mouse enters the scrollable area."""
        # Using bind_all ensures that the canvas's _on_mousewheel gets the event
        # even if another widget within scrollable_frame has focus.
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)

    def _unbind_mousewheel_globally(self, event):
        """Unbind global mousewheel events when mouse leaves the scrollable area."""
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")

    def _on_frame_configure(self, event=None):
        """Reset the scroll region to encompass the inner frame."""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        """Resize the inner frame to match the canvas width."""
        self.canvas.itemconfig(self.canvas_window, width=event.width)

    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling."""
        # Determine scroll direction and amount
        if sys.platform.startswith('linux'): # Linux uses event.num
            if event.num == 4: # Scroll up
                scroll_val = -1
            elif event.num == 5: # Scroll down
                scroll_val = 1
            else:
                return # Should not happen
        elif sys.platform == 'darwin': # macOS specific delta
            scroll_val = -1 * event.delta # macOS delta is often 1 or -1 per notch
        else: # Windows and other platforms (event.delta is usually +/-120)
            if event.delta == 0: return # Avoid division by zero if delta is unexpectedly 0
            scroll_val = -1 * (event.delta // 120)

        current_y_view = self.canvas.yview()
        if scroll_val < 0: # Scrolling up
            if current_y_view[0] > 0.0001: # Add a small tolerance for floating point
                self.canvas.yview_scroll(scroll_val, "units")
        elif scroll_val > 0: # Scrolling down
            if current_y_view[1] < 0.9999: # Add a small tolerance
                self.canvas.yview_scroll(scroll_val, "units")

    def _safe_float(self, value):
        """Safely convert value to float."""
        if isinstance(value, tuple): return float(value[0])
        try: return float(value)
        except: return 0.0

    def _load_thumbnail(self, abs_path, size=(150, 150)):
        """Loads and caches thumbnail images."""
        if not abs_path or not os.path.exists(abs_path):
            self.logger.warning(f"HistoryPage: Thumbnail absolute path not found or invalid: {abs_path}")
            return None
        cache_key = (abs_path, size)
        if cache_key in self.thumbnail_cache:
            return self.thumbnail_cache[cache_key]
        try:
            img = Image.open(abs_path)
            img.thumbnail(size, Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.thumbnail_cache[cache_key] = photo
            return photo
        except FileNotFoundError:
            self.logger.error(f"HistoryPage: Thumbnail file not found during open: {abs_path}")
        except Exception as e:
            self.logger.error(f"HistoryPage: Failed loading thumbnail {abs_path}: {e}", exc_info=True)
        self.thumbnail_cache[cache_key] = None # Cache failure
        return None

    def display_history(self, history_data):
        """Clears and rebuilds the history display."""
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        self.thumbnail_cache = {}

        if not history_data:
            ttk.Label(self.scrollable_frame, text="No scan history found.", style="Detail.TLabel").pack(pady=20)
            # Still need to configure scrollregion even if empty, or it might keep old size
            self.scrollable_frame.update_idletasks()
            self._on_frame_configure()
            return

        threshold = self.controller.detector.optimal_threshold
        columns = 3 # Max columns for cards

        # Configure columns in the scrollable_frame to expand
        for i in range(columns):
            self.scrollable_frame.grid_columnconfigure(i, weight=1)

        for idx, scan in enumerate(history_data):
            row_idx = idx // columns
            col_idx = idx % columns
            
            detection_type = scan.get("detection_type", "scanned")
            timestamp_str = scan.get("timestamp", "Unknown Time")
            try:
                ts_obj = datetime.strptime(timestamp_str.split('.')[0], "%Y%m%d_%H%M%S") # Handle potential microseconds more robustly
                display_ts = ts_obj.strftime("%Y-%m-%d %H:%M")
            except ValueError:
                display_ts = timestamp_str[:16]

            result_card = ttk.Frame(self.scrollable_frame, padding=10, style="Card.TFrame")
            result_card.grid(row=row_idx, column=col_idx, padx=10, pady=10, sticky="nsew")
            
            # --- Card Content ---
            ttk.Label(result_card, text=f"Scan #{len(history_data)-idx}", font=('Helvetica', 9, 'bold')).pack(anchor="nw")
            ttk.Label(result_card, text=display_ts, font=('Helvetica', 10)).pack(anchor="center", pady=2)

            # Determine overall result and style for the card header
            card_overall_result = "Unknown"
            card_avg_prob_text = ""
            card_status_style = "Detail.TLabel" # Default style
            
            results_data = scan.get("results", [])
            valid_probs = [r for r in results_data if isinstance(r, (float, int))]

            if valid_probs:
                avg_prob_for_card = sum(valid_probs) / len(valid_probs)
                card_overall_result = "FAKE" if avg_prob_for_card >= threshold else "REAL"
                card_status_style = "StatusFake.TLabel" if card_overall_result == "FAKE" else "StatusReal.TLabel"
                card_avg_prob_text = f"Avg. Fake Prob: {avg_prob_for_card*100:.1f}%"

            if detection_type == "real-time-detailed" or detection_type == "real-time-summary":
                type_str = "Real-Time Session"
                summary_data = scan.get("summary", {}) # Summary dict still exists for overall counts etc.
                # If overall_result is in summary (from new saving logic), prefer it
                # Otherwise, calculate from results as done above
                if 'overall_result' in summary_data:
                    card_overall_result = summary_data.get('overall_result', card_overall_result)
                    card_status_style = "StatusFake.TLabel" if "FAKE" in card_overall_result else "StatusReal.TLabel"
                if 'average_fake_probability' in summary_data: # Text like "XX.X%"
                     card_avg_prob_text = f"Avg. Fake Prob: {summary_data['average_fake_probability']}"

                ttk.Label(result_card, text=f"Type: {type_str}", font=('Helvetica', 10, 'italic')).pack(anchor="center", pady=1)
                ttk.Label(result_card, text=f"Overall: {card_overall_result}", font=('Helvetica', 11, 'bold'), style=card_status_style).pack(anchor="center", pady=2)
                if card_avg_prob_text:
                    ttk.Label(result_card, text=card_avg_prob_text, font=('Helvetica', 10)).pack(anchor="center", pady=1)
                
                sample_rel_path = scan.get("face_thumbnails", [None])[0]
                if sample_rel_path:
                    photo = self._load_thumbnail(sample_rel_path)
                    if photo:
                        img_label = ttk.Label(result_card, image=photo); img_label.image = photo; img_label.pack(pady=5)

            elif detection_type == "scanned":
                type_str = "Uploaded Media"
                file_name = os.path.basename(scan.get("file_path", "")) or "Unknown File"
                
                ttk.Label(result_card, text=f"Type: {type_str}", font=('Helvetica', 10, 'italic')).pack(anchor="center", pady=1)
                ttk.Label(result_card, text=file_name, font=('Helvetica', 10), wraplength=200).pack(anchor="center", pady=1)
                ttk.Label(result_card, text=f"Overall: {card_overall_result}", font=('Helvetica', 11, 'bold'), style=card_status_style).pack(anchor="center", pady=2)
                if card_avg_prob_text:
                    ttk.Label(result_card, text=card_avg_prob_text, font=('Helvetica', 10)).pack(anchor="center", pady=1)

                thumb_rel_path = scan.get("face_thumbnails", [None])[0]
                if thumb_rel_path:
                    photo = self._load_thumbnail(thumb_rel_path)
                    if photo:
                        img_label = ttk.Label(result_card, image=photo); img_label.image = photo; img_label.pack(pady=5)
            else:
                # Fallback for any other unknown types
                type_str = detection_type.replace('-', ' ').title()
                ttk.Label(result_card, text=f"Type: {type_str}", font=('Helvetica', 10, 'italic')).pack(anchor="center", pady=1)
                ttk.Label(result_card, text="Details in Report", font=('Helvetica', 10)).pack(anchor="center", pady=5)
                thumb_rel_path = scan.get("face_thumbnails", [None])[0]
                if thumb_rel_path:
                    photo = self._load_thumbnail(thumb_rel_path);
                    if photo: img_label = ttk.Label(result_card, image=photo); img_label.image = photo; img_label.pack(pady=5)

            action_frame = ttk.Frame(result_card)
            action_frame.pack(anchor="center", pady=5)
            ttk.Button(action_frame, text="Details", command=lambda s=scan: self.controller.show_detailed_report(s)).pack(side="left", padx=5)
            ttk.Button(action_frame, text="Delete", command=lambda s=scan: self.controller.delete_scan(s)).pack(side="left", padx=5)

        self.scrollable_frame.update_idletasks()
        self._on_frame_configure()
        self.canvas.yview_moveto(0) # Ensure scroll to top after display