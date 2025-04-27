import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os
import json
from datetime import datetime

class HistoryPage(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.thumbnail_cache = {} # Cache PhotoImage objects

        # --- Basic Layout ---
        header = ttk.Label(self, text="View Scan History", style="Header.TLabel")
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
        self.scrollable_frame = ttk.Frame(self.canvas)
        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        # Bind events for scrolling
        self.scrollable_frame.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        # Bind mouse wheel scrolling for different platforms
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)  # Windows/Mac
        self.canvas.bind("<Button-4>", self._on_mousewheel)    # Linux scroll up
        self.canvas.bind("<Button-5>", self._on_mousewheel)    # Linux scroll down

    def _on_frame_configure(self, event=None):
        """Reset the scroll region to encompass the inner frame."""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        """Resize the inner frame to match the canvas width."""
        self.canvas.itemconfig(self.canvas_window, width=event.width)

    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling, preventing scrolling past top."""
        scroll_up = (event.num == 4 or event.delta > 0)
        scroll_down = (event.num == 5 or event.delta < 0)

        if scroll_up:
            # Check if already at the top before scrolling up
            # canvas.yview() returns a tuple like (0.0, 0.123) representing the
            # fraction of the content visible, starting from the top.
            # If the first value is 0.0, we are at the top.
            current_y_view = self.canvas.yview()
            if current_y_view[0] > 0.0:
                self.canvas.yview_scroll(-1, "units")
        elif scroll_down:
            # No check needed for scrolling down past the bottom (usually handled fine)
            self.canvas.yview_scroll(1, "units")

    def _safe_float(self, value):
        """Safely convert value to float."""
        if isinstance(value, tuple): return float(value[0])
        try: return float(value)
        except: return 0.0

    def _load_thumbnail(self, rel_path, size=(150, 150)):
        """Loads and caches thumbnail images."""
        # Basic check if the provided path exists
        if not abs_path or not os.path.exists(abs_path): # Check if path is valid
            logger.warning(f"Thumbnail absolute path not found or invalid: {abs_path}")
            return None

        # Use cache if available
        cache_key = (abs_path, size)
        if cache_key in self.thumbnail_cache:
            return self.thumbnail_cache[cache_key]

        try:
            img = Image.open(abs_path)
            img.thumbnail(size, Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.thumbnail_cache[cache_key] = photo # Store in cache
            return photo
        except FileNotFoundError:
            logger.error(f"[ReportPage] Thumbnail file not found during open: {abs_path}")
            self.thumbnail_cache[cache_key] = None
            return None
        except Exception as e:
            logger.error(f"[ReportPage] Failed loading thumbnail {abs_path}: {e}", exc_info=True)
            self.thumbnail_cache[cache_key] = None
            return None

    def display_history(self, history_data):
        """Clears and rebuilds the history display."""
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        self.thumbnail_cache = {}

        if not history_data:
            ttk.Label(self.scrollable_frame, text="No detection results found.", style="Detail.TLabel").pack(pady=20)
            return

        threshold = self.controller.detector.optimal_threshold
        columns = 3

        for idx, scan in enumerate(history_data):
            row_idx = idx // columns
            col_idx = idx % columns
            detection_type = scan.get("detection_type", "scanned")
            timestamp_str = scan.get("timestamp", "Unknown Time")
            # Attempt to format timestamp
            try:
                    if '.' in timestamp_str: # Format with microseconds
                        ts_obj = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S.%f")
                    else: # Format without microseconds
                        ts_obj = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                    display_ts = ts_obj.strftime("%Y-%m-%d %H:%M") # Shorter format for card
            except ValueError:
                    display_ts = timestamp_str[:16] # Fallback to first 16 chars

            result_card = ttk.Frame(self.scrollable_frame, padding=10, style="Card.TFrame")
            result_card.grid(row=row_idx, column=col_idx, padx=10, pady=10, sticky="nsew")
            self.scrollable_frame.grid_columnconfigure(col_idx, weight=1)

            # --- Card Content ---
            ttk.Label(result_card, text=f"Scan #{len(history_data)-idx}", font=('Helvetica', 9, 'bold')).pack(anchor="nw")
            ttk.Label(result_card, text=display_ts, font=('Helvetica', 10)).pack(anchor="center", pady=2)

            if detection_type == "real-time-summary":
                type_str = "Real-Time Session"
                summary_data = scan.get("summary", {})
                avg_prob_list = scan.get("results", [0.0]) # Should contain the single avg prob
                avg_prob_norm = avg_prob_list[0] if avg_prob_list else 0.0
                avg_conf_pct = avg_prob_norm * 100
                result_label = summary_data.get('overall_result', "Unknown")
                status_style = "StatusFake.TLabel" if "Fake" in result_label else "StatusReal.TLabel"

                ttk.Label(result_card, text=f"Type: {type_str}", font=('Helvetica', 10, 'italic')).pack(anchor="center", pady=1)
                # --- APPLY STYLE HERE ---
                ttk.Label(result_card, text=f"Overall: {result_label}", font=('Helvetica', 11, 'bold'), style=status_style).pack(anchor="center", pady=2)
                ttk.Label(result_card, text=f"Avg. Fake Prob: {avg_conf_pct:.1f}%", font=('Helvetica', 10)).pack(anchor="center", pady=1)

                # Display sample image if available (using 'face_thumbnails' key for summary sample)
                sample_rel_path = scan.get("face_thumbnails", [None])[0] # Sample stored here
                if sample_rel_path:
                    photo = self._load_thumbnail(sample_rel_path)
                    if photo:
                        img_label = ttk.Label(result_card, image=photo)
                        img_label.image = photo
                        img_label.pack(pady=5)

            elif detection_type == "scanned":
                type_str = "Uploaded Media"
                file_path = scan.get("file_path", "")
                file_name = os.path.basename(file_path) if file_path else "Unknown File"
                results = scan.get("results", [])
                valid_results = [r for r in results if r is not None]
                avg_conf = (sum(valid_results) / len(valid_results) * 100) if valid_results else 0.0
                norm_conf = avg_conf / 100.0
                label = "FAKE" if norm_conf >= threshold else "REAL"
                status_style = "StatusFake.TLabel" if label == "FAKE" else "StatusReal.TLabel"

                ttk.Label(result_card, text=f"Type: {type_str}", font=('Helvetica', 10, 'italic')).pack(anchor="center", pady=1)
                ttk.Label(result_card, text=file_name, font=('Helvetica', 10), wraplength=200).pack(anchor="center", pady=1)
                ttk.Label(result_card, text=f"Overall: {label}", font=('Helvetica', 11, 'bold'), style=status_style).pack(anchor="center", pady=2)
                ttk.Label(result_card, text=f"Avg. Fake Prob: {avg_conf:.1f}%", font=('Helvetica', 10)).pack(anchor="center", pady=1)

                # Display first thumbnail preview
                thumb_rel_path = scan.get("face_thumbnails", [None])[0]
                if thumb_rel_path:
                    photo = self._load_thumbnail(thumb_rel_path)
                    if photo:
                        img_label = ttk.Label(result_card, image=photo)
                        img_label.image = photo
                        img_label.pack(pady=5)

            else: # Fallback for potentially old 'real-time' individual logs or unknown
                    type_str = detection_type.replace('-', ' ').title()
                    ttk.Label(result_card, text=f"Type: {type_str}", font=('Helvetica', 10, 'italic')).pack(anchor="center", pady=1)
                    ttk.Label(result_card, text="Details in Report", font=('Helvetica', 10)).pack(anchor="center", pady=5)
                    # Display first thumbnail if available
                    thumb_rel_path = scan.get("face_thumbnails", [None])[0]
                    if thumb_rel_path:
                        photo = self._load_thumbnail(thumb_rel_path)
                        if photo:
                            img_label = ttk.Label(result_card, image=photo)
                            img_label.image = photo
                            img_label.pack(pady=5)


            # --- Action Buttons ---
            action_frame = ttk.Frame(result_card)
            action_frame.pack(anchor="center", pady=5)
            ttk.Button(action_frame, text="Details", command=lambda s=scan: self.controller.show_detailed_report(s)).pack(side="left", padx=5)
            ttk.Button(action_frame, text="Delete", command=lambda s=scan: self.controller.delete_scan(s)).pack(side="left", padx=5)

        self.scrollable_frame.update_idletasks()
        self._on_frame_configure()