import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os
from datetime import datetime
import sys
import logging

logger = logging.getLogger(__name__)

class HistoryPage(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.thumbnail_cache = {}
        self.logger = logger

        header = ttk.Label(self, text="Scan History", style="Header.TLabel")
        header.pack(pady=10)

        scroll_container = ttk.Frame(self)
        scroll_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        self.canvas = tk.Canvas(scroll_container, highlightthickness=0)
        scrollbar = ttk.Scrollbar(scroll_container, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.scrollable_frame = ttk.Frame(self.canvas)
        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        self.canvas.bind("<Configure>", self._on_canvas_resize_configure_scrollregion)

        # Bind mousewheel to the canvas itself
        self.canvas.bind("<MouseWheel>", self._do_scroll)
        self.canvas.bind("<Button-4>", self._do_scroll)
        self.canvas.bind("<Button-5>", self._do_scroll)

        # Initial binding for the scrollable_frame and its (currently non-existent) children
        self._bind_children_for_scrolling(self.scrollable_frame)

    def _on_canvas_resize_configure_scrollregion(self, event=None):
        if not self.canvas.winfo_exists(): return
        canvas_width = self.canvas.winfo_width()
        if self.canvas_window:
            current_item_width = self.canvas.itemcget(self.canvas_window, "width")
            if str(canvas_width) != current_item_width:
                self.canvas.itemconfig(self.canvas_window, width=canvas_width)
        self.canvas.after_idle(self._update_scrollregion)

    def _update_scrollregion(self):
        if self.canvas.winfo_exists() and self.scrollable_frame.winfo_exists():
            self.scrollable_frame.update_idletasks()
            bbox = self.canvas.bbox("all")
            if bbox: self.canvas.configure(scrollregion=bbox)

    def _do_scroll(self, event):
        current_y_view = self.canvas.yview()
        scroll_val = 0

        if sys.platform.startswith('linux'):
            if event.num == 4: scroll_val = -1
            elif event.num == 5: scroll_val = 1
            else: return "break"
        elif sys.platform == 'darwin':
            if event.delta < 0: scroll_val = 1
            elif event.delta > 0: scroll_val = -1
            else: return "break"
        else: # Windows
            if event.delta == 0: return "break"
            scroll_val = -1 * (event.delta // 120)
        
        did_scroll = False
        if scroll_val < 0 and current_y_view[0] > 0.00001:
            self.canvas.yview_scroll(scroll_val, "units")
            did_scroll = True
        elif scroll_val > 0 and current_y_view[1] < 0.99999:
            self.canvas.yview_scroll(scroll_val, "units")
            did_scroll = True
        
        return "break" # Always consume to prevent parent/other weird scroll behaviors

    def _bind_children_for_scrolling(self, parent_widget):
        parent_widget.bind("<MouseWheel>", self._do_scroll, add="+")
        parent_widget.bind("<Button-4>", self._do_scroll, add="+")
        parent_widget.bind("<Button-5>", self._do_scroll, add="+")
        for child in parent_widget.winfo_children():
            if not isinstance(child, ttk.Scrollbar): # Avoid binding to scrollbar itself
                self._bind_children_for_scrolling(child)

    def on_show(self):
        current_history = self.controller.load_scan_history()
        sorted_history = sorted(current_history, key=lambda x: x.get("timestamp", ""), reverse=True)
        self.display_history(sorted_history)
        if self.canvas.winfo_exists(): # Ensure canvas exists before yview_moveto
             self.canvas.yview_moveto(0)

    def _safe_float(self, value):
        if isinstance(value, tuple): return float(value[0])
        try: return float(value)
        except: return 0.0

    def _load_thumbnail(self, abs_path, size=(150, 150)):
        if not abs_path or not os.path.exists(abs_path):
            self.logger.warning(f"HistoryPage: Thumbnail path invalid: {abs_path}")
            return None
        cache_key = (abs_path, size)
        if cache_key in self.thumbnail_cache: return self.thumbnail_cache[cache_key]
        try:
            img = Image.open(abs_path); img.thumbnail(size, Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img); self.thumbnail_cache[cache_key] = photo
            return photo
        except FileNotFoundError: self.logger.error(f"HistoryPage: Thumb not found: {abs_path}")
        except Exception as e: self.logger.error(f"HistoryPage: Thumb load fail {abs_path}: {e}", exc_info=True)
        self.thumbnail_cache[cache_key] = None; return None

    def display_history(self, history_data):
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        self.thumbnail_cache.clear()

        if not history_data:
            ttk.Label(self.scrollable_frame, text="No scan history found.", style="Detail.TLabel").pack(pady=20)
        else:
            threshold = self.controller.detector.optimal_threshold
            columns = 3
            for i in range(columns): self.scrollable_frame.grid_columnconfigure(i, weight=1)

            for idx, scan in enumerate(history_data):
                row_idx = idx // columns
                col_idx = idx % columns
                result_card = ttk.Frame(self.scrollable_frame, padding=10, style="Card.TFrame")
                result_card.grid(row=row_idx, column=col_idx, padx=10, pady=10, sticky="nsew")
                
                detection_type = scan.get("detection_type", "scanned")
                timestamp_str = scan.get("timestamp", "Unknown Time")
                try:
                    ts_obj = datetime.strptime(timestamp_str.split('.')[0], "%Y%m%d_%H%M%S")
                    display_ts = ts_obj.strftime("%Y-%m-%d %H:%M")
                except ValueError:
                    display_ts = timestamp_str[:16] if len(timestamp_str) >=16 else timestamp_str

                ttk.Label(result_card, text=f"Scan #{len(history_data)-idx}", font=('Helvetica', 9, 'bold')).pack(anchor="nw")
                ttk.Label(result_card, text=display_ts, font=('Helvetica', 10)).pack(anchor="center", pady=2)

                card_overall_result, card_avg_prob_text, card_status_style = "Unknown", "", "Detail.TLabel"
                results_data = scan.get("results", [])
                valid_probs = [r for r in results_data if isinstance(r, (float, int))]

                if valid_probs:
                    avg_prob_for_card = sum(valid_probs) / len(valid_probs)
                    card_overall_result = "FAKE" if avg_prob_for_card >= threshold else "REAL"
                    card_status_style = "StatusFake.TLabel" if card_overall_result == "FAKE" else "StatusReal.TLabel"
                    card_avg_prob_text = f"Avg. Fake Prob: {avg_prob_for_card*100:.1f}%"

                type_display_str, additional_info = "", ""
                thumb_path_to_load = scan.get("face_thumbnails", [None])[0]

                if detection_type == "real-time-detailed" or detection_type == "real-time-summary":
                    type_display_str = "Real-Time Session"
                    summary_data = scan.get("summary", {})
                    if 'overall_result' in summary_data:
                        card_overall_result = summary_data.get('overall_result', card_overall_result)
                        card_status_style = "StatusFake.TLabel" if "FAKE" in card_overall_result else "StatusReal.TLabel"
                    if 'average_fake_probability' in summary_data:
                        card_avg_prob_text = f"Avg. Fake Prob: {summary_data['average_fake_probability']}"
                elif detection_type == "scanned":
                    type_display_str = "Uploaded Media"
                    additional_info = os.path.basename(scan.get("file_path", "")) or "Unknown File"
                else:
                    type_display_str = detection_type.replace('-', ' ').title()
                    additional_info = "Details in Report"

                ttk.Label(result_card, text=f"Type: {type_display_str}", font=('Helvetica', 10, 'italic')).pack(anchor="center", pady=1)
                if additional_info: ttk.Label(result_card, text=additional_info, font=('Helvetica', 10), wraplength=200).pack(anchor="center", pady=1)
                ttk.Label(result_card, text=f"Overall: {card_overall_result}", font=('Helvetica', 11, 'bold'), style=card_status_style).pack(anchor="center", pady=2)
                if card_avg_prob_text: ttk.Label(result_card, text=card_avg_prob_text, font=('Helvetica', 10)).pack(anchor="center", pady=1)
                
                if thumb_path_to_load:
                    photo = self._load_thumbnail(thumb_path_to_load)
                    if photo:
                        img_label = ttk.Label(result_card, image=photo); img_label.image = photo; img_label.pack(pady=5)

                action_frame = ttk.Frame(result_card)
                action_frame.pack(anchor="center", pady=5)
                ttk.Button(action_frame, text="Details", command=lambda s=scan: self.controller.show_detailed_report(s)).pack(side="left", padx=5)
                ttk.Button(action_frame, text="Delete", command=lambda s=scan: self.controller.delete_scan(s)).pack(side="left", padx=5)

        self.scrollable_frame.update_idletasks()
        self._update_scrollregion()
        self._bind_children_for_scrolling(self.scrollable_frame) # Re-bind after creating children
        if self.canvas.winfo_exists():
            self.canvas.yview_moveto(0)