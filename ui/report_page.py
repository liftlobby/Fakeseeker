import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import os
import logging
import sys

logger = logging.getLogger(__name__)

class ReportPage(ttk.Frame):
    def __init__(self, parent, controller):
        self._report_bg_color = "#F0F0F0"
        try:
            if hasattr(controller, 'style') and controller.style:
                self._report_bg_color = controller.style.lookup("ReportPage.TFrame", "background")
        except tk.TclError:
            logger.warning("Could not look up ReportPage.TFrame background style, using default #F0F0F0.")
        
        super().__init__(parent, style="ReportPage.TFrame") 
        self.controller = controller
        self.thumbnail_cache = {}
        self.logger = logger

        scroll_container = ttk.Frame(self, style="ReportPage.TFrame")
        scroll_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        self.canvas = tk.Canvas(scroll_container, highlightthickness=0, background=self._report_bg_color) 
        scrollbar = ttk.Scrollbar(scroll_container, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.scrollable_frame = ttk.Frame(self.canvas, style="ReportPage.TFrame")
        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        self.canvas.bind("<Configure>", self._on_canvas_resize_configure_scrollregion)

        self.canvas.bind("<MouseWheel>", self._do_scroll)
        self.canvas.bind("<Button-4>", self._do_scroll)
        self.canvas.bind("<Button-5>", self._do_scroll)

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
        
        return "break"

    def _bind_children_for_scrolling(self, parent_widget):
        parent_widget.bind("<MouseWheel>", self._do_scroll, add="+")
        parent_widget.bind("<Button-4>", self._do_scroll, add="+")
        parent_widget.bind("<Button-5>", self._do_scroll, add="+")
        for child in parent_widget.winfo_children():
            if not isinstance(child, ttk.Scrollbar):
                self._bind_children_for_scrolling(child)

    def on_show(self):
        if self.canvas.winfo_exists():
            self.canvas.yview_moveto(0)

    def on_hide(self):
        pass

    def _load_thumbnail(self, abs_path, size=(150, 150)):
        if not abs_path or not os.path.exists(abs_path):
            logger.warning(f"ReportPage: Thumbnail path invalid: {abs_path}")
            return None
        cache_key = (abs_path, size)
        if cache_key in self.thumbnail_cache: return self.thumbnail_cache[cache_key]
        try:
            img = Image.open(abs_path); img.thumbnail(size, Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img); self.thumbnail_cache[cache_key] = photo
            return photo
        except FileNotFoundError: logger.error(f"[ReportPage] Thumb not found: {abs_path}"); self.thumbnail_cache[cache_key] = None; return None
        except Exception as e: logger.error(f"[ReportPage] Thumb load fail {abs_path}: {e}", exc_info=True); self.thumbnail_cache[cache_key] = None; return None

    def display_report(self, scan_data):
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        self.thumbnail_cache = {}
        
        effective_bg_color = self._report_bg_color
        content_frame = ttk.Frame(self.scrollable_frame, padding="15", style="ReportContent.TFrame")
        content_frame.pack(fill=tk.BOTH, expand=True)

        header = ttk.Label(content_frame, text="Detailed Scan Report", style='Header.TLabel', background=effective_bg_color, anchor=tk.CENTER)
        header.pack(pady=15, fill=tk.X)

        results_outer_frame = ttk.LabelFrame(content_frame, text="Detection Summary", padding="15", style="ReportSection.TLabelframe")
        results_outer_frame.pack(fill=tk.X, padx=20, pady=10)

        results_list = scan_data.get('results', [])
        valid_results = [r for r in results_list if isinstance(r, (float, int))]
        avg_prob = sum(valid_results) / len(valid_results) if valid_results else 0.0
        threshold = self.controller.detector.optimal_threshold
        is_fake = avg_prob >= threshold
        status = "⚠ Potential Deepfake" if is_fake else "✅ Likely Real"
        status_style_name = "StatusFake.TLabel" if is_fake else "StatusReal.TLabel"
        
        ttk.Label(results_outer_frame, text=f"Overall Status: {status}", style=status_style_name, background=effective_bg_color).pack(pady=5)
        avg_conf_percent = avg_prob * 100
        ttk.Label(results_outer_frame, text=f"Average Fake Probability: {avg_conf_percent:.2f}% (Threshold: {threshold*100:.2f}%)", style="Detail.TLabel", background=effective_bg_color).pack(pady=5)
        if len(valid_results) != len(results_list):
             failed_count = len(results_list) - len(valid_results)
             ttk.Label(results_outer_frame, text=f"({failed_count} face(s) failed analysis)", style="Detail.TLabel", background=effective_bg_color, foreground="grey").pack(pady=2)

        detection_type = scan_data.get("detection_type", "scanned")
        if detection_type == 'scanned' and 'file_path' in scan_data and scan_data['file_path']:
            details_frame = ttk.LabelFrame(content_frame, text="File Details", padding="15", style="ReportSection.TLabelframe")
            details_frame.pack(fill=tk.X, padx=20, pady=10)
            try:
                file_details = self.controller.get_file_details(scan_data['file_path'])
                for key, value in file_details.items():
                    ttk.Label(details_frame, text=f"{key.capitalize()}: {value}", style="Detail.TLabel", background=effective_bg_color).pack(anchor="w", padx=10, pady=2)
            except Exception as e:
                logger.error(f"Report: Error getting file details: {e}", exc_info=True)
                ttk.Label(details_frame, text="Error getting file details.", foreground="red", background=effective_bg_color).pack(anchor="w", padx=10, pady=2)

        if detection_type in ["real-time-summary", "real-time-detailed"]:
            summary_text = "Real-Time Session Info"
            if detection_type == "real-time-summary" and not scan_data.get('results', []): summary_text = "Real-Time Session Summary (Overall)"
            summary_frame = ttk.LabelFrame(content_frame, text=summary_text, padding="15", style="ReportSection.TLabelframe")
            summary_frame.pack(fill=tk.X, padx=20, pady=10)
            summary = scan_data.get("summary", {})
            ttk.Label(summary_frame, text=f"Overall Session Result: {summary.get('overall_result', 'N/A')}", style="Detail.TLabel", background=effective_bg_color).pack(anchor="w", padx=10, pady=2)
            ttk.Label(summary_frame, text=f"Total Faces Analyzed/Stored: {summary.get('total_faces_processed', 'N/A')}", style="Detail.TLabel", background=effective_bg_color).pack(anchor="w", padx=10, pady=2)
            ttk.Label(summary_frame, text=f"Real Detections (based on stored): {summary.get('real_detections', 'N/A')}", style="Detail.TLabel", background=effective_bg_color).pack(anchor="w", padx=10, pady=2)
            ttk.Label(summary_frame, text=f"Fake Detections (based on stored): {summary.get('fake_detections', 'N/A')}", style="Detail.TLabel", background=effective_bg_color).pack(anchor="w", padx=10, pady=2)

        if detection_type == "real-time-detailed": faces_frame_text = "Detected Faces (Real-Time - Individual Results)"
        elif detection_type == "scanned": faces_frame_text = "Detected Faces Analysis (Individual Results)"
        elif detection_type == "real-time-summary": faces_frame_text = "Session Face Samples (No Individual Results)"
        else: faces_frame_text = f"Detected Faces ({detection_type})"
        faces_frame = ttk.LabelFrame(content_frame, text=faces_frame_text, padding="10", style="ReportSection.TLabelframe")
        faces_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        grid_frame = ttk.Frame(faces_frame, style="ReportContent.TFrame")
        grid_frame.pack(pady=5)
        face_thumbnails = scan_data.get('face_thumbnails', [])
        
        if not face_thumbnails: ttk.Label(grid_frame, text="No face thumbnails available.", style='Detail.TLabel', background=effective_bg_color).pack()
        else:
            max_cols = 4
            for i, abs_path in enumerate(face_thumbnails):
                row_idx, col_idx = i // max_cols, i % max_cols
                face_container = ttk.Frame(grid_frame, padding=5, style="ReportContent.TFrame")
                face_container.grid(row=row_idx, column=col_idx, padx=5, pady=5, sticky="n")
                photo = self._load_thumbnail(abs_path, size=(120, 120))
                if photo:
                    img_label = ttk.Label(face_container, image=photo); img_label.image = photo; img_label.pack(pady=2)
                    if detection_type in ['scanned', 'real-time-detailed'] and i < len(results_list):
                        prob = results_list[i]
                        if prob is not None:
                            face_status = "FAKE" if prob >= threshold else "REAL"
                            color = "red" if face_status == "FAKE" else "green"
                            ttk.Label(face_container, text=f"{face_status} ({prob * 100:.1f}%)", font=('Helvetica', 10), foreground=color, background=effective_bg_color).pack(pady=1)
                        else: ttk.Label(face_container, text="Analysis Failed", font=('Helvetica', 10, 'italic'), background=effective_bg_color, foreground="grey").pack(pady=1)
                    elif detection_type == 'real-time-summary': ttk.Label(face_container, text=f"Face #{i+1}", font=('Helvetica', 9, 'italic'), background=effective_bg_color, foreground="grey").pack(pady=1)
                else: ttk.Label(face_container, text="[Image N/A]", background=effective_bg_color, foreground="grey").pack(pady=5, ipadx=10, ipady=10)

        self.scrollable_frame.update_idletasks()
        self._update_scrollregion()
        self._bind_children_for_scrolling(self.scrollable_frame)
        if self.canvas.winfo_exists():
            self.canvas.yview_moveto(0)