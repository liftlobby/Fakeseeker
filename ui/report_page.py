import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import os
import logging # Import logging

logger = logging.getLogger(__name__) # Get logger instance

class ReportPage(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent) # Correct call to parent __init__
        self.controller = controller
        self.thumbnail_cache = {} # Cache for face thumbnails

        # --- Basic Layout ---
        scroll_container = ttk.Frame(self)
        scroll_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        self.canvas = tk.Canvas(scroll_container, highlightthickness=0, background="#FFFFFF") # White background
        scrollbar = ttk.Scrollbar(scroll_container, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        # This frame holds the actual content
        self.scrollable_frame = ttk.Frame(self.canvas, style="WhiteBackground.TFrame") # Use white background style
        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        # Bind events for layout updates
        self.scrollable_frame.bind("<Configure>", self._on_frame_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel)    # Linux scroll up
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)   # Linux scroll down

    def on_show(self):
        """Called when the page is raised."""
        # Reset scroll position to top
        self.canvas.yview_moveto(0)

    def on_hide(self):
         """Called when switching away from the page (if needed)."""
         # No specific action needed currently
         pass

    def _on_frame_configure(self, event=None):
        """Updates the scroll region when the inner frame size changes."""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        """Adjusts the inner frame width to match the canvas width."""
        self.canvas.itemconfig(self.canvas_window, width=event.width)

    def _on_mousewheel(self, event):
        """Handle mouse wheel scrolling, preventing scrolling past top."""
        scroll_up = (event.num == 4 or event.delta > 0)
        scroll_down = (event.num == 5 or event.delta < 0)

        # Get current view BEFORE scrolling
        current_y_view = self.canvas.yview()

        if scroll_up:
            if current_y_view[0] > 0.0:
                self.canvas.yview_scroll(-1, "units")
        elif scroll_down:
            # Check if not already at the bottom (optional, but good practice)
            if current_y_view[1] < 1.0:
                 self.canvas.yview_scroll(1, "units")

    def _load_thumbnail(self, abs_path, size=(150, 150)):
        """Loads and caches thumbnail images."""
        # Basic check if the provided path exists
        if not abs_path or not os.path.exists(abs_path): # Check if path is valid
            logger.warning(f"Thumbnail absolute path not found or invalid: {abs_path}")
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
        except FileNotFoundError: # More specific error
            logger.error(f"[ReportPage] Thumbnail file not found during open: {abs_path}")
            self.thumbnail_cache[cache_key] = None
            return None
        except Exception as e:
            # Use logger for errors
            logger.error(f"[ReportPage] Failed loading thumbnail {abs_path}: {e}", exc_info=True)
            self.thumbnail_cache[cache_key] = None
            return None

    def display_report(self, scan_data):
        """Clears and rebuilds the report display."""
        # Clear previous content FIRST
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        self.thumbnail_cache = {} # Clear cache for new report

        # --- Create the main content frame INSIDE scrollable_frame ---
        content_frame = ttk.Frame(self.scrollable_frame, padding="15", style="WhiteBackground.TFrame")
        # Use grid for better control within scrollable_frame if needed later, but pack is fine for single column
        content_frame.pack(fill=tk.BOTH, expand=True)
        # Make the content frame's column stretch if using grid later
        # content_frame.grid_columnconfigure(0, weight=1)

        # --- Header (Parent is content_frame) ---
        header = ttk.Label(content_frame, text="Detailed Scan Report",
                           style='Header.TLabel', background="#FFFFFF", anchor=tk.CENTER)
        header.pack(pady=15, fill=tk.X)

        # --- Results Summary Frame (Parent is content_frame) ---
        results_outer_frame = ttk.LabelFrame(content_frame, text="Detection Summary", padding="15", style="White.TLabelframe")
        results_outer_frame.pack(fill=tk.X, padx=20, pady=10)

        # Calculate average probability (using 'results' list)
        # 'results' holds individual probs for 'scanned', avg prob for 'summary'
        results_list = scan_data.get('results', [])
        valid_results = [r for r in results_list if isinstance(r, (float, int))] # Ensure numeric and not None

        avg_prob = 0.0
        num_valid = len(valid_results)
        if num_valid > 0:
            avg_prob = sum(valid_results) / num_valid

        threshold = self.controller.detector.optimal_threshold
        is_fake = avg_prob >= threshold
        status = "⚠ Potential Deepfake" if is_fake else "✅ Likely Real"
        status_style = "StatusFake.TLabel" if is_fake else "StatusReal.TLabel"

        ttk.Label(results_outer_frame, text=f"Overall Status: {status}",
                  style=status_style, background="#FFFFFF").pack(pady=5)
        avg_conf_percent = avg_prob * 100
        ttk.Label(results_outer_frame, text=f"Average Fake Probability: {avg_conf_percent:.2f}% (Threshold: {threshold*100:.2f}%)",
                  style="Detail.TLabel", background="#FFFFFF").pack(pady=5)
        if num_valid != len(results_list):
             failed_count = len(results_list) - num_valid
             ttk.Label(results_outer_frame, text=f"({failed_count} face(s) failed analysis)",
                       style="Detail.TLabel", background="#FFFFFF", foreground="grey").pack(pady=2)


        # --- File Details (Parent is content_frame) ---
        detection_type = scan_data.get("detection_type", "scanned") # Get type early
        if detection_type == 'scanned' and 'file_path' in scan_data and scan_data['file_path']:
            details_frame = ttk.LabelFrame(content_frame, text="File Details", padding="15", style="White.TLabelframe")
            details_frame.pack(fill=tk.X, padx=20, pady=10)
            try:
                file_details = self.controller.get_file_details(scan_data['file_path'])
                for key, value in file_details.items():
                    detail_label = ttk.Label(details_frame, text=f"{key.capitalize()}: {value}",
                                            style="Detail.TLabel", background="#FFFFFF")
                    detail_label.pack(anchor="w", padx=10, pady=2)
            except Exception as e:
                logger.error(f"Error getting file details in report: {e}", exc_info=True)
                ttk.Label(details_frame, text="Error getting file details.", foreground="red", background="#FFFFFF").pack(anchor="w", padx=10, pady=2)


        # --- Real-time Summary Details (Parent is content_frame) ---
        if detection_type == "real-time-summary":
            summary_frame = ttk.LabelFrame(content_frame, text="Real-Time Session Info", padding="15", style="White.TLabelframe")
            summary_frame.pack(fill=tk.X, padx=20, pady=10)
            summary = scan_data.get("summary", {})
            # Display specific summary fields cleanly
            ttk.Label(summary_frame, text=f"Overall Session Result: {summary.get('overall_result', 'N/A')}",
                      style="Detail.TLabel", background="#FFFFFF").pack(anchor="w", padx=10, pady=2)
            ttk.Label(summary_frame, text=f"Total Faces Analyzed: {summary.get('total_faces_processed', 'N/A')}",
                      style="Detail.TLabel", background="#FFFFFF").pack(anchor="w", padx=10, pady=2)
            ttk.Label(summary_frame, text=f"Real Detections: {summary.get('real_detections', 'N/A')}",
                      style="Detail.TLabel", background="#FFFFFF").pack(anchor="w", padx=10, pady=2)
            ttk.Label(summary_frame, text=f"Fake Detections: {summary.get('fake_detections', 'N/A')}",
                      style="Detail.TLabel", background="#FFFFFF").pack(anchor="w", padx=10, pady=2)
            # Note: Avg Prob already shown in the main summary section above

        # --- Detected Faces Frame (Parent is content_frame) ---
        # Now content_frame is defined before this section
        if detection_type == "real-time-summary":
            faces_frame_text = "Session Face Thumbnails"
        elif detection_type == "scanned":
            faces_frame_text = "Detected Faces Analysis"
        else: # Handle old 'real-time' logs or unknown
            faces_frame_text = f"Detected Faces ({detection_type})"

        faces_frame = ttk.LabelFrame(content_frame, text=faces_frame_text, style="White.TLabelframe", padding="10")
        faces_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        grid_frame = ttk.Frame(faces_frame, style="WhiteBackground.TFrame")
        grid_frame.pack(pady=5) # Center the grid frame

        face_thumbnails = scan_data.get('face_thumbnails', [])

        if not face_thumbnails:
            ttk.Label(grid_frame, text="No face thumbnails available.", style='Detail.TLabel', background="#FFFFFF").pack()
        else:
            max_cols = 4 # Faces per row
            for i, abs_path in enumerate(face_thumbnails):
                row_idx = i // max_cols
                col_idx = i % max_cols
                face_container = ttk.Frame(grid_frame, padding=5, style="WhiteBackground.TFrame")
                face_container.grid(row=row_idx, column=col_idx, padx=5, pady=5, sticky="n")

                photo = self._load_thumbnail(abs_path, size=(120, 120))

                if photo:
                    img_label = ttk.Label(face_container, image=photo, background="#FFFFFF")
                    img_label.image = photo
                    img_label.pack(pady=2)

                    # Show Probability ONLY for 'scanned' type where results list matches thumbnails
                    if detection_type == 'scanned' and i < len(results_list):
                        prob = results_list[i]
                        if prob is not None: # Check if prediction was successful for this face
                            face_status = "FAKE" if prob >= threshold else "REAL"
                            color = "red" if face_status == "FAKE" else "green"
                            ttk.Label(face_container, text=f"{face_status} ({prob * 100:.1f}%)",
                                      font=('Helvetica', 10), foreground=color, background="#FFFFFF").pack(pady=1)
                        else:
                            ttk.Label(face_container, text="Analysis Failed",
                                      font=('Helvetica', 10, 'italic'), background="#FFFFFF", foreground="grey").pack(pady=1)
                    # No individual probability shown for summary thumbnails
                    elif detection_type == 'real-time-summary':
                          # Optionally add a simple counter label
                          ttk.Label(face_container, text=f"Face #{i+1}", font=('Helvetica', 9, 'italic'), background="#FFFFFF", foreground="grey").pack(pady=1)


                else: # Thumbnail failed to load
                    ttk.Label(face_container, text="[Image N/A]", background="#FFFFFF", foreground="grey").pack(pady=5, ipadx=10, ipady=10)


        # Update scrollregion after adding ALL content
        self.scrollable_frame.update_idletasks() # Ensure frame size is calculated
        self._on_frame_configure() # Update scroll region
        self.canvas.yview_moveto(0) # Scroll back to top