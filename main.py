import sys
import ctypes
# --- AppUserModelID Setting ---
_appid_set_successfully = False
_appid_error_message = ""
_myappid_for_log = ""
if sys.platform == 'win32':
    try:
        myappid_val = u'ChuaKaiZenUTHM.FakeSeeker.1.0'
        # Ensure shell32 is available
        if hasattr(ctypes, 'windll') and hasattr(ctypes.windll, 'shell32'):
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid_val)
            _appid_set_successfully = True
            _myappid_for_log = myappid_val
        else:
            _appid_error_message = "ctypes.windll.shell32 not available."
    except Exception as e_appid:
        _appid_error_message = str(e_appid)

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import os
from datetime import datetime
import numpy as np
import json
import shutil
from mss import mss
import pygetwindow as gw
import threading
import logging
import time
import queue
import appdirs
import hashlib
import requests

# --- Stdout/Stderr Redirection ---
_stdout_redirected = False
_stderr_redirected = False
if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'): 
    if sys.stdout is None or not hasattr(sys.stdout, 'fileno'):
        sys.stdout = open(os.devnull, 'w')
        _stdout_redirected = True
    if sys.stderr is None or not hasattr(sys.stderr, 'fileno'):
        sys.stderr = open(os.devnull, 'w')
        _stderr_redirected = True

# logic
from deepfake_detector import DeepfakeDetector
from face_extractor import FaceExtractor

# UI pages
from ui.home_page import HomePage
from ui.sidebar import Sidebar
from ui.upload_page import UploadPage
from ui.realtime_page import RealtimePage
from ui.history_page import HistoryPage
from ui.report_page import ReportPage
from ui.toolbar import FloatingToolbar

# --- Setup Logging ---
from logger_setup import setup_logging, get_logger
setup_logging()
logger = get_logger(__name__)

# --- Log AppUserModelID and Redirection Status---
if sys.platform == 'win32':
    if _appid_set_successfully:
        logger.info(f"AppUserModelID successfully set to: {_myappid_for_log}")
    else:
        logger.warning(f"Could not set AppUserModelID: {_appid_error_message}")
if _stdout_redirected:
    logger.info("sys.stdout redirected to os.devnull for bundled app.")
if _stderr_redirected:
    logger.info("sys.stderr redirected to os.devnull for bundled app.")

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
        logger.debug(f"Accessing bundled resource from _MEIPASS: {base_path}")
    except Exception:
        # If not bundled, use the script's directory
        base_path = os.path.abspath(os.path.dirname(__file__))
        logger.debug(f"Accessing resource from script path: {base_path}")

    path = os.path.join(base_path, relative_path)
    logger.debug(f"Resource path requested for '{relative_path}', resolved to: {path}") # Optional: Extra logging
    return path

history_lock = threading.Lock() # Lock for history file access

class FakeSeekerApp:
    def __init__(self, root):
        logger.info("FakeSeekerApp initializing...")
        self.app_name = "FakeSeeker"
        self.app_author = "ChuaKaiZen_UTHM"
        self.root = root

        # --- SETTING THE WINDOW ICON ---
        try:
            icon_path_for_window = resource_path(os.path.join('images', 'fakeseeker.ico'))
            if os.path.exists(icon_path_for_window):
                self.root.iconbitmap(icon_path_for_window) # SET ICON HERE
                logger.info(f"Window icon set from: {icon_path_for_window}")
            else:
                logger.warning(f"Window icon 'fakeseeker.ico' not found: {icon_path_for_window}")
        except Exception as e:
            logger.error(f"Failed to set window icon: {e}", exc_info=True)

        self.root.title("FakeSeeker - Deepfake Detection Tool")
        
        self.root.geometry("1600x900")
        self.root.state("zoomed")
        self.root.resizable(True, True)  # Allow resizing if needed
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing) # Handle main window close
        
        # --- Core Logic Attributes ---
        self.selected_file = None
        self.scan_history = [] # Holds loaded history (dictionaries)

        # Real-time detection variables
        self.camera_on = False
        self.detection_active = False
        self.screen_monitoring = False
        self.cap = None
        self.camera_thread = None
        self.screen_update_job = None

        # Real-time results storage (For Throttled Saving & Summary)
        self.rt_frame_count = 0 # Total faces processed in detection mode
        self.rt_real_count = 0 # Count based on threshold
        self.rt_fake_count = 0 # Count based on threshold
        self.rt_results_list = [] # List to store DICTS: {'prob': float, 'thumb_path': str}

        self.rt_store_interval = 1.0 # Store result approx every N seconds
        self.rt_last_store_time = 0.0 # Timestamp of the last stored result

        # UI State
        self.floating_toolbar = None
        self.current_page_instance = None
        self.report_came_from_scan = False
        self.current_report_data = None # Holds data for the current report being viewed

        # Threading Queue
        self.scan_queue = queue.Queue()
        self.scan_status_label = None # Reference to status label in UploadPage

        # --- Determine User Data Directory ---
        try:
             user_data_dir = appdirs.user_data_dir(self.app_name, self.app_author)
             logger.info(f"Using user data directory: {user_data_dir}")
        except Exception as e:
             logger.error(f"Could not determine user data directory via appdirs: {e}. Falling back.")
             # Fallback to a directory next to the script/executable (less ideal)
             script_dir = os.path.abspath(os.path.dirname(__file__))
             user_data_dir = os.path.join(script_dir, "user_data")
             logger.info(f"Using fallback user data directory: {user_data_dir}")

        # --- Define User-Specific Paths ---
        self.user_data_dir = user_data_dir # Store for potential use elsewhere
        self.reports_dir = os.path.join(user_data_dir, 'reports')
        self.thumbnails_dir = os.path.join(self.reports_dir, 'thumbnails')
        self.history_file = os.path.join(self.reports_dir, 'scan_history.json')
        self.user_model_dir = os.path.join(user_data_dir, 'models')
        self.user_threshold_path = os.path.join(user_data_dir, 'optimal_threshold.json')
        self.local_version_file = os.path.join(user_data_dir, 'version.txt') # For update check

        # --- Bundled Asset Paths (Using resource_path) ---
        self.images_dir = resource_path('images') # Get path to bundled images

        # --- Setup Directories (This creates the user data dirs) ---
        self._setup_directories()

        try:
            self.face_extractor = FaceExtractor(min_confidence=0.9, min_face_size=50)
            # Pass user paths so detector knows where to look first
            self.detector = DeepfakeDetector(
                user_model_dir=self.user_model_dir,
                user_threshold_path=self.user_threshold_path
            )
            logger.info("FaceExtractor and DeepfakeDetector initialized successfully.")
        except Exception as e:
            logger.critical(f"Failed to initialize core components: {e}", exc_info=True)
            messagebox.showerror("Critical Error", f"Failed to initialize components: {str(e)}\nCheck logs.")
            self.root.destroy()
            return

        # --- Load Shared Resources (Icons) ---
        self._load_icons()

        # --- Load History ---
        self.load_scan_history()

        # --- Setup Styles ---
        self.setup_styles()

        self._setup_layout()

        self.show_home_page()

        # Call update check at the end
        self.check_for_updates(background=True)

    def _setup_directories(self):
        """Creates necessary directories."""
        logger.debug("Setting up directories...")
        dirs_to_create = [self.user_data_dir, self.reports_dir, self.thumbnails_dir, self.user_model_dir]
        for directory in dirs_to_create:
            try:
                os.makedirs(directory, exist_ok=True)
                # Simple write test
                test_file = os.path.join(directory, '.test_write')
                with open(test_file, 'w') as f: f.write('test')
                os.remove(test_file)
            except Exception as e:
                logger.critical(f"Failed to create/access user data dir: {directory}: {e}", exc_info=True)
                messagebox.showerror("Startup Error", f"Failed to access directory:\n{directory}\n\nError: {e}\n\nPlease check permissions.")
                # Should probably exit cleanly
                if hasattr(self, 'root'): self.root.destroy()
                raise RuntimeError("Directory setup failed")
        logger.debug("User data directories checked/created.")

    def _load_icons(self):
        """Loads icons needed by the UI."""
        logger.debug("Loading icons...")
        icon_size_large = (80,80)
        self.upload_icon = self._load_single_icon('upload.png', icon_size_large)
        self.realtime_icon = self._load_single_icon('realtime.png', icon_size_large)
        self.results_icon = self._load_single_icon('results.png', icon_size_large)
        self.screen_icon = self._load_single_icon('home.png', icon_size_large)

    def _load_single_icon(self, filename, size):
        """Helper to load and resize a single icon."""
        try:
            icon_path = resource_path(os.path.join('images', filename))
            if os.path.exists(icon_path):
                icon = Image.open(icon_path).convert("RGBA")
                icon = icon.resize(size, Image.Resampling.LANCZOS)
                return ImageTk.PhotoImage(icon)
            else:
                logger.warning(f"Icon file not found: {icon_path}")
        except Exception as e:
            logger.error(f"Error loading icon {filename}: {e}", exc_info=True)
        # Fallback
        fallback_img = Image.new('RGB', size, color='lightgray')
        return ImageTk.PhotoImage(fallback_img)

    def setup_styles(self):
        """Configure the styles for a clean, modern UI with readable fonts."""
        logger.debug("Setting up ttk styles...")
        # --- Create the Style object FIRST ---
        self.style = ttk.Style() # Store style object on self for potential later use
        style = self.style # Use a local variable 'style' for convenience below
        try:
            style.theme_use('clam')
        except tk.TclError:
            logger.warning("Clam theme not available, using default.")

        # Configure Styles
        style.configure("Home.TButton", font=('Helvetica', 15), padding=10)
        style.configure("TLabel", font=('Helvetica', 14)) # Slightly larger default
        style.configure("Header.TLabel", font=('Helvetica', 30, 'bold')) # Significantly larger Header
        style.configure("SectionTitle.TLabel", font=('Helvetica', 18, 'bold')) # Larger Section Title
        style.configure("TLabelframe.Label", font=('Helvetica', 16, 'bold')) # Larger Labelframe Title
        style.configure("Detail.TLabel", font=('Helvetica', 14)) # Slightly larger Detail
        style.configure("StatusReal.TLabel", font=('Helvetica', 16, 'bold'), foreground="#2E8B57") # Larger Status
        style.configure("StatusFake.TLabel", font=('Helvetica', 16, 'bold'), foreground="#DC143C") # Larger Status
        style.configure("TButton", font=('Helvetica', 16), padding=12)
        style.configure("Toolbar.TButton", font=('Helvetica', 10), padding=5) # Smaller font and padding
        style.configure("Card.TFrame", relief="solid", borderwidth=1)
        style.configure("WhiteBackground.TFrame", background="#FFFFFF")
        style.configure("White.TLabelframe", background="#FFFFFF")
        style.configure("White.TLabelframe.Label", font=('Helvetica', 14, 'bold'), background="#FFFFFF")
        # for report page
        report_bg_color = "#F0F0F0"
        style.configure("ReportPage.TFrame", background=report_bg_color)
        style.configure("ReportContent.TFrame", background=report_bg_color) # For the content_frame
        style.configure("ReportSection.TLabelframe", background=report_bg_color)
        style.configure("ReportSection.TLabelframe.Label", font=('Helvetica', 14, 'bold'), background=report_bg_color)
        style.configure("Detail.TLabel", font=('Helvetica', 14)) # No background specified, should inherit
        style.configure("Header.TLabel", font=('Helvetica', 30, 'bold')) # No background, should inherit or be set explicitly
        style.configure("Instructions.TLabelframe", padding=10)
        style.configure("Instructions.TLabelframe.Label", font=('Helvetica', 16, 'bold')) # For the "How to Use FakeSeeker" title
        style.configure("Instruction.TLabel", font=('Helvetica', 13), padding=2) # For the actual instruction lines

        logger.info("Styles configured successfully.")

    def _setup_layout(self):
        """Sets up the main application layout."""
        logger.debug("Setting up main layout...")

        # --- Footer Frame (Packed FIRST at the absolute bottom of ROOT) ---
        footer_frame = ttk.Frame(self.root)
        footer_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5, padx=10)
        ttk.Label(footer_frame, text="Version Beta - Enhanced with EfficientNet-b2", style='TLabel').pack(side=tk.LEFT, padx=10)
        ttk.Button(footer_frame, text="Exit", command=self.on_closing).pack(side=tk.RIGHT, padx=10)

        # --- Top Area Frame (Holds Sidebar and Main Area) ---
        top_area = ttk.Frame(self.root)
        top_area.pack(side=tk.TOP, fill=tk.BOTH, expand=True) # Fills space above footer

        # --- Sidebar (Created, packed in show_frame) ---
        self.sidebar = Sidebar(top_area, self)

        # --- Main Area (Parent is top_area) ---
        self.main_area = ttk.Frame(top_area)
        self.main_area.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5), pady=5)

        # --- Elements INSIDE main_area ---

        # 1. Action Buttons Frame (Pack this at the BOTTOM of main_area FIRST)
        self.action_button_frame = ttk.Frame(self.main_area)
        self.action_button_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(5, 0)) # Reserves space at bottom

        # 2. Container for page frames (Pack this ABOVE action buttons, filling remaining space)
        self.container = ttk.Frame(self.main_area)
        self.container.pack(side=tk.TOP, fill=tk.BOTH, expand=True) # Expands to fill space above action buttons
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)
        # --- End Elements INSIDE main_area ---

        # Instantiate pages and place them in the container (grid)
        self.pages = {}
        for PageClass in (HomePage, UploadPage, RealtimePage, HistoryPage, ReportPage):
            page_name = PageClass.__name__
            frame = PageClass(self.container, self) # Parent is container
            self.pages[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew") # Pages fill the container

        # Get reference to UploadPage's status label
        upload_page_instance = self.pages.get("UploadPage")
        if upload_page_instance and hasattr(upload_page_instance, 'status_label'):
            self.scan_status_label = upload_page_instance.status_label
            logger.debug("Scan status label assigned from UploadPage.")
        else:
            logger.error("UploadPage instance or its status_label not found.")
            self.scan_status_label = None
        logger.debug("Main layout setup complete.")

    def _configure_action_buttons(self, page_name):
        """Configures buttons in the action_button_frame (bottom of main_area)."""
        # Clear previous buttons first
        for widget in self.action_button_frame.winfo_children():
            widget.destroy()

        # Define pages that should have a simple "Back to Home" button
        back_to_home_pages = ["UploadPage", "RealtimePage", "HistoryPage"]

        if page_name in back_to_home_pages:
            back_btn = ttk.Button(self.action_button_frame, text="Back to Home",
                                  command=self.show_home_page)
            back_btn.pack(pady=5) # Simple pack, centers it by default

        # --- Specific Buttons for Report Page ---
        elif page_name == "ReportPage":
            # Back button (goes to History or Upload)
            back_btn = ttk.Button(self.action_button_frame, text="Back",
                                  command=self._go_back_from_report)
            back_btn.pack(side=tk.LEFT, padx=10, pady=5)

        # --- No action buttons needed for HomePage ---
        elif page_name == "HomePage":
            pass # Explicitly do nothing

    def show_frame(self, page_name):
        """Raises the specified page frame and manages sidebar/action buttons."""
        logger.info(f"Switching to page: {page_name}")

        # --- Sidebar Management ---
        pages_with_sidebar = ["UploadPage", "RealtimePage", "HistoryPage", "ReportPage"]
        if page_name in pages_with_sidebar:
            if not self.sidebar.winfo_ismapped():
                logger.debug("Packing sidebar.")
                # Pack sidebar INSIDE top_area, BEFORE main_area
                self.sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5, before=self.main_area)
        else: # Hide for HomePage
            if self.sidebar.winfo_ismapped():
                logger.debug("Hiding sidebar.")
                self.sidebar.pack_forget()

        # --- Raise Page and Configure Buttons ---
        if page_name in self.pages:
            if self.current_page_instance and hasattr(self.current_page_instance, 'on_hide'):
                self.current_page_instance.on_hide()

            frame = self.pages[page_name]
            self.current_page_instance = frame
            frame.tkraise()

            if hasattr(frame, 'on_show'): frame.on_show()

            self._configure_action_buttons(page_name) # Configure buttons AFTER raising frame
        else:
            logger.error(f"Page '{page_name}' not found.")

    def show_home_page(self):
        self.show_frame("HomePage")

    def show_upload_page(self):
        self.selected_file = None
        self.show_frame("UploadPage")
        upload_page = self.pages.get("UploadPage")
        if upload_page:
            upload_page.update_preview(None)
            upload_page.set_scan_button_state(tk.DISABLED)
            self._update_status_label("") # Clear status on page load

    def show_realtime_page(self):
        self.show_frame("RealtimePage")
        rt_page = self.pages.get("RealtimePage")
        if rt_page: rt_page.update_button_states() # Ensure buttons reflect state

    def show_history_page(self):
        self.show_frame("HistoryPage")
        history_page = self.pages.get("HistoryPage")
        if history_page:
            current_history = self.load_scan_history()
            # Sort history data before displaying
            sorted_history = sorted(current_history, key=lambda x: x.get("timestamp", ""), reverse=True)
            history_page.display_history(sorted_history)

    def show_detailed_report(self, scan_data, from_scan=False):
        self.report_came_from_scan = from_scan
        self.current_report_data = scan_data
        self.show_frame("ReportPage")
        report_page = self.pages.get("ReportPage")
        if report_page: report_page.display_report(scan_data)

    def _go_back_from_report(self):
        """Handles the 'Back' button logic when on the report page."""
        logger.debug(f"Go back from report. Came from scan: {self.report_came_from_scan}")
        current_data = self.current_report_data # Keep data reference

        # Determine where to navigate back to
        next_page_func = self.show_upload_page if self.report_came_from_scan else self.show_history_page

        # --- Ask to save ONLY if came from a scan ---
        if self.report_came_from_scan and current_data:
            # Pass the actual next navigation function to the confirmation method
            self.confirm_save_before_leaving(current_data, next_page_func)
        else:
            # If not from scan (e.g., from history), just navigate back
            self.current_report_data = None # Clear data reference
            self.report_came_from_scan = False # Reset flag
            next_page_func() # Go back directly

    def load_scan_history(self):
        """Load scan history from JSON file."""
        logger.debug("Loading scan history...")
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r') as f:
                    # Add basic validation
                    loaded_data = json.load(f)
                    if isinstance(loaded_data, list):
                        self.scan_history = loaded_data
                        logger.info(f"Loaded {len(self.scan_history)} items from history.")
                    else:
                         logger.warning(f"Scan history file format error: Expected list, got {type(loaded_data)}. Resetting.")
                         self.scan_history = []
            else:
                logger.info("Scan history file not found. Starting fresh.")
                self.scan_history = []
        except json.JSONDecodeError:
            logger.error(f"Failed to decode scan history JSON from {self.history_file}. Resetting.", exc_info=True)
            self.scan_history = []
        except Exception as e:
            logger.error(f"Error loading scan history: {e}", exc_info=True)
            messagebox.showerror("History Load Error", f"Failed to load scan history: {str(e)}")
            self.scan_history = []
        return self.scan_history # Return the loaded history

    def save_scan_history(self):
        """Save scan history to JSON file."""
        logger.debug("Attempting to save scan history...")
        try:
            if not isinstance(self.scan_history, list):
                 logger.error(f"Scan history is not a list (type: {type(self.scan_history)}). Cannot save.")
                 return

            # Log number of items instead of full content
            num_items = len(self.scan_history)
            logger.debug(f"Attempting to save {num_items} history items.")

            os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
            
            def safe_float(value):
                if isinstance(value, tuple):
                    logger.warning(f"Converting tuple to float: {value}")
                    return float(value[0])
                try: return float(value)
                except (ValueError, TypeError):
                    logger.warning(f"Cannot convert {value} (type {type(value)}) to float. Using 0.0.")
                    return 0.0

            serializable_history = []
            for i, scan in enumerate(self.scan_history):
                if not isinstance(scan, dict):
                     logger.warning(f"Skipping non-dictionary item at index {i} in scan_history: {scan}")
                     continue
                try:
                    # Define the base structure expected for any entry
                    serializable_scan = {
                        'timestamp': scan.get('timestamp', f'UNKNOWN_{i}'),
                        'detection_type': scan.get('detection_type', 'unknown'),
                        'results': [safe_float(r) for r in scan.get('results', [])], # Common results format
                        'face_thumbnails': scan.get('face_thumbnails', []) # Common format
                    }
                    # Add fields specific to certain types if they exist
                    if 'file_path' in scan: serializable_scan['file_path'] = scan.get('file_path')
                    if 'summary' in scan: serializable_scan['summary'] = scan.get('summary')
                    if 'sample_image' in scan: serializable_scan['sample_image'] = scan.get('sample_image')
                    if 'detection_result' in scan: serializable_scan['detection_result'] = scan.get('detection_result')
                    if 'probability_fake' in scan: serializable_scan['probability_fake'] = scan.get('probability_fake')

                    serializable_history.append(serializable_scan)
                except Exception as item_err:
                    logger.error(f"Error processing history item with timestamp {scan.get('timestamp', 'UNKNOWN')}: {item_err}", exc_info=True)

            with open(self.history_file, 'w') as f:
                json.dump(serializable_history, f, indent=4)
            logger.info(f"Scan history saved successfully to {self.history_file}")

        except Exception as e: # Outer exception handler remains correct
            logger.error(f"Fatal error occurred during saving scan history: {e}", exc_info=True)
            if self.root.winfo_exists():
                 messagebox.showerror("History Save Error", f"Failed to save scan history:\n{e}")

    def on_closing(self):
        """Save scan history before closing the application."""
        logger.info("Application closing...")
        # Stop any running detection/camera/screen feeds
        if self.detection_active or self.camera_on or self.screen_monitoring:
             self.stop_detection() # stop_detection handles all these flags now

        # Clean up extractor temp dir
        if hasattr(self, 'face_extractor') and hasattr(self.face_extractor, 'cleanup'):
             logger.debug("Cleaning up face extractor temp.")
             self.face_extractor.cleanup()

        # Save history
        self.save_scan_history()

        logger.info("FakeSeeker application closed.")
        # Ensure root is destroyed
        try:
            if self.root.winfo_exists():
                self.root.destroy()
        except tk.TclError:
            pass # Ignore if already destroying

    def upload_file(self):
        """Handles file selection dialog."""
        logger.debug("Opening file dialog...")
        try:
            file_path = filedialog.askopenfilename(
                title="Select Image or Video",
                filetypes=[
                    ("All Supported Files", "*.jpg *.jpeg *.png *.mp4 *.avi *.mov"),
                    ("Image Files", "*.jpg *.jpeg *.png"),
                    ("Video Files", "*.mp4 *.avi *.mov"),
                    ("All Files", "*.*")
                ]
            )
            if file_path:
                self.selected_file = file_path
                logger.info(f"File selected: {file_path}")
                upload_page = self.pages.get("UploadPage")
                if upload_page:
                    upload_page.update_preview(file_path)
                    upload_page.set_scan_button_state(tk.NORMAL)
            else:
                logger.debug("File selection cancelled.")
                self.selected_file = None
                upload_page = self.pages.get("UploadPage")
                if upload_page:
                    upload_page.update_preview(None)
                    upload_page.set_scan_button_state(tk.DISABLED)
        except Exception as e:
            logger.error(f"Error during file dialog: {e}", exc_info=True)
            messagebox.showerror("Error", f"Could not open file dialog:\n{e}")

    def start_scan(self):
        """Initiates the scan process in a background thread."""
        if not self.selected_file:
            messagebox.showwarning("No File", "Please select a file first.")
            return

        logger.info(f"Starting scan for: {self.selected_file}")
        upload_page = self.pages.get("UploadPage")
        if upload_page:
                upload_page.set_scan_button_state(tk.DISABLED)

        self._update_status_label("Starting scan...") # Update status via helper

        scan_thread = threading.Thread(target=self._perform_scan_async,
                                        args=(self.selected_file,),
                                        daemon=True)
        scan_thread.start()
        self.root.after(100, self.process_scan_queue) # Start queue checker

    def _perform_scan_async(self, file_path):
        """Performs the actual scan logic in a background thread."""
        logger.info(f"Scan thread started for: {file_path}")
        scan_data = None
        try:
            self.scan_queue.put({'status': 'Extracting faces...'})
            is_video = file_path.lower().endswith(('.mp4', '.avi', '.mov'))
            faces = self.face_extractor.extract_faces_from_video(file_path) if is_video else self.face_extractor.extract_faces_from_image(file_path)

            if not faces:
                raise ValueError("No faces detected in the selected file.")

            self.scan_queue.put({'status': f'Found {len(faces)} faces. Saving thumbnails...'})
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # Use consistent timestamp
            scan_thumbnails_dir = os.path.join(self.thumbnails_dir, f"scan_{timestamp}")
            os.makedirs(scan_thumbnails_dir, exist_ok=True)
            face_thumbnails_rel_paths = []
            valid_faces_for_analysis = [] # Store faces corresponding to saved thumbs

            for i, face in enumerate(faces):
                thumb_path = os.path.join(scan_thumbnails_dir, f'face_{i}.png')
                try:
                    face.save(thumb_path, "PNG")
                    face_thumbnails_rel_paths.append(thumb_path)
                    valid_faces_for_analysis.append(face) # Add face if thumb saved
                except Exception as save_err:
                    logger.error(f"Worker: Failed to save thumbnail {i}: {save_err}")
                    # Skip this face if thumbnail saving fails

            if not valid_faces_for_analysis:
                raise ValueError("Failed to save any face thumbnails.")

            self.scan_queue.put({'status': 'Analyzing faces...'})
            results_probabilities = []
            num_faces_to_analyze = len(valid_faces_for_analysis)

            for i, face_pil in enumerate(valid_faces_for_analysis):
                self.scan_queue.put({'status': f'Analyzing face {i+1}/{num_faces_to_analyze}...'})
                prediction_result = self.detector.predict_pil(face_pil) # Predict the valid face
                if prediction_result:
                    results_probabilities.append(prediction_result[1]) # Store probability
                else:
                    logger.warning(f"Worker: Prediction failed for saved face {i}")
                    results_probabilities.append(None) # Mark as failed

            # Check if *any* prediction succeeded
            valid_results = [p for p in results_probabilities if p is not None]
            if not valid_results:
                raise ValueError("Face analysis failed for all saved faces.")

            scan_data = {
                'timestamp': timestamp,
                'file_path': file_path,
                'results': results_probabilities, # May contain None for failed predictions
                'face_thumbnails': face_thumbnails_rel_paths, # Paths corresponding to results
                'detection_type': 'scanned'
            }
            logger.info("Worker: Scan complete.")
            self.scan_queue.put({'result': scan_data})

        except Exception as e:
            logger.error(f"Error during async scan for {file_path}: {e}", exc_info=True)
            self.scan_queue.put({'error': f"Scan failed: {str(e)}"})
        finally:
            logger.debug(f"Worker: Scan thread finishing for {file_path}")
            self.scan_queue.put({'status': 'done'})

    def process_scan(self, scan_data):
        """Process completed scan data by showing the report."""
        logger.debug(f"Processing scan result for: {scan_data.get('file_path')}")
        self.show_detailed_report(scan_data, from_scan=True)

    def process_scan_queue(self):
        """Checks the scan queue and updates UI."""
        reschedule_check = True # Assume we need to check again
        try:
            message = self.scan_queue.get_nowait() # Non-blocking get

            if 'status' in message:
                status = message['status']
                self._update_status_label(status)
                if status == 'done':
                    upload_page = self.pages.get("UploadPage")
                    if upload_page and upload_page.winfo_exists():
                         upload_page.set_scan_button_state(tk.NORMAL)
                    # Optionally clear status after a short delay or leave "Scan finished."
                    # self.root.after(3000, lambda: self._update_status_label(""))
                    reschedule_check = False # Stop checking queue
            elif 'result' in message:
                self.process_scan(message['result'])
                # No status update here, process_scan navigates away
                # Still need to signal done if this is the last message
            elif 'error' in message:
                error_msg = message['error']
                logger.error(f"Scan error from worker: {error_msg}")
                messagebox.showerror("Scan Error", error_msg)
                self._update_status_label("Scan failed.")
                upload_page = self.pages.get("UploadPage")
                if upload_page and upload_page.winfo_exists():
                     upload_page.set_scan_button_state(tk.NORMAL)
                reschedule_check = False # Stop checking queue

        except queue.Empty:
            pass # No message yet, keep checking
        except Exception as e:
             logger.error(f"Error processing scan queue: {e}", exc_info=True)
             reschedule_check = False # Stop checking on unexpected error

        # Reschedule check if needed
        if reschedule_check and hasattr(self.root, 'after'): # Check root exists
            try:
                 self.root.after(100, self.process_scan_queue)
            except tk.TclError:
                 logger.warning("Failed to reschedule queue check (window closed?).")

    def _update_status_label(self, text):
        """Safely updates the scan status label."""
        if self.scan_status_label and self.scan_status_label.winfo_exists():
            try:
                self.scan_status_label.config(text=text)
                logger.info(f"Scan Status Updated: {text}")
            except tk.TclError:
                logger.warning("Failed to update status label (window likely closing).")
        else:
            logger.warning(f"Scan status label not available. Status: {text}")
    
    def toggle_camera(self):
        """Toggle the camera on and off safely."""
        if self.screen_monitoring:
             messagebox.showwarning("Mode Conflict", "Please stop screen monitoring before turning on the camera.")
             return

        rt_page = self.pages.get("RealtimePage") # Get page reference

        if hasattr(self, 'camera_on') and self.camera_on:
             # --- Turning Camera Off ---
             logger.info("Turning camera OFF.")
             self.camera_on = False # Signal thread to stop
             if hasattr(self, 'camera_thread') and self.camera_thread.is_alive():
                  logger.debug("Waiting for camera thread to join...")
                  self.camera_thread.join(timeout=1.0) # Wait briefly
                  if self.camera_thread.is_alive(): logger.warning("Camera thread join timed out.")

             if hasattr(self, 'cap') and self.cap:
                  if self.cap.isOpened(): self.cap.release()
                  self.cap = None
                  logger.info("Camera released.")

             # Reset UI on the realtime page
             if rt_page: rt_page.update_button_states() # Update buttons via page method
             if hasattr(self, 'video_label') and self.video_label.winfo_exists():
                  self.video_label.configure(image='', text="Camera Off") # Reset label

        else:
            # --- Turning Camera On ---
            logger.info("Turning camera ON.")
            try:
                backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
                cap = None
                # ... (camera opening logic - keep from main_1.py) ...
                for index in [0, 1]:
                     for backend in backends:
                          cap = cv2.VideoCapture(index, backend)
                          if cap.isOpened(): break
                     if cap and cap.isOpened(): break
                     if cap: cap.release()

                if cap is None or not cap.isOpened():
                    raise IOError("Could not access any camera.")

                self.cap = cap
                self.camera_on = True

                # Reset real-time session state
                self.rt_frame_count = 0
                self.rt_real_count = 0
                self.rt_fake_count = 0
                self.rt_results_list = []
                self.rt_last_store_time = time.time()

                # Start feed thread
                self.camera_thread = threading.Thread(target=self.update_camera_feed, daemon=True)
                self.camera_thread.start()
                logger.info("Camera feed thread started.")

                # Update UI on the realtime page
                if rt_page: rt_page.update_button_states()
                if hasattr(self, 'video_label') and self.video_label.winfo_exists():
                     self.video_label.configure(text="") # Clear label text

            except Exception as e:
                logger.error(f"Failed to open camera: {e}", exc_info=True)
                messagebox.showerror("Camera Error", f"Could not access the camera:\n{e}")
                if hasattr(self, 'cap') and self.cap: self.cap.release()
                self.cap = None
                self.camera_on = False
                if rt_page: rt_page.update_button_states() # Ensure buttons reset

    def stop_camera(self):
        """Stops camera resources and related state (internal use)."""
        logger.debug("Internal stop_camera called.")
        self.camera_on = False # Signal thread flag first
        if hasattr(self, 'camera_thread') and self.camera_thread and self.camera_thread.is_alive():
            if threading.current_thread() != self.camera_thread:
                logger.debug("Joining camera thread...")
                self.camera_thread.join(timeout=0.5) # Short timeout
            else: logger.warning("stop_camera called from camera thread itself.")
        self.camera_thread = None

        cap_instance = getattr(self, 'cap', None)
        if cap_instance:
            logger.debug("Releasing camera capture...")
            if cap_instance.isOpened(): cap_instance.release()
            self.cap = None
            logger.debug("Camera capture released and set to None.")

        # Clear video label (might be redundant if toggle_camera also does it)
        # if hasattr(self, 'video_label') and self.video_label.winfo_exists():
        #      self.video_label.configure(image='', text="Camera Off")

    def toggle_screen_monitoring(self):
        """Toggle screen monitoring mode."""
        if self.camera_on:
            messagebox.showwarning("Mode Conflict", "Please turn off the camera first.")
            return

        rt_page = self.pages.get("RealtimePage")

        if not self.screen_monitoring:
            # --- Start Screen Monitoring ---
            logger.info("Starting screen monitoring.")
            self.screen_monitoring = True
            self.detection_active = False # Reset detection state

            # Reset session state
            self.rt_frame_count = 0
            self.rt_real_count = 0
            self.rt_fake_count = 0
            self.rt_results_list = []
            self.rt_last_store_time = time.time()

            # Update main page buttons BEFORE hiding window
            self._update_realtime_button_states()

            try:
                if self.root.winfo_exists(): self.root.withdraw()
                self.create_floating_toolbar() # This now handles potential previous toolbar cleanup

                # Check if toolbar creation failed within create_floating_toolbar
                if not hasattr(self, 'floating_toolbar') or self.floating_toolbar is None:
                     raise RuntimeError("Floating toolbar creation failed.") # Force jump to except block

                self.screen_update_job = self.root.after(100, self.update_screen_feed)
                logger.info("Screen feed loop scheduled.")
            except Exception as e:
                logger.error(f"Error starting screen monitor UI: {e}", exc_info=True)
                self.screen_monitoring = False # Revert state
                # Try to restore UI
                if hasattr(self, 'root') and self.root.winfo_exists() and self.root.state() == 'withdrawn': 
                    try:
                        self.root.deiconify()
                    except tk.TclError:
                        pass
                # Ensure toolbar attribute is None if creation failed
                self.floating_toolbar = None
                self.toolbar_video_label = None
                # Reset buttons on main page
                self._update_realtime_button_states()

        else:
            # --- Stop Screen Monitoring (usually triggered by stop_detection or handle_toolbar_close) ---
            logger.info("Stopping screen monitoring via toggle button.")
            self.stop_detection() # Use the main stop function

    def create_floating_toolbar(self):
        """Creates or focuses the floating toolbar."""
        # Check if toolbar exists AND is not None AND window exists
        logger.debug("Attempting to create floating toolbar...")
        if hasattr(self, 'floating_toolbar') and isinstance(self.floating_toolbar, tk.Toplevel):
            logger.debug("Previous floating_toolbar attribute exists.")
            try:
                if self.floating_toolbar.winfo_exists():
                    logger.info("Destroying existing floating toolbar window.")
                    self.floating_toolbar.destroy()
                else:
                    logger.debug("Existing floating_toolbar window already destroyed.")
            except tk.TclError as e:
                logger.warning(f"TclError destroying existing toolbar (already gone?): {e}")
            except Exception as e:
                logger.error(f"Error destroying existing toolbar: {e}", exc_info=True)

        # --- Ensure attribute is None before creating new ---
        self.floating_toolbar = None
        self.toolbar_video_label = None

        # Create the new toolbar instance
        try:
            self.floating_toolbar = FloatingToolbar(self.root, self) # Creates the Toplevel window
            # Store reference to its video label if needed elsewhere (only if toolbar was created)
            if hasattr(self.floating_toolbar, 'video_label'):
                self.toolbar_video_label = self.floating_toolbar.video_label
            else:
                logger.error("Newly created FloatingToolbar instance does not have 'video_label' attribute!")
                self.toolbar_video_label = None

            logger.info("Floating toolbar created successfully.")
            self._update_toolbar_button_states() # Update state of buttons on the new toolbar
        except Exception as e:
            logger.error(f"Failed to create FloatingToolbar window: {e}", exc_info=True)
            messagebox.showerror("Toolbar Error", f"Could not create the monitor window:\n{e}")
            # Ensure cleanup if creation fails
            self.floating_toolbar = None
            self.toolbar_video_label = None
            raise # Re-raise the exception so the toggle logic can catch it

    def handle_toolbar_close(self):
        """Callback when toolbar 'X' or 'Show Main' is clicked."""
        logger.info("Toolbar handle_close called by toolbar.")
        # This should stop everything and show the main window
        self.stop_detection() # This already handles cleanup and showing main window

    def start_detection(self):
        """Activates deepfake detection on the active feed (camera or screen)."""
        if not self.camera_on and not self.screen_monitoring:
            messagebox.showwarning("Cannot Start", "Turn on Camera or start Screen Monitoring first.")
            return
        if self.detection_active:
            logger.debug("Detection already active.")
            return

        logger.info("Starting deepfake detection...")
        self.detection_active = True

        # Reset session results
        self.rt_frame_count = 0
        self.rt_real_count = 0
        self.rt_fake_count = 0
        self.rt_results_list = []
        self.rt_last_store_time = time.time()

        self._update_realtime_button_states() # Update main page buttons
        self._update_toolbar_button_states() # Update toolbar buttons

    def stop_detection(self):
        """Stops detection, associated feed (if needed), saves summary, resets UI."""
        logger.info(f"stop_detection called. DA:{self.detection_active}, SM:{self.screen_monitoring}, CO:{self.camera_on}")

        # Store current states before changing them
        was_detection_active = self.detection_active
        was_screen_monitoring = self.screen_monitoring
        was_camera_on = self.camera_on

        # --- Set flags first ---
        self.detection_active = False
        if was_screen_monitoring:
            self.screen_monitoring = False

        # --- Stop Feed Source ---
        if was_camera_on:
            logger.info("Stopping camera as part of stop_detection.")
            self.stop_camera() # Stops thread, releases cap

        if was_screen_monitoring: # Check original state
            # Cancel the screen update job
            if hasattr(self, 'screen_update_job') and self.screen_update_job:
                try: self.root.after_cancel(self.screen_update_job)
                except tk.TclError: pass
                self.screen_update_job = None
                logger.info("Screen update job cancelled.")
            # Destroy toolbar
            if hasattr(self, 'floating_toolbar') and self.floating_toolbar is not None:
                 if self.floating_toolbar.winfo_exists():
                      try:
                           # Unbind protocol handler first to prevent recursion if destroyed by X button
                           self.floating_toolbar.protocol("WM_DELETE_WINDOW", lambda: None)
                           self.floating_toolbar.destroy()
                           logger.info("Floating toolbar destroyed.")
                      except tk.TclError: pass
            self.floating_toolbar = None
            # Restore main window
            if hasattr(self, 'root') and self.root.winfo_exists() and self.root.state() == 'withdrawn':
                try: self.root.deiconify()
                except tk.TclError: pass
                logger.info("Main window restored.")

        # --- Process and Save Summary (Only if detection was active) ---
        if was_detection_active and self.rt_frame_count > 0 and self.rt_results_list:
            logger.info(f"Processing real-time session summary: {len(self.rt_results_list)} results stored.")
            try:
                all_individual_probs = [item['prob'] for item in self.rt_results_list if 'prob' in item]
                all_thumb_paths = [item['thumb_path'] for item in self.rt_results_list if 'thumb_path' in item]
                avg_fake_prob = sum(all_individual_probs) / len(all_individual_probs) if all_individual_probs else 0.0
                avg_confidence_percent = avg_fake_prob * 100
                overall_result_label = "FAKE" if avg_fake_prob >= self.detector.optimal_threshold else "REAL"

                detailed_report = {
                    "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
                    "detection_type": "real-time-detailed",
                    "file_path": None,
                    "summary": {
                        "total_faces_processed": self.rt_frame_count,
                        "real_detections": self.rt_real_count,
                        "fake_detections": self.rt_fake_count,
                        "average_fake_probability": f"{avg_confidence_percent:.2f}%",
                        "overall_result": overall_result_label,
                    },
                    # Store the *list* of individual probabilities
                    "results": all_individual_probs,
                    # Store the corresponding list of thumbnail paths
                    "face_thumbnails": all_thumb_paths
                }
                logger.debug(f"STOP_DETECTION - Saving detailed_report:")
                logger.debug(f"  Timestamp: {detailed_report.get('timestamp')}")
                logger.debug(f"  Type: {detailed_report.get('detection_type')}")
                logger.debug(f"  Num Results: {len(detailed_report.get('results', []))}")
                logger.debug(f"  Num Thumbnails: {len(detailed_report.get('face_thumbnails', []))}")
                logger.debug(f"  First 5 results: {detailed_report.get('results', [])[:5]}")
                logger.debug(f"  First 5 thumbnails: {detailed_report.get('face_thumbnails', [])[:5]}")
                self.update_scan_history(detailed_report) # Save the modified structure
                logger.info("Real-time session details saved.") # Changed log message
            except Exception as e:
                logger.error(f"Failed to process or save real-time detailed report: {e}", exc_info=True)


        # --- Reset counters (only if detection was active) ---
        if was_detection_active:
            self.rt_frame_count = 0
            self.rt_real_count = 0
            self.rt_fake_count = 0
            self.rt_results_list = []

        # --- Reset UI Button States LAST ---
        self._update_realtime_button_states() # Update main RealtimePage buttons
        self._update_toolbar_button_states() # Attempt to update toolbar buttons (harmless if gone)

        logger.info("stop_detection finished.")

    def update_camera_feed(self):
        """Continuously update the camera feed. Performs detection only if detection_active is True."""
        # Loop based primarily on the self.camera_on flag
        while self.camera_on:
            frame = None # Initialize frame to None for each loop iteration
            try:
                # --- Check camera state *inside* the loop, BEFORE reading ---
                cap_instance = getattr(self, 'cap', None)
                if cap_instance is None:
                    # print("[DEBUG] Camera object (self.cap) is None. Waiting...") # Reduce noise
                    time.sleep(0.5)
                    continue # Skip rest of the loop iteration, check self.camera_on again

                if not cap_instance.isOpened():
                    print("[WARNING] Camera is not opened. Waiting...")
                    time.sleep(0.5)
                    continue # Skip rest of the loop iteration

                # --- Read and Flip Frame ---
                ret, frame = cap_instance.read()
                if not ret or frame is None:
                    print("[ERROR] Failed to capture frame (ret=False or frame is None).")
                    time.sleep(0.1)
                    continue # Try reading again

                frame = cv2.flip(frame, 1) # 1 for horizontal flip

            except cv2.error as e:
                print(f"[ERROR] OpenCV error during capture/check/flip: {e}")
                time.sleep(0.5)
                continue
            except Exception as e:
                print(f"[ERROR] Unexpected error during camera read/check phase: {e}")
                time.sleep(1.0)
                continue

            # --- Face Detection and Prediction Logic (conditional) ---
            if frame is not None:
                frame_for_display = frame.copy()
                if self.detection_active:
                    # frame_with_boxes = frame.copy() # No need for this copy if drawing on frame_for_display
                    try:
                        rgb_frame_detect = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_image_detect = Image.fromarray(rgb_frame_detect)
                        # Get faces *and* boxes
                        detected_faces_pil, detected_boxes = self.face_extractor.extract_faces_and_boxes_realtime(pil_image_detect)

                        if detected_faces_pil:
                            for face_pil, box in zip(detected_faces_pil, detected_boxes):
                                # Box is [x1, y1, x2, y2]
                                prediction_result = self.detector.predict_pil(face_pil)

                                if prediction_result:
                                    result_label, probability_fake = prediction_result
                                    current_time = time.time()
                                    store_this_result = (current_time - self.rt_last_store_time) >= self.rt_store_interval

                                    if store_this_result:
                                        # --- Store Throttled Result ---
                                        timestamp_ms = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                                        thumb_filename = f"rt_face_{timestamp_ms}.jpg"
                                        abs_thumb_path = os.path.join(self.thumbnails_dir, thumb_filename)
                                        try:
                                            os.makedirs(self.thumbnails_dir, exist_ok=True)
                                            face_thumb_save = face_pil.copy()
                                            face_thumb_save.thumbnail((100, 100))
                                            face_thumb_save.save(abs_thumb_path)

                                            self.rt_results_list.append({'prob': probability_fake, 'thumb_path': abs_thumb_path})
                                            self.rt_last_store_time = current_time
                                            self.rt_frame_count += 1 # Increment total count only when stored
                                            is_fake_for_count = probability_fake >= self.detector.optimal_threshold
                                            if is_fake_for_count: self.rt_fake_count += 1
                                            else: self.rt_real_count += 1
                                        except Exception as save_err:
                                            logger.error(f"Failed to save rt thumbnail {abs_thumb_path}: {save_err}")
                                    # --- End Store Throttled Result ---

                                    # --- Determine Label/Conf for *Display* (Always) ---
                                    is_fake = probability_fake >= self.detector.optimal_threshold
                                    display_conf = probability_fake * 100 if is_fake else (1 - probability_fake) * 100
                                    color = (0, 0, 255) if is_fake else (0, 255, 0)
                                    text = f"{result_label} ({display_conf:.1f}%)"

                                    # --- Draw Box on Display Frame ---
                                    x1, y1, x2, y2 = box # Use the box returned by extractor
                                    cv2.rectangle(frame_for_display, (x1, y1), (x2, y2), color, 2)
                                    cv2.putText(frame_for_display, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                else:
                                    logger.warning("Real-time camera prediction failed for a face.")
                                    # Optionally draw a neutral box for undetected?
                                    # x1, y1, x2, y2 = box
                                    # cv2.rectangle(frame_for_display, (x1, y1), (x2, y2), (128, 128, 128), 1)

                    except Exception as e:
                        logger.error(f"Error during real-time detection processing: {e}", exc_info=True)

                # --- Display Logic (Always runs if frame acquired) ---
                try:
                    image_display = Image.fromarray(cv2.cvtColor(frame_for_display, cv2.COLOR_BGR2RGB))
                    photo = ImageTk.PhotoImage(image_display)

                    rt_page = self.pages.get("RealtimePage")
                    video_label_widget = rt_page.video_label if rt_page else None

                    if video_label_widget and video_label_widget.winfo_exists():
                        video_label_widget.configure(image=photo, text="")
                        video_label_widget.image = photo
                    elif self.camera_on: # Avoid warning if camera is being stopped
                        logger.warning("Camera feed: RealtimePage video label lost.")

                    # No need for root.after here, as update_idletasks is okay from thread
                    if hasattr(self.root, 'update_idletasks') and self.root.winfo_exists():
                        self.root.update_idletasks()

                except (RuntimeError, tk.TclError) as e:
                    logger.warning(f"Error updating UI (window likely closed): {e}")
                    self.camera_on = False # Stop loop if UI gone
                except Exception as e:
                    logger.error(f"Unexpected error during camera UI update: {e}", exc_info=True)
                    self.camera_on = False # Stop loop on error

            # Yield CPU time slightly longer if detection is off?
            sleep_time = 0.01 if self.detection_active else 0.03
            time.sleep(sleep_time)

        # --- Loop End ---
        logger.info("Camera feed loop finished (camera_on is False).")

    def update_screen_feed(self):
        """Update the screen feed and handle face detection."""
        if not self.screen_monitoring: 
            return

        try:
            with mss() as sct:
                monitor = sct.monitors[1]
                screenshot = sct.grab(monitor)
                frame = np.array(screenshot)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                frame_with_boxes = frame.copy()
                detected_in_this_frame = False

                # --- Detection Logic ---
                if self.detection_active:
                    try:
                        rgb_frame_detect = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_image_detect = Image.fromarray(rgb_frame_detect)
                        # Get faces *and* boxes
                        detected_faces_pil, detected_boxes = self.face_extractor.extract_faces_and_boxes_realtime(pil_image_detect)

                        if detected_faces_pil:
                            for face_pil, box in zip(detected_faces_pil, detected_boxes):
                                # --- Prediction ---
                                prediction_result = self.detector.predict_pil(face_pil)

                                if prediction_result:
                                    detected_in_this_frame = True # Mark frame as having detections
                                    result_label, probability_fake = prediction_result
                                    current_time = time.time()
                                    store_this_result = (current_time - self.rt_last_store_time) >= self.rt_store_interval

                                    # --- Store Throttled Result ---
                                    if store_this_result:
                                        timestamp_ms = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                                        thumb_filename = f"rt_face_{timestamp_ms}.jpg"
                                        abs_thumb_path = os.path.join(self.thumbnails_dir, thumb_filename)
                                        try:
                                            os.makedirs(self.thumbnails_dir, exist_ok=True)
                                            face_thumb_save = face_pil.copy()
                                            face_thumb_save.thumbnail((100, 100))
                                            face_thumb_save.save(abs_thumb_path)

                                            self.rt_results_list.append({'prob': probability_fake, 'thumb_path': abs_thumb_path})
                                            self.rt_last_store_time = current_time
                                            self.rt_frame_count += 1
                                            is_fake_for_count = probability_fake >= self.detector.optimal_threshold
                                            if is_fake_for_count: self.rt_fake_count += 1
                                            else: self.rt_real_count += 1
                                        except Exception as save_err:
                                            logger.error(f"Failed to save rt thumbnail {abs_thumb_path}: {save_err}")
                                    # --- End Store Throttled Result ---

                                    # --- Determine Label/Conf for *Display* (Always) ---
                                    is_fake = probability_fake >= self.detector.optimal_threshold
                                    display_conf = probability_fake * 100 if is_fake else (1 - probability_fake) * 100
                                    color = (0, 0, 255) if is_fake else (0, 255, 0)
                                    text = f"{result_label} ({display_conf:.1f}%)"

                                    # --- Draw Box on Display Frame ---
                                    x1, y1, x2, y2 = box
                                    cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), color, 2)
                                    cv2.putText(frame_with_boxes, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                                else:
                                    logger.warning("Real-time screen prediction failed for a face.")

                    except Exception as det_err:
                        logger.error(f"Error during screen detection processing: {det_err}", exc_info=True)

        except mss.ScreenShotError as sct_err:
            logger.error(f"Screen capture error: {sct_err}", exc_info=True)
            # Avoid messagebox loop if error persists, just stop
            if self.screen_monitoring: self.toggle_screen_monitoring()
            return
        except Exception as e:
            logger.error(f"Error in screen capture/processing loop: {str(e)}", exc_info=True)
            time.sleep(1) # Wait before retrying

        # Schedule the next update ONLY if screen monitoring is still active
        if self.screen_monitoring and hasattr(self.root, 'after'):
            try:
                self.screen_update_job = self.root.after(100, self.update_screen_feed)
            except tk.TclError:
                logger.warning("Failed to reschedule screen update (window closed?).")

    def _update_realtime_button_states(self):
        """Helper to update buttons on the RealtimePage."""
        rt_page = self.pages.get("RealtimePage")
        if rt_page and hasattr(rt_page, 'update_button_states') and rt_page.winfo_exists():
            try: 
                rt_page.update_button_states()
            except tk.TclError: pass # Ignore if page destroyed

    def _update_toolbar_button_states(self):
        """Helper to update buttons ON the FloatingToolbar."""
        if hasattr(self, 'floating_toolbar') and self.floating_toolbar is not None and self.floating_toolbar.winfo_exists():
            try:
                # Access the update method of the toolbar INSTANCE
                self.floating_toolbar.update_button_states()
            except tk.TclError: pass # Ignore if toolbar destroyed
            except AttributeError: logger.error("FloatingToolbar instance missing update_button_states method?")

    def show_notifications(self):
        """Display current session status."""
        if not self.detection_active:
             messagebox.showinfo("Notifications", "Real-time detection is not active.")
             return

        total_processed = self.rt_frame_count
        msg = f"Current Real-Time Session:\n\n"
        msg += f"- Faces Analyzed & Stored: {total_processed}\n"
        msg += f"- Real Detections (Stored): {self.rt_real_count}\n"
        msg += f"- Fake Detections (Stored): {self.rt_fake_count}\n\n"

        if self.rt_results_list:
             all_probs = [item['prob'] for item in self.rt_results_list if 'prob' in item]
             avg_fake_prob = sum(all_probs) / len(all_probs) if all_probs else 0.0
             avg_conf_pct = avg_fake_prob * 100
             overall_status = "Likely Fake" if avg_fake_prob >= self.detector.optimal_threshold else "Likely Real"
             msg += f"- Avg. Fake Probability (Stored): {avg_conf_pct:.1f}%\n"
             msg += f"- Overall Trend (Stored): {overall_status}"
        else:
             msg += "(No results stored yet)"
        messagebox.showinfo("Real-Time Status", msg)

    def back_to_home_from_function(self):
        """Navigate back to the home page, stopping activity."""
        if self.detection_active or self.screen_monitoring or self.camera_on:
            logger.info("Stopping active detection/monitoring/camera before going home...")
            self.stop_detection() # This will also stop camera if needed

        # Go home (clear_frame is handled by page creation)
        self.show_home_page()

    def confirm_save_before_leaving(self, scan_data, next_action):
        """Ask user to save result before leaving detailed report (only if generated from a scan)."""
        response = messagebox.askyesno("Save Result?", "Save this scan result to history?")
        if response:
            ts_to_check = scan_data.get('timestamp')
            # Check against the current in-memory history
            exists = any(s.get('timestamp') == ts_to_check and s.get('detection_type') == 'scanned'
                        for s in self.scan_history)
            if not exists:
                self.scan_history.append(scan_data) # Add to in-memory list
                self.save_scan_history() # Save the whole list to file
                logger.info("Scan result saved from confirmation prompt.")
            else:
                logger.info("Scan result already exists in history (from confirm prompt).")
        else:
            logger.info("Scan result discarded from confirmation prompt.")
            # Clean up corresponding scan thumbnails if discarded
            if scan_data.get('face_thumbnails'):
                try:
                    first_thumb_abs = scan_data['face_thumbnails'][0]
                    scan_thumb_dir = os.path.dirname(first_thumb_abs)
                    # Basic safety check: ensure we are deleting within the expected thumbnails dir
                    if os.path.exists(scan_thumb_dir) and scan_thumb_dir.startswith(self.thumbnails_dir):
                        shutil.rmtree(scan_thumb_dir)
                        logger.info(f"Discarded thumbnails directory: {scan_thumb_dir}")
                    else:
                        logger.warning(f"Refusing to delete potentially incorrect directory: {scan_thumb_dir}")
                except IndexError:
                    logger.warning("Thumbnail list was empty when trying to discard.")
                except Exception as e:
                    logger.warning(f"Could not remove discarded thumbnails dir: {e}")

        # --- Clear report state AFTER handling save/discard ---
        self.current_report_data = None
        self.report_came_from_scan = False
        # --- Proceed with navigation ---
        next_action()

    def save_scan_result(self, scan_data):
        """Adds scan data to history and saves."""
        # Ensure data is not already present if called multiple times
        # Simple check based on timestamp might suffice, or use a more robust ID
        # For now, assume it's called correctly once per scan to save
        logger.info("Saving scan result to history.")
        self.scan_history.append(scan_data)
        self.save_scan_history() # Persist changes
        messagebox.showinfo("Saved", "Scan result saved to history.")

    def delete_scan(self, scan_data_to_delete):
        """Deletes a scan result from history and files."""
        logger.warning(f"Attempting to delete scan: {scan_data_to_delete.get('timestamp')}")
        response = messagebox.askyesno("Confirm Delete", "Are you sure you want to delete this scan result and its associated files?")
        if not response:
            logger.info("Deletion cancelled by user.")
            return

        original_length = len(self.scan_history)
        # Find the specific entry to remove (matching timestamp might be sufficient)
        ts_to_delete = scan_data_to_delete.get('timestamp')
        new_history = [scan for scan in self.scan_history if scan.get('timestamp') != ts_to_delete]

        if len(new_history) < original_length:
             self.scan_history = new_history
             logger.info("Scan entry removed from internal history list.")

             # Delete associated thumbnail files/directory
             if "face_thumbnails" in scan_data_to_delete:
                logger.info(f"Deleting {len(scan_data_to_delete['face_thumbnails'])} associated thumbnails.")
                # Keep track of unique directories to potentially remove later if empty
                thumb_dirs = set()
                for thumb_abs_path in scan_data_to_delete["face_thumbnails"]:
                    try:
                        if os.path.exists(thumb_abs_path):
                            os.remove(thumb_abs_path)
                            logger.debug(f"Deleted thumbnail file: {thumb_abs_path}")
                            # Record the directory
                            thumb_dirs.add(os.path.dirname(thumb_abs_path))
                        else:
                            logger.warning(f"Thumbnail file not found for deletion: {thumb_abs_path}")
                    except Exception as e:
                        logger.error(f"Error deleting thumbnail file {thumb_abs_path}: {e}")
                        
                for thumb_dir in thumb_dirs:
                    try:
                        # Basic safety check: only remove if it's within the main thumbnails dir
                        if os.path.exists(thumb_dir) and thumb_dir.startswith(self.thumbnails_dir) and not os.listdir(thumb_dir):
                            os.rmdir(thumb_dir)
                            logger.debug(f"Removed empty thumbnail directory: {thumb_dir}")
                    except OSError as e: # Catch directory not empty or permissions error
                        logger.warning(f"Could not remove directory {thumb_dir} (maybe not empty?): {e}")
                    except Exception as e:
                        logger.error(f"Error removing directory {thumb_dir}: {e}")

             # Save the updated history back to the file
             self.save_scan_history()
             messagebox.showinfo("Deleted", "Scan result deleted successfully.")
             # Refresh the history view
             self.show_history_page()
        else:
             logger.warning("Scan data to delete not found in current history.")
             messagebox.showwarning("Not Found", "Could not find the specified scan result to delete.")

    def update_scan_history(self, new_entry):
        history_path = self.history_file
        history = []
        logger.debug(f"Attempting to update history file: {history_path}")

        # Use lock for thread safety
        with history_lock:
            if os.path.exists(history_path):
                try:
                    # Read existing history inside the lock
                    with open(history_path, 'r', encoding='utf-8') as f: # Add encoding
                        loaded_data = json.load(f)
                        if isinstance(loaded_data, list):
                            history = loaded_data
                        else:
                            logger.warning(f"History file format error {history_path}, resetting.")
                            history = []
                except json.JSONDecodeError:
                    logger.error(f"Failed to decode history JSON from {history_path}. Resetting.", exc_info=True)
                    history = []
                except Exception as e:
                    logger.error(f"Error reading scan history {history_path}: {e}", exc_info=True)
                    history = []
            else:
                logger.info(f"History file {history_path} not found, creating new list.")
                history = []

            # Append new entry (e.g., real-time summary)
            history.append(new_entry)
            # Update the in-memory history as well
            self.scan_history = history

            # Write the entire updated list back
            try:
                # Ensure directory exists (might be redundant but safe)
                os.makedirs(os.path.dirname(history_path), exist_ok=True)
                with open(history_path, 'w', encoding='utf-8') as f: # Add encoding
                    json.dump(history, f, indent=4)
                logger.info(f"Scan history successfully updated: {history_path}")
            except Exception as e:
                logger.error(f"Error writing scan history {history_path}: {e}", exc_info=True)

    def check_for_updates(self, background=True):
        """Checks server for new model/threshold versions."""
        if background:
            # Run check in a separate thread to avoid blocking UI on start
            thread = threading.Thread(target=self._perform_update_check, daemon=True)
            thread.start()
        else:
            # Run synchronously (e.g., from a button click - add button later if desired)
            self._perform_update_check()

    def _perform_update_check(self):
        """Internal helper to perform the update check logic."""
        logger.info("Checking for updates...")
        version_url = "https://raw.githubusercontent.com/liftlobby/Fakeseeker/refs/heads/main/version.json"

        if version_url == "YOUR_HOSTED_VERSION_JSON_URL":
             logger.warning("Update check URL is not configured. Skipping check.")
             return # Don't proceed if URL is not set

        try:
            # Read local version (create file if doesn't exist)
            local_version = "0.0.0" # Default if no file exists
            os.makedirs(os.path.dirname(self.local_version_file), exist_ok=True)
            if os.path.exists(self.local_version_file):
                try:
                    with open(self.local_version_file, 'r', encoding='utf-8') as f:
                        local_version = f.read().strip()
                        if not local_version: local_version = "0.0.0" # Handle empty file
                except Exception as read_err:
                     logger.error(f"Error reading local version file: {read_err}. Assuming 0.0.0")
                     local_version = "0.0.0"

            logger.info(f"Local version: {local_version}")

            # Fetch server version manifest
            logger.debug(f"Fetching update manifest from: {version_url}")
            response = requests.get(version_url, timeout=15) # Increased timeout slightly
            response.raise_for_status() # Raise error for bad status codes (4xx, 5xx)
            manifest = response.json()
            server_version = manifest.get('version', '0.0.0')
            logger.info(f"Server version: {server_version}")

            # Simple string comparison (for versions like 1.0.0, 1.1.0, 2.0.0)
            if server_version > local_version:
                logger.info("Update available!")
                # Ask user on main thread using root.after (safer than direct call)
                # Ensure root window still exists before scheduling
                if hasattr(self.root, 'after') and self.root.winfo_exists():
                     self.root.after(0, lambda m=manifest: self._ask_user_to_update(m))
                else:
                     logger.warning("Root window gone, cannot ask user to update.")
            else:
                logger.info("Application is up-to-date.")
                # Optionally inform user if check was manual via a button later

        except requests.exceptions.Timeout:
             logger.error("Network timeout checking for updates.")
             # Inform user only if check was manual? Avoid bothering on startup timeout.
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error checking for updates: {e}")
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding version manifest from {version_url}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error checking for updates: {e}", exc_info=True)

    def _ask_user_to_update(self, manifest):
        """Runs in main thread to ask user via messagebox."""
        if not self.root.winfo_exists(): return # Abort if window closed

        notes = manifest.get('release_notes', 'No details available.')
        server_version = manifest.get('version', 'N/A')
        msg = f"A new version ({server_version}) is available!\n\nChanges:\n{notes}\n\nDownload model and threshold update now?"
        try:
             if messagebox.askyesno("Update Available", msg):
                 logger.info("User accepted update. Starting download.")
                 # Start download in background thread
                 thread = threading.Thread(target=self._download_update_files, args=(manifest,), daemon=True)
                 thread.start()
             else:
                 logger.info("User declined update.")
        except Exception as e:
             logger.error(f"Error showing update dialog: {e}")

    def _download_update_files(self, manifest):
        """Downloads model and threshold in a background thread."""
        logger.info("Starting update download process...")
        # Ensure target directories exist (should have been done in _setup_directories)
        try:
             os.makedirs(self.user_model_dir, exist_ok=True)
             os.makedirs(os.path.dirname(self.user_threshold_path), exist_ok=True)
        except Exception as e:
             logger.error(f"Failed to create user directories for update: {e}")
             self.root.after(0, lambda: messagebox.showerror("Update Failed", f"Could not create necessary directories:\n{e}"))
             return

        model_url = manifest.get('model_url')
        threshold_url = manifest.get('threshold_url')
        model_filename = manifest.get('model_filename') # Use filename from manifest
        server_version = manifest.get('version', 'unknown') # Get version to save

        if not model_url or not threshold_url or not model_filename:
             logger.error("Update manifest is missing required URLs or model filename.")
             self.root.after(0, lambda: messagebox.showerror("Update Failed", "Update information from server is incomplete."))
             return

        model_save_path = os.path.join(self.user_model_dir, model_filename)
        threshold_save_path = self.user_threshold_path

        try:
            # Download Model (with simple progress logging)
            logger.info(f"Downloading model from {model_url} to {model_save_path}")
            total_size = 0
            with requests.get(model_url, stream=True, timeout=60) as r: # Longer timeout for model
                 r.raise_for_status()
                 total_size = int(r.headers.get('content-length', 0))
                 bytes_downloaded = 0
                 last_log_time = time.time()
                 with open(model_save_path, 'wb') as f:
                      for chunk in r.iter_content(chunk_size=8192 * 4): # Larger chunk size
                           f.write(chunk)
                           bytes_downloaded += len(chunk)
                           # Log progress periodically
                           current_time = time.time()
                           if total_size > 0 and current_time - last_log_time > 2: # Log every 2 seconds
                                progress = (bytes_downloaded / total_size) * 100
                                logger.info(f"Model download progress: {progress:.1f}%")
                                last_log_time = current_time
            logger.info(f"Model download complete ({bytes_downloaded} bytes).")

            # Checksum verification
            checksum_expected = manifest.get('model_checksum')
            if checksum_expected:
                logger.info(f"Verifying model checksum. Expected: {checksum_expected}") # Add log
                # Calculate hash of downloaded file
                with open(model_save_path, 'rb') as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()
                logger.info(f"Calculated checksum for downloaded file: {file_hash}") # Add log
                if file_hash != checksum_expected:
                    logger.error(f"Model checksum verification FAILED: Expected {checksum_expected}, got {file_hash}")
                    # Optionally delete the corrupted downloaded file
                    try:
                        os.remove(model_save_path)
                        logger.info(f"Deleted corrupted model file: {model_save_path}")
                    except Exception as del_e:
                        logger.error(f"Error deleting corrupted model file: {del_e}")
                    self.root.after(0, lambda: messagebox.showerror("Update Failed", "Model checksum verification failed. The downloaded file might be corrupted. Please try the update again."))
                    return # Stop the update process
                logger.info("Model checksum verification successful.")
            else:
                logger.warning("No model_checksum found in manifest. Skipping verification.")

            # Download Threshold
            logger.info(f"Downloading threshold from {threshold_url} to {threshold_save_path}")
            response_thresh = requests.get(threshold_url, timeout=15)
            response_thresh.raise_for_status()
            with open(threshold_save_path, 'wb') as f: 
                f.write(response_thresh.content)
            logger.info("Threshold download complete.")

            # Update local version file
            try:
                 with open(self.local_version_file, 'w', encoding='utf-8') as f:
                     f.write(server_version)
                 logger.info(f"Local version updated to {server_version}")
            except Exception as vf_err:
                 logger.error(f"Failed to write local version file: {vf_err}")

            # Inform user on main thread
            if hasattr(self.root, 'after') and self.root.winfo_exists():
                 self.root.after(0, lambda: messagebox.showinfo("Update Complete", f"Update {server_version} downloaded successfully.\nPlease restart the application to use the new model."))

        except requests.exceptions.RequestException as e:
             logger.error(f"Download error: {e}")
             if hasattr(self.root, 'after') and self.root.winfo_exists():
                  self.root.after(0, lambda: messagebox.showerror("Update Failed", f"Download failed:\n{e}"))
        except Exception as e:
             logger.error(f"Error saving downloaded files: {e}", exc_info=True)
             if hasattr(self.root, 'after') and self.root.winfo_exists():
                  self.root.after(0, lambda: messagebox.showerror("Update Failed", f"Could not save downloaded files:\n{e}"))

    def get_file_details(self, file_path):
        """Get detailed information about the file."""
        details = {}
        try:
            # Basic file info
            file_stats = os.stat(file_path)
            details['size'] = self.format_file_size(file_stats.st_size)
            details['type'] = os.path.splitext(file_path)[1].upper()[1:]  # Remove the dot
            
            # Get image/video resolution
            if details['type'].lower() in ['jpg', 'jpeg', 'png', 'bmp']:
                with Image.open(file_path) as img:
                    details['resolution'] = f"{img.width}x{img.height}"
            elif details['type'].lower() in ['mp4', 'avi', 'mov']:
                cap = cv2.VideoCapture(file_path)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                details['resolution'] = f"{width}x{height} @ {fps}fps"
                cap.release()
            
            # Get file creation and modification times
            details['created'] = datetime.fromtimestamp(file_stats.st_ctime).strftime('%Y-%m-%d %H:%M:%S')
            details['modified'] = datetime.fromtimestamp(file_stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            
            # Get file source (directory path)
            details['source'] = os.path.dirname(file_path)
            
        except Exception as e:
            logger.error(f"Error getting file details: {e}")
        
        return details

    def format_file_size(self, size_in_bytes):
        """Convert bytes to human readable format."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_in_bytes < 1024:
                return f"{size_in_bytes:.1f} {unit}"
            size_in_bytes /= 1024
        return f"{size_in_bytes:.1f} TB"
    
    def _update_realtime_button_states(self):
        """Helper to update buttons on the RealtimePage."""
        rt_page = self.pages.get("RealtimePage")
        if rt_page and hasattr(rt_page, 'update_button_states') and rt_page.winfo_exists():
            rt_page.update_button_states()

if __name__ == "__main__":
    if sys.platform == 'win32':
        try:
            awareness = 1
            ctypes.windll.shcore.SetProcessDpiAwareness(awareness)
        except AttributeError:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception as e:
                logger.warning(f"Error calling SetProcessDPIAware: {e}")

    logger.info("Starting FakeSeeker application...")
    root = tk.Tk(className="FakeSeeker")
    root.withdraw()
    app = None
    try:
        app = FakeSeekerApp(root)
        root.after(100, root.deiconify)
        root.mainloop()
    except Exception as main_err:
         logger.critical(f"Unhandled exception in main loop: {main_err}", exc_info=True)
         try:
             if root.winfo_exists(): # Check if window exists before showing error
                  messagebox.showerror("Fatal Error", f"A critical error occurred:\n{main_err}\n\nCheck fakeseeker.log for details.")
         except: pass # Ignore errors during error display itself
    finally:
         logger.info("Application attempting final cleanup...")
         # Ensure cleanup runs even if __init__ failed partially or mainloop exited abruptly
         if app and hasattr(app, 'on_closing') and callable(app.on_closing):
              pass
         elif root and root.winfo_exists():
             try: root.destroy()
             except tk.TclError: pass # Ignore if already destroying
         logger.info("Application finished.")