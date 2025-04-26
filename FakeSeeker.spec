# -*- mode: python ; coding: utf-8 -*-

# Required for finding data files when bundled
import sys
import os
import facenet_pytorch

block_cipher = None

# --- DEFINE PATHS relative to this .spec file ---
project_root = os.getcwd()

# --- Find the path to the installed facenet_pytorch data ---
facenet_pytorch_pkg_dir = os.path.dirname(facenet_pytorch.__file__)
facenet_pytorch_data_path = os.path.join(facenet_pytorch_pkg_dir, 'data')

# --- Paths to your project files/dirs ---
threshold_file_path = os.path.join(project_root, 'optimal_threshold.json') # Ensure this file exists at project root
images_data_path = os.path.join(project_root, 'images')
ui_pages_path = os.path.join(project_root, 'ui')
logger_setup_path = os.path.join(project_root, 'logger_setup.py')
detector_path = os.path.join(project_root, 'deepfake_detector.py')
extractor_path = os.path.join(project_root, 'face_extractor.py')
# Ensure this path and filename are EXACTLY correct
default_model_src_path = os.path.join(project_root, 'best_model_DEFAULT.pth')

# --- Analysis Section ---
a = Analysis(
    ['main.py'],
    pathex=[project_root],
    binaries=[],
    datas=[
        (threshold_file_path, '.'), # Use the variable
        (images_data_path, 'images'),
        (ui_pages_path, 'ui'),
        (logger_setup_path, '.'),
        (detector_path, '.'),
        (extractor_path, '.'),
        (facenet_pytorch_data_path, 'facenet_pytorch/data'),
        (default_model_src_path, 'default_model'),
    ],
    hiddenimports=[
        'torch', 'torchvision', 'torchaudio',
        'torch.nn', 'torch.nn.functional', 'torch.optim', 'torch.utils', 'torch.utils.data',
        'cv2', 'cv2.data',
        'PIL', 'PIL._imaging', 'PIL._imagingft', 'PIL._imagingtk',
        'numpy', 'numpy.core._multiarray_umath',
        'scipy', # Keep main scipy
        'efficientnet_pytorch',
        'facenet_pytorch', 'facenet_pytorch.models', 'facenet_pytorch.models.mtcnn', 'facenet_pytorch.models.inception_resnet_v1',
        # Removed duplicate mtcnn entry
        'facenet_pytorch.models.utils.detect_face',
        'mss', 'mss.base', 'mss.linux', 'mss.windows', 'mss.darwin',
        'pygetwindow', # Keep main pygetwindow
        'requests', 'requests.certs', 'requests.adapters', 'urllib3', 'urllib3.util', 'urllib3.response',
        'appdirs',
        'tkinter', 'tkinter.ttk', 'tkinter.filedialog', 'tkinter.messagebox',
        'queue', 'threading', 'datetime', 'time', 'json', 'shutil',
        'hashlib',
        'logging.handlers',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='FakeSeeker',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    onefile=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # *** VERIFY THIS FILENAME ***
    icon=os.path.join('images', 'app_icon.ico'), # Or 'fakeseeker_logo.ico'?
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='FakeSeeker',
)

# --- Optional: macOS Application Bundle ---
# Uncomment this section when building ON MACOS
# app = BUNDLE(
#    coll,
#    name='FakeSeeker.app',    # Output application bundle name
#    icon=os.path.join('images', 'app_icon.icns'), # Icon for macOS
#    bundle_identifier='com.yourcompany.fakeseeker', # Optional: Reverse DNS identifier (Change 'yourcompany')
#    # info_plist={             # Optional: Add custom Info.plist entries
#    #     'NSPrincipalClass': 'NSApplication',
#    #     'NSAppleScriptEnabled': False,
#    #     # 'LSUIElement': '1' # Set to 1 to hide dock icon if desired (usually not for main apps)
#    # }
# )