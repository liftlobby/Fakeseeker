import tkinter as tk
from tkinter import ttk

class BasePage(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller # Store reference to the main FakeSeekerApp

    def get_style(self, style_name):
        # Helper to potentially get styles if needed, though direct use is fine
        return self.controller.root.style.lookup(style_name, '.')