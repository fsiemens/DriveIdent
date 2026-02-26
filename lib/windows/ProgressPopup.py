import tkinter as tk
from typing import cast, Callable, Literal
from .frames.ProgressFrame import ProgressFrame

class ProgressPopup(tk.Frame):
    """
    This Class creates a popup displaying the progress of a process.
    The Windows size is 500x600 by default and can not be resized.
    """

    def __init__(self, parent, styleConfig : dict, title : str):
        '''
        Constructor of EditableTable.

        Args:
            parent: Tkinter Parent Object where this Object is depending on
            styleConfig: A dictionary containing stylization information
            title: Title of the Window, which is also displayed at the center
        '''

        popup = tk.Toplevel(parent)
        popup.title(title)
        popup.geometry("500x600")
        popup.resizable(False, False)
        popup.attributes('-topmost', True)  # Allways stay on top
        popup.protocol("WM_DELETE_WINDOW", self.close)  # Execute close if window is closed using the OS-Buttons

        popup.transient(parent)
        popup.grab_set()          # makes window modal

        self.styleConfig = styleConfig
        self.popup = popup

        self.frame = ProgressFrame(popup, styleConfig, title, 100, onCancel=self.close)
        self.frame.pack(pady=styleConfig["paddings"]["default"], expand=True)

    def updateProgress(self, phase, total, completed_list, in_progress_list, message, remaining, percent):
        ''' Callback function which can be handed to the backend_adapters train or predict functions to update their progress. '''
        if not self.popup.winfo_exists():
            return

        self.frame.messageLabel.config(text=message)
        self.frame.progress.config(value=int(percent))

        if phase == "done":
            self.close()

    def close(self):
        ''' Closes the Popup '''
        self.popup.destroy()