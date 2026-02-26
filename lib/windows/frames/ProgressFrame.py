import tkinter as tk
from tkinter import ttk
from typing import Callable

class ProgressFrame(tk.Frame):
    '''
    This Class creates a Frame to display the progress of a process.
    It contains a progress bar and a Cancel-Button
    '''
    
    def __init__(self, parent, styleConfig : dict, title : str, maxProgress : int, onCancel : Callable):
        '''
        Constructs a ProgressFrame.

        Args:
            parent: Tkinter Parent Object where this Object is placed into (ie. a Frame)
            styleConfig: A dictionary containing stylization information
            title: Title of this Popup
            maxProgress: integer maximum value of the progress bar
            onCancel: Callback Function which is called when the Abbrechen-Button is pressed
        '''
        super().__init__(parent)
        titleLabel = tk.Label(self, text=title, font=styleConfig["font"]["h1"])
        titleLabel.pack(pady=styleConfig["paddings"]["default"])

        self.progress = ttk.Progressbar(
            self,
            orient="horizontal",
            length=300,
            mode="determinate",
            maximum=maxProgress
        )
        self.progress.pack(pady=styleConfig["paddings"]["default"])

        self.messageLabel = tk.Label(self, text="Starte...", font=styleConfig["font"]["text"], wraplength=400)
        self.messageLabel.pack(pady=styleConfig["paddings"]["default"])

        cancelButton = tk.Button(self, text="Abbrechen", width=12, command=onCancel)
        cancelButton.pack(pady=styleConfig["paddings"]["default"], side="bottom")
        