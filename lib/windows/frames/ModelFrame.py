import tkinter as tk
from tkinter import ttk
import pandas as pd
from pathlib import Path
from typing import Callable

from DriveIdent.lib.components.ImageGallery import ImageGallery

class ModelFrame(tk.Frame):
    '''
    This Class creates a Frame to display Model information and plots after training is completed.
    It consists of a LabelFrame on the left side, which displays the model accuracies 
    and a LabelFrame on the right, which contaings a ImageGallery to display the plots generated in the training process.
    '''

    def __init__(self, parent, styleConfig : dict, onNext : Callable, modelAccuracyData : pd.DataFrame = pd.DataFrame(columns=["Model", "Precision"]), images : list[str] = []):
        '''
        Constructs a ModelFrame.

        Args:
            parent: Tkinter Parent Object where this Object is placed into (ie. a Frame)
            styleConfig: A dictionary containing stylization information
            onNext: Callback Function which is called when the Weiter-Button is pressed
            modelAccuracyData: pd.DataFrame containing the accuracy data of the models
            images: A List of strings containing file paths of images
        '''
        super().__init__(parent)
        self.modelAccurarcyData = modelAccuracyData
        print(self.modelAccurarcyData)
        self.images = images

        leftFrame = tk.LabelFrame(self, text="Genauigkeit", padx=styleConfig["paddings"]["default"], pady=styleConfig["paddings"]["default"], font=styleConfig["font"]["h3"])
        leftFrame.grid(row=0, column=0, padx=styleConfig["paddings"]["default"], pady=styleConfig["paddings"]["default"], sticky="nsew")

        for _, row in modelAccuracyData.iterrows():
            tk.Label(leftFrame, text=f"{row['Model']}-Modell:", font=styleConfig["font"]["h4"], anchor="w").pack(fill="x", pady=styleConfig["paddings"]["slim"])
            tk.Label(leftFrame, text=f"{row['Precision']*100:.2f}% Genauigkeit", font=styleConfig["font"]["text"], anchor="w").pack(fill="x")

        rightFrame = tk.LabelFrame(self, text="Diagramme", padx=styleConfig["paddings"]["default"], pady=styleConfig["paddings"]["default"], font=styleConfig["font"]["h3"])
        rightFrame.grid(row=0, column=1, padx=styleConfig["paddings"]["default"], pady=styleConfig["paddings"]["default"], sticky="nsew")

        gallery = ImageGallery(rightFrame, styleConfig, images)
        gallery.pack(fill="both", expand=True)

        nextButton = tk.Button(self, text="Weiter", command=onNext, width=20, height=2)
        nextButton.grid(row=1, column=0, columnspan=2, pady=styleConfig["paddings"]["wide"])
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(0, weight=1)
        rightFrame.grid_propagate(False)
        leftFrame.grid_propagate(False)
