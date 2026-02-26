import tkinter as tk
from PIL import Image, ImageTk
from .GenericButton import GenericButton


class ImageGallery(tk.Frame):
    '''
    This Component class creates a ImageGallery.
    It takes a List of File-Paths to load images.
    Has Buttons to cycle between these images.
    '''

    def __init__(self, parent, styleConfig, images: list[str]):
        '''
        Constructor of EditableTable.

        Args:
            parent: Tkinter Parent Object where this Object is placed into (ie. a Frame)
            styleConfig: A dictionary containing stylization information
            images: A List of strings containing file paths of images
        '''
        super().__init__(parent)

        self.images = images
        self.index = 0

        self.original_img = None
        self.tk_img = None

        # Displays the images
        self.imageLabel = tk.Label(self, bd=0)
        self.imageLabel.pack(fill="both", expand=True)

        # Buttons to cycle images
        button_frame = tk.Frame(self)
        button_frame.pack(pady=styleConfig["paddings"]["tight"])

        self.prevButton = GenericButton(
            button_frame, styleConfig,
            text="Zurück",
            command=self.prevImage
        )
        self.prevButton.pack(side="left", padx=5)

        self.nextButton = GenericButton(
            button_frame, styleConfig,
            text="Weiter",
            command=self.nextImage
        )
        self.nextButton.pack(side="left", padx=5)

        # Resize automatically
        self.imageLabel.bind("<Configure>", self.onResize)

        # Load first image
        self.loadImage()

    def loadImage(self):
        """Loads a image and displays it. Automatically resizes the Container"""
        path = self.images[self.index]
        self.original_img = Image.open(path)
        self.resizeAndShow()
        self.prevButton.config(state="normal" if self.index > 0 else "disabled")
        self.nextButton.config(state="normal" if self.index < len(self.images)-1 else "disabled")

    def resizeAndShow(self):
        """Resizes the Image and displays it"""

        if self.original_img is None:
            return

        w = self.imageLabel.winfo_width()
        h = self.imageLabel.winfo_height()

        if w < 10 or h < 10:
            return

        # Retain aspect ratio, to avoid stretching / compression
        img = self.original_img.copy()
        img.thumbnail((w, h))

        self.tk_img = ImageTk.PhotoImage(img)
        self.imageLabel.config(image=self.tk_img)

    def onResize(self, event):
        """Resizes the displayed image upon resizing of the window"""
        self.resizeAndShow()

    def nextImage(self):
        """Displays the next image in the List"""
        if self.index < len(self.images) - 1:
            self.index += 1
            self.loadImage()

    def prevImage(self):
        """Displays the previous image in the List"""
        if self.index > 0:
            self.index -= 1
            self.loadImage()
