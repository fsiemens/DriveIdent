import tkinter as tk
from typing import Callable

class GenericButton(tk.Button):
    '''
    This Component class creates is used as the default Button in this application.
    This allows to use the identical button styling in the whole UI, without unnecessary code repitition
    '''

    def __init__(self, parent, styleConfig : dict, text : str, command : Callable, width : int = -1, height : int = -1):
        '''
        Constructs a GenericButton (a tk.Button with predefined styling).

        Args:
            parent: Tkinter Parent Object where this Object is placed into (ie. a Frame)
            styleConfig: A dictionary containing stylization information
            text: The Label of the button
            command: A Callable which is called when the button is pressed
            width: Integer Width of the button. Use -1 or leave as default to use the width defined in styleConfig
            height: Integer Height of the button. Use -1 or leave as default to use the width defined in styleConfig
        '''
        super().__init__(
            parent, 
            text=text, 
            width=styleConfig["buttonWidth"] if width < 0 else width,
            height=styleConfig["buttonHeight"] if height < 0 else height, 
            command=command,
            relief=styleConfig["buttonRelief"], 
            bg=styleConfig["colors"]["buttonBg"], 
            fg=styleConfig["colors"]["buttonFg"],
            font=styleConfig["font"]["buttonText"]
        )