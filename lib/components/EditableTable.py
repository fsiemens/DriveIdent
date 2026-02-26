import tkinter as tk
from tkinter import ttk
import pandas as pd
import os

class EditableTable(ttk.Frame):
    '''
    This Component class creates a table, with editable cells.
    It takes a dataframe and displays all columns in a vertically scrollable tk.Treeview.
    On Double-Click it overlays a tk.Entry field above the clicked cell.
    Once the user un-selects the Entry field, its content is saved at the respective field of the treeview.
    The content of the table can be accessed and converted to a pd.DataFrame by calling EditableTable#getData()
    '''

    def __init__(self, parent, data : pd.DataFrame, editable : bool = True):
        '''
        Constructor of EditableTable.

        Args:
            parent: Tkinter Parent Object where this EditableTable is placed into (ie. a Frame)
            data: pd.DataFrame containing the data that should be displayed
            editable: boolean flag whether or not this EditableTable should be editable
        '''
        super().__init__(parent)
        self.data = data
        self.editable = editable
        self.totalWidth = self.winfo_width()
        
        self.tree = ttk.Treeview(self, columns=list(data.columns), show="headings")
        for i, col in enumerate(data.columns):
            self.tree.heading(col, text=col)
            self.tree.column(col, anchor="w", width=int(self.totalWidth*(2 if i == 0 else 1)/(1 + len(data.columns))))

        # Put data in table
        self._addRows()

        # Make Cells editable on double click
        self.tree.bind("<Double-1>", self._onDoubleClick)

        # Add scroll bar
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        self.tree.config(yscrollcommand=scrollbar.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        self.entry = None

    def _onDoubleClick(self, event):
        '''
        Opens an Entry Field on top of a double clicked treeview cell

        Args:
            event: tkinter Treeview event
        '''
        if not self.editable:
            return

        region = self.tree.identify("region", event.x, event.y)
        if region != "cell":
            return
        
        rowId = self.tree.identify_row(event.y)
        col = self.tree.identify_column(event.x)
        
        # Make the left-most column uneditable
        if col == "#1":
            return

        x, y, width, height = self.tree.bbox(rowId, col)
        colIndex = int(col.replace("#", "")) - 1
        editingColumn = self.data.columns[colIndex]
        value = self.tree.item(rowId, "values")[colIndex]
        
        # Create Overlay for editing the cell
        self.entry = tk.Entry(self.tree)
        self.entry.place(x=x, y=y, width=width, height=height)
        self.entry.insert(0, value)
        self.entry.focus()

        # Save Entry in cell after the user un-focusses the entry field
        self.entry.bind("<Return>", lambda e: self._saveEdit(rowId, editingColumn))     
        self.entry.bind("<FocusOut>", lambda e: self._saveEdit(rowId, editingColumn))

    def _saveEdit(self, rowId, colId):
        '''
        Stores the input in an Entry to the corresponding treeview cell

        Args:
            rowId: String Id of the Row of the cell
            colId: String Id of the Column of the cell
        '''
        if self.entry:
            newValue = self.entry.get()
            rowIndex = self.tree.index(rowId)
            self.data.iloc[rowIndex, self.data.columns.get_loc(colId)] = newValue   # type: ignore
            self.tree.set(rowId, column=colId, value=newValue)
            self.entry.destroy()
            self.entry = None

    def refresh(self):
        '''
        Deletes all treeview rows and re-adds them based on the contents of EditableTable#data
        '''
        print("refreshing table")
        for row_id in self.tree.get_children():
            self.tree.delete(row_id)

        self._addRows()

    def _addRows(self):
        '''
        Adds Rows to the treeview based on the contents of EditableTable#data.
        '''
        for _, row in self.data.iterrows():
            values = list(row)
            # Crop File-Column so that only the file name is shown
            values[0] = os.path.basename(str(values[0]))
            self.tree.insert("", "end", values=values)

    def getData(self) -> pd.DataFrame:
        '''
        Returns a pd.DataFrame containing the content of this table
        '''
        data = [self.tree.item(row_id, "values") for row_id in self.tree.get_children()]
        return pd.DataFrame(data, columns=self.data.columns)