import tkinter as tk
from tkinter import messagebox
from DriveIdent.lib.components.StepProgressBar import StepProgress
from .frames.TrainingFrame import TrainingFrame
from .frames.ModelFrame import ModelFrame
from .frames.PredictionFrame import PredictionFrame
from .ValidationPopup import ValidationPopup
from .ProgressPopup import ProgressPopup
from DriveIdent.lib.FileImporter import selectFilesFromOS, loadCsvAsDataFrame
from DriveIdent.lib.FileExporter import saveLabelFileOS
from DriveIdent.lib.core.backend_adapter import train, predict, validate_csv, set_config
from typing import Literal, Callable
import pandas as pd
import threading
from pathlib import Path
import os

class MainWindow(tk.Tk):
    """
    This Class creates the main window of this application.
    The Windows size is 1000x900 by default, but it can be resized.
    Furthermore, this Class also contains most of the Frontend logic and data
    """
    def __init__(self, styleConfig : dict):
        '''
        Constructor of EditableTable.

        Args:
            styleConfig: A dictionary containing stylization information
        '''
         
        super().__init__()

        self.styleConfig = styleConfig
        self.geometry("1000x900")
        self.title("DriveIdent")

        self.dataDir = ""
        self.trainFiles : pd.DataFrame = pd.DataFrame(columns=["File", "Label"])
        self.testFiles = pd.DataFrame(columns=["File", "randomforest", "logreg", "gradientboosting"])
        self.options = {}
        self.modelAccuracy = pd.DataFrame(columns=["Model", "Precision"])

        self.stepProgress = StepProgress(self, self.styleConfig, labels=[("Training", self.openTrainingFrame), ("Modell", self.openModelFrame), ("Vorhersage", self.openPredictionFrame)], stepAccessValidationFct=lambda step:step <= self.stepProgress.getCurrentStep())
        self.stepProgress.pack(pady=self.styleConfig["paddings"]["wide"])

        self.currentFrame = None
        self.openTrainingFrame()

    def selectTrainFiles(self):
        ''' Button Callback for the selection of training files '''
        self.selectRecordings(self.trainFiles, "Trainingsdateien auswählen", self.onPopupClose)

    def selectTestFiles(self):
        ''' Button Callback for the selection of test files '''
        self.selectRecordings(self.testFiles, "Test-Dateien auswählen", self.onPopupClose)

    def selectRecordings(self, destination : pd.DataFrame, title : str, onCloseCallback : Callable):
        '''
        Opens a File Selection Window of the OS and allows the user to select multiple recordings (CSVs).
        Then validates those selected recordings, by calling the validation-method of the backend_adapter.
        While validating opens a Popup-Window, which displays the progress of the validation 
        '''
        files = selectFilesFromOS(title, [("CSV-Recordings", "*.csv")])
        popup = ValidationPopup(self, self.styleConfig, destination, files, onPopupClose=onCloseCallback)

        def postValidation(faultyFiles : list[str]):
            ''' 
            Helper Callback, which is called once validation is completed.
            If faulty files are found while validating, opens the post validation frame (where the user must discard those files or cancel the selection), otherwise skips this step and closes the popup 
            '''
            popup.setFaultyFiles(faultyFiles)
            if len(faultyFiles) <= 0:
                popup.proceed() # Close Popup if no faulty files are found
            else:
                popup.showPostValidationFrame()

        def validateFile(index : int = 0, faultyFiles : list[str] = []):
            '''
            Helper Function (recursive) to validate files and update progress bar, without validation blocking the thread
            '''
            if index >= len(files):
                postValidation(faultyFiles)   # Once all files have been processed invoke callback
                return

            file = files[index]
            popup.updateProgress(index + 1, file)
            if not validate_csv(file):
                faultyFiles.append(file)
            # Since the verification operation might take some time, we will use an asynchronous recursion using tkinters after-method to allow the UI to update
            self.after(100, lambda: validateFile(index +1, faultyFiles))
        
        validateFile()

    def loadLabelFile(self):
        ''' 
        Button Callback for the import of label files
        Opens a File Selection Window of the OS and allows the user to select a single label file (.lbl).
        '''
        labelFile = selectFilesFromOS("Label-Datei auswählen", [("Label-Datei", "*.lbl")], True)
        if not labelFile or len(labelFile) <= 0:
            return

        labels = pd.read_csv(labelFile[0])
        mapping = dict(zip(labels["File"].apply(os.path.basename),labels["Label"]))
        self.trainFiles["Label"] = self.trainFiles["File"].apply(os.path.basename).map(mapping).fillna("")
        if isinstance(self.currentFrame, TrainingFrame):
            self.currentFrame.updateTables()
    
    def saveLabelFile(self):
        '''
        Button Callback for the export of label files.
        Opens a File Saving Window of the OS and allows the user to save is labels to a label file (.lbl).
        '''
        data = pd.DataFrame(self.trainFiles.copy())
        data["File"] = data["File"].apply(lambda x: os.path.basename(str(x)))
        saveLabelFileOS(data)

    def onPopupClose(self, isSuccess : bool, destination : pd.DataFrame, files : list[str], faultyFiles : list[str]):
        '''
        Callback which is called once the ValidationPopup is closed.
        If the popup closes with a success (all files validated), 
        stores the selected files in the pd.DataFrames trainData or testData depending on the context that the file selection was commenced 
        and initiates a refresh of the editableTables displaying this dataframes contents
        '''
        files = [f for f in files if f not in faultyFiles]
        # Remove stored files in the destination dataframe regardless of the success, to allow the user to clear selected files by closing the file selection window without selecting a file
        destination.drop(destination.index, inplace=True)

        if isSuccess:
            # Fill destination dataframe with selected file paths and clear labels (if column exists)
            destination["File"] = files
            if "Label" in destination.columns:
                destination["Label"] = [""] * len(files)
            if len(files) > 0:
                self.dataDir = Path(files[0]).parent

        # Update the editable tables to display the selected files
        if isinstance(self.currentFrame, TrainingFrame) or isinstance(self.currentFrame, PredictionFrame):
            self.currentFrame.updateTables()

    def mapConfig(self) -> dict:
        '''
        Maps the configurations selected in the TrainingFrame to a dictionary that is compatible with the backend_adapters set_config function
        '''
        config = {}
        models = []
        if bool(self.options["model_use_randomforest"].get()): models.append("randomforest")
        if bool(self.options["model_use_logreg"].get()): models.append("logreg") 
        if bool(self.options["model_use_gradientboosting"].get()): models.append("gradientboosting") 
        
        useFeatureTools = bool(self.options["features_use_featuretools"].get())
        useTsFresh = bool(self.options["features_use_tsfresh"].get())

        featureSets = "both"
        if useFeatureTools and not useTsFresh:
            featureSets = "featuretools"
        elif useTsFresh and not useFeatureTools:
            featureSets = "tsfresh"

        config["models"] = models
        config["feature_set"] = featureSets
        for k, v in self.options.items():
            config[k] = v.get()
        return config

    def startTraining(self):
        ''' Starts the training process if preconditions are met. Opens a ProgressPopup to display training progress. If preconditions are not met, displays a warning message '''
        proceed, reason = self.canStartTraining()
        if proceed:
            set_config(self.mapConfig())
            popup = ProgressPopup(self, self.styleConfig, "Training")
            # Starts the training helper function in a second thread to avoid blocking the main thread. This allows the GUI (mostly the ProgressPopup) to update while training
            threading.Thread(
                target=self.training,
                args=(popup.updateProgress,popup,),
                daemon=True
            ).start()
            return
        messagebox.showwarning("Unvollständiges Setup", reason)

    def training(self, callback, popup : ProgressPopup):
        '''
        Helper function which starts the training by calling backend_adapters train function. 
        Once training is completed, closes the popup and either displays an error, or loads model accuracy data and opens the ModelFrame 
        '''
        success, out = train(self.dataDir, self.trainFiles, self.styleConfig["paths"]["artifacts"], progress_callback=callback)
        if not success:
            popup.close()
            messagebox.showerror("Kritischer Fehler", "Training fehlgeschlagen" if out is None else out)
            return
        
        self.modelAccuracy = loadCsvAsDataFrame(self.styleConfig["paths"]["accuracyData"])
        print(self.modelAccuracy)
        self.openModelFrame()
        
    def startPrediction(self):
        ''' Starts the prediction process. Opens a ProgressPopup to display prediction progress.'''
        popup = ProgressPopup(self, self.styleConfig, "Vorhersagen")

        # Starts the prediction helper function in a second thread to avoid blocking the main thread. This allows the GUI (mostly the ProgressPopup) to update while training
        threading.Thread(
            target=self.prediction,
            args=(popup.updateProgress,),
            daemon=True
        ).start()

    def prediction(self, callback):
        '''
        Helper function which starts the prediction by calling backend_adapters predict function. 
        Once prediction is completed, closes the popup and either displays an error, or loads the prediction results into the testFiles DataFrame and updates the EditableTable displaying this data
        '''
        success, out, result = predict(self.dataDir, self.testFiles.copy(), self.styleConfig["paths"]["artifacts"], progress_callback=callback)
        
        if not success:
            messagebox.showerror("Kritischer Fehler", "Vorhersage fehlgeschlagen" if out is None else out)
            return

        # Adds predicted labels to testFiles dataframe for each model
        for model, predictions in result.items():
            mapping = {p["recording"]: p["ist"] for p in predictions}

            self.testFiles.loc[:, model] = (
                self.testFiles["File"]
                .apply(os.path.basename)
                .map(mapping)
            )

        if isinstance(self.currentFrame, PredictionFrame):
            self.currentFrame.testFiles = self.testFiles
            self.currentFrame.updateTables()

    def openTrainingFrame(self):
        ''' Replaces the current frame with the TrainingFrame '''
        trainingFrame = TrainingFrame(self, self.styleConfig, self.options, self.selectTrainFiles, self.startTraining, self.loadLabelFile, self.saveLabelFile, self.trainFiles)
        self.openFrame(trainingFrame)

    def openModelFrame(self):
        ''' Replaces the current frame with the ModelFrame '''
        self.stepProgress.next()
        modelFrame = ModelFrame(self, self.styleConfig, self.openPredictionFrame, modelAccuracyData=self.modelAccuracy, images=getPlotPaths(self.styleConfig["paths"]["plots"]))
        self.openFrame(modelFrame)

    def openPredictionFrame(self):
        ''' Replaces the current frame with the PredictionFrame '''
        predictionFrame = PredictionFrame(self, self.styleConfig, self.testFiles, self.selectTestFiles, self.startPrediction)
        self.stepProgress.next()
        self.openFrame(predictionFrame)

    def openFrame(self, frame):
        ''' Replaces the current frame with given Frame. Destroying the old frame in the process '''
        if self.currentFrame is not None:
            self.currentFrame.destroy()
        self.currentFrame = frame
        self.currentFrame.pack(expand=True, fill="both")

    def canStartTraining(self) -> tuple[bool, str]:
        ''' 
        Determines wheter or not all preconditions to start training are met. 
        Selection of at least one training file including labeling it, as well as selecting at least one model and feature algorithm are required.
        Furthermore, no input field is allowed to be empty or contain just 0
        '''
        trainFilesEmpty = self.trainFiles.empty
        if trainFilesEmpty:
            return False, "Es muss mindestens eine Trainingsdatei ausgewählt werden."

        trainFilesValid = (
            self.trainFiles["File"].astype(str).str.strip().ne("") &
            self.trainFiles["Label"].astype(str).str.strip().ne("")
        ).all()

        if not trainFilesValid:
            return False, "Allen Trainingsdateien muss ein Label zugewiesen sein."

        if not bool(self.options["model_use_randomforest"].get()) and not bool(self.options["model_use_logreg"].get()) and not bool(self.options["model_use_gradientboosting"].get()):
            return False, "Es muss mindestens einen Modell ausgewählt sein."
        
        if not bool(self.options["features_use_featuretools"].get()) and not bool(self.options["features_use_tsfresh"].get()):
            return False, "Es muss mindestens ein Feature-Algorithmus ausgewählt sein."

        windowSec = int(self.options["window_sec"].get()) if not self.options["window_sec"].get() == "" else 0
        stepSec = int(self.options["step_sec"].get()) if not self.options["step_sec"].get() == "" else 0
        minPoints = int(self.options["min_points"].get()) if not self.options["min_points"].get() == "" else 0
        maxPoints = int(self.options["max_points"].get()) if not self.options["max_points"].get() == "" else 0
        cvSplits = int(self.options["cv_splits"].get()) if not self.options["cv_splits"].get() == "" else 0

        if windowSec < 1 or stepSec < 1 or minPoints < 1 or maxPoints < 1:
            return False, "Window-Parameter müssen größer 0 sein."
        
        if cvSplits < 1:
            return False, "Anzahl Cross-Validation-Folds muss größer 0 sein."

        return True, ""
    
def getPlotPaths(path : str) -> list[str]:
    ''' Returns a List of file paths to plot images '''
    return [str(p) for p in Path(path).rglob("*.png")]