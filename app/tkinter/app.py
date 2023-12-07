import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from fonctions import *
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import regularizers
from PIL import Image
import cv2
from keras.utils import to_categorical


class ImagePredictionApp:

    def __init__(self, root):

        self.root = root
        self.root.title("Application d'Authentification")
        # self.model1 = self.load_model(r'C:\VSCODE\iris\modeles\oeil_gauche_droite_model')
        self.modelgauche = self.load_model(r'C:\VSCODE\Projets\ecole\brief\classer_ses_iris\modeles\modelgauche.hdf5')
        self.modeldroite = self.load_model(r'C:\VSCODE\Projets\ecole\brief\classer_ses_iris\modeles\modeldroite.hdf5')

        self.select_button = tk.Button(root, text="Sélectionner une image", command=self.load_image)
        self.select_button.place(x = 140, y = 220)
        self.imagepath_label = tk.Label(root)
        self.imagepath_label.pack()
        self.imagedisp_label = tk.Label(root)
        self.imagedisp_label.pack()
        
        self.prediction_label = tk.Label(root, text="")
        self.prediction_label.pack()

    def load_model(self, path):

        model = tf.keras.models.load_model(path)
        model.trainable = False

        return model
    
    def load_image(self):

        file_path = filedialog.askopenfilename()

        if file_path:

            self.image = cv2.imread(file_path)
            self.image_prep=preprocess_img_without_norm(self.image)  # Redimensionnez l'image pour l'affichage
            self.imagepath_label.config(text=f"Image : {file_path}")
            self.photo = ImageTk.PhotoImage(Image.open(file_path))
            self.imagedisp_label.config(image=self.photo)

    def predict_image(self):

        if hasattr(self, 'image'):

            # image_array = np.array(self.image)
            # image_array = tf.image.resize(image_array, (224, 224))
            
            # Faites la prédiction en utilisant votre modèle
            prediction = self.model1.predict(np.array([self.image_prep]))  # Remplacez par la méthode de prédiction de votre modèle
            self.prediction_label.config(text=f"Prédiction du modèle : {prediction}")

        else:

            self.prediction_label.config(text="Aucune image sélectionnée")

if __name__ == "__main__":

    root = tk.Tk()
    root.geometry("400x500")
    app = ImagePredictionApp(root)
    root.mainloop()