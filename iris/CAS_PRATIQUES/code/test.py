import streamlit as st
import tensorflow as tf
import cv2
from PIL import Image
import numpy as np
from sklearn import preprocessing
import json

# image = cv2.imread(r"C:\Users\utilisateur\Pictures\pour modele\3gauche.bmp")
# print(image)
# image = np.asarray(image).astype('float32')
# image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
# print(image.shape)
# model1 = tf.keras.models.load_model(r'C:\VSCODE\iris\modeles\oeil_gauche_droite_model')
# encoder = preprocessing.LabelEncoder()
# encoder.classes_ = np.load(r'C:\VSCODE\iris\modeles\labelsencoder\classes.npy')
# prediction = model1.predict(image)
# prediction = np.argmax(prediction, keepdims = True)
# prediction = np.ravel(prediction)
# print(prediction)
# # prediction = encoder.inverse_transform(prediction)
# print(encoder.inverse_transform(prediction))

with open('C:\VSCODE\iris\CAS_PRATIQUES\employees_info.json', 'r', encoding = "utf-8") as f:
    data = json.load(f)

print(data["1"])
