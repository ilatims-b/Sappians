import pickle
import os
from keras._tf_keras import keras
import numpy as np
from pdf2image import convert_from_path
import cv2
import sys

DEBUG = True
TEST_FILE = ""

SAVE_PATH_MODEL = f'{os.path.abspath(os.path.dirname(__file__))}/model_save/tf_model.keras'
SAVE_PATH_DICT = f'{os.path.abspath(os.path.dirname(__file__))}/model_save/tf_labels.pkl'


def debug(text):
    if DEBUG: print(text)

def load_model(model_path, dict_path):

    with open(dict_path, 'rb') as f:
        labels = pickle.load(f)

    model = keras.models.load_model(model_path)

    return model, labels

def run_classify(filename, model, labels):

    image = convert_from_path(filename,first_page=1,last_page=1,dpi=300)
    image = np.array(image[0])
    image_resized = cv2.resize(image,(512,512))
    image_array = keras.preprocessing.image.img_to_array(image_resized) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Predict the class
    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions)
    debug(predictions)
    debug(labels)

    predicted_label = labels[predicted_class]

    debug(f"Predicted Class Index: {predicted_class}")
    debug(f"Predicted Label: {predicted_label}")

    return predicted_label


def analyze_doc(filename):

    model, labels = load_model(SAVE_PATH_MODEL, SAVE_PATH_DICT)
    predicted_label = run_classify(filename, model, labels)
    return predicted_label

if __name__=="__main__":
    try:
        filename = sys.argv[1]
    except:
        filename = TEST_FILE
    
    print(analyze_doc(filename))