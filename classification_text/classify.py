from .img_processing.image_processing import *
import joblib
import os
import json
# import xgboost as xgb
# from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import classification_report
# from sklearn.model_selection import train_test_split
from pdf2image import convert_from_path

DEBUG=False

def debug(text):
    if DEBUG:
        print(text)

def read_model():
        
    try:
        # Load the saved model
        loaded_model = joblib.load(f'{os.path.abspath(os.path.dirname(__file__))}/model_save/xgb_model.sav')
        # loaded_model = xgb.XGBClassifier()
        # loaded_model.load_model(f'{os.path.abspath(os.path.dirname(__file__))}/model_save/xgb_model.bin')  # load model data
        # loaded_model = xgb.Booster()
        # loaded_model.load_model(f'{os.path.abspath(os.path.dirname(__file__))}/model_save/xgb_model.bin')

        # Load the saved vectorizer
        loaded_vectorizer = joblib.load(f'{os.path.abspath(os.path.dirname(__file__))}/model_save/tfidf_vectorizer.sav')
        # loaded_vectorizer = TfidfVectorizer(max_features=10000)

        # Load labels
        with open(f'{os.path.abspath(os.path.dirname(__file__))}/model_save/labels.sav') as f:
            labels = json.load(f)

    except:
        raise(FileNotFoundError)

    print("Model, vectorizer and labels loaded successfully!")

    return loaded_model, loaded_vectorizer, labels

def predict(model, vectorizer, filename, labels):
    #  Predict the category using the trained XGBoost model
    image = convert_from_path(filename,first_page=1,last_page=1,dpi=300)
    image[0].save(f"{os.path.abspath(os.path.dirname(__file__))}/processing/image.png", "PNG")
    processed_text=clean_text(extract_text_from_image(f"{os.path.abspath(os.path.dirname(__file__))}/processing/image.png"))
    os.remove(f"{os.path.abspath(os.path.dirname(__file__))}/processing/image.png")
    document_tfidf = vectorizer.transform([processed_text])
    debug(vectorizer)
    #retrieving the lost label names

    label_encoder = LabelEncoder()
    label_encoder.fit(list(set(labels)))

    # Retrieve the mapping
    label_map = {index: label for index, label in enumerate(label_encoder.classes_)}
    predicted_class_index = model.predict(document_tfidf)[0]
    debug(label_map)
    debug(predicted_class_index)

    predicted_label = label_map[predicted_class_index]
    
    return predicted_label, processed_text
    

def analyze_doc(filename):
    print("Starting model")
    model, vectorizer, labels = read_model()

    print("Reading file")
    return predict(model, vectorizer, filename, labels)

