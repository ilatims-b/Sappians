# Team Name: Sappians
Appian Challenge '24

This is our submission to the Appian AI Application challenge conducted by Shastraa. 
We are working on Problem Statement 1.

Classifying documents by text extraction using OCR-tesseract. Trained a XGBoost classifier on the vectorized text (TF-IDF vectorizer) from the document to classify it into document categories like bank statements, aadhar card, driving license, passport, etc. Also identifies the user it belongs to and creates an entry in the database to store the user documents in one place.

Alternatively tried training a basic CNN model for the classification task. Also tried Llama prompt-based classification. Text extraction and XGB classifier gave best results.

Example:

Running OCR and Classifying

Starting model

Model, vectorizer and labels loaded successfully!

Reading file

OCR and Classification Time: 3.42 seconds

Running LLM to extract data
LLM Data Extraction Time: 3.34 seconds

Data pulled
Doc type: bank_statements, data: {'person_name': 'mrs baddu,'}

Copying the file
File Operations Time: 0.00 seconds

Updating Database...

On a one-page bank-statement pdf of mrs baddu, scrapped from internet.



## How to run:
 - Follow the instructions in the _ocr_analysis_ README to download the required llm model.
 - Start flask server by running website/app.py
 - Delete the database.csv file (if required).
 - Run main\_script.py _input.pdf_ to analyze a pdf.
 - The entry should become visible in the web interface (after reloading). The file will be renamed (as per the key visible in the web\_interface) and saved in sorted\_files.

## Training the classification model:
 - Run classification/train_model. The dataset is present in classfication/training/train_data

### TO-DO:
 - Add various formats for documents in ocr\_analysis/json_formats.py and test out other data. (DONE!) 


#### Team Members:
- Smitali Bhandari ce24b119@smail.iitm.ac.in
- Atharva Moghe cs23b096@smail.iitm.ac.in
- Achintya Raghavan ee23b189@smail.iitm.ac.in
- Aditi Vaidya me23b248@smail.iitm.ac.in
