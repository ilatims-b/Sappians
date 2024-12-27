# Team Name: Sappians
Appian Challenge '24

This is our submission to the Appian AI Application challenge conducted by Shastraa. 
We are working on Problem Statement 1



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