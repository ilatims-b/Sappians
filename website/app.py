from flask import Flask, render_template, request
import pandas as pd
import os

app = Flask(__name__)

# Load the data
df = pd.read_csv(f"{os.path.abspath(os.path.dirname(__file__))}/../database.csv", delimiter="|")

@app.route('/')
def home():
    global df 
    df = pd.read_csv(f"{os.path.abspath(os.path.dirname(__file__))}/../database.csv", delimiter="|")
    # Pass unique person names to the frontend
    people = df['Person'].unique()
    return render_template('index.html', people=people)

@app.route('/get-data', methods=['GET'])
def get_data():
    # Get the selected person from the request
    person = request.args.get('person')
    # Filter data for the selected person
    filtered_data = df[df['Person'] == person]
    return filtered_data.to_dict(orient='records')

if __name__ == '__main__':
    app.run(debug=True)
