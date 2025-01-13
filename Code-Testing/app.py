# app.py
from flask import Flask, jsonify

app = Flask(__name__)   #creates an instance of the Flask class. name is a variable that helps identify the root path of the app


@app.route('/')
def home():
    return jsonify(message="Hello level 400 FET, Quality Assurance!")   #in flask, the message is stored in a dictionary. Jasonify convets it to a Json file


if __name__ == '__main__':
    app.run(debug=True)