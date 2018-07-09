""" File : hello_ekimetrics.py 
"""

from flask import Flask 

app = Flask(__name__)

@app.route('users/<string:username>')
def hello_ekimetrics(username=None):
    return("Welcome {} to Ekimetrics!".format(username))

