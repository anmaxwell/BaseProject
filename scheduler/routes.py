import os
from flask import render_template, flash
from scheduler import app

@app.route('/') #Page to display tiles
def home():

    return render_template('home.html')