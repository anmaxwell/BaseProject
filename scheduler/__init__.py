from flask import Flask
import os

app = Flask(__name__)

#to prevent getting stuck in circular imports create this last
from scheduler import routes, models