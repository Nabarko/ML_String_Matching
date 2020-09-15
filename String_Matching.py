# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 15:21:28 2020

@author: eroynab
"""

from flask import Flask, make_response, request, jsonify
import config as cfg
import requests
import Service_Functions as sf
from flasgger import Swagger

app = Flask(__name__)
Swagger(app)

################# MAIN FUNCTION TO RUN THE WHOLE PROGRAM ###########
if __name__=='__main__':
    """
    DocString: Main function for the whole program operation.
    """
    pass