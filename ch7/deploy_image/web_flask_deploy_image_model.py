# example of using Flask to deploy a fastai deep learning model trained on an image dataset
import json
import os
import urllib.request
import numpy as np
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
from flask import Flask, render_template, request
from fastai.vision.all import *

image_directory = "test_images"

# build the path for the trained model
path = Path(os.getcwd())
full_path = os.path.join(path,'fruits_360may3.pkl')
print("path is:",path)
print("full_path is: ",full_path)
# load the model
learner = load_learner(full_path)


app = Flask(__name__)


@app.route('/')
def home():   
    ''' render home.html - page that is served at localhost that allows user to enter model scoring parameters'''
    title_text = "fastai deployment"
    title = {'titlename':title_text}
    return render_template('home.html',title=title) 
    
@app.route('/show-prediction/')
def show_prediction():
    ''' 
    get the scoring parameters entered in home.html and render show-prediction.html
    '''
    # the scoring parameters are sent to this page as parameters on the URL link from home.html
    # load the scoring parameter
    image_file_name = request.args.get("file_name")
    # build the fully qualified file name
    full_path = os.path.join(path,image_directory,image_file_name)
    print("full_path is: ",full_path)
    img = PILImage.create(full_path)
    # apply the model to the image
    pred_class, ti1, ti2 = learner.predict(img)
    print("pred_class is: ",pred_class)
    predict_string = "Predicted object is: "+pred_class
    # build parameter to pass on to show-prediction.html
    prediction = {'prediction_key':predict_string}
    # render the page that will show the prediction
    return(render_template('show-prediction.html',prediction=prediction))
    
    
    
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')