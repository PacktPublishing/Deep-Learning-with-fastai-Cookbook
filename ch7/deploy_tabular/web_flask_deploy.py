# example of using Flask to deploy a fastai deep learning model trained on a tabular dataset
import json
import os
import urllib.request
import numpy as np
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
from flask import Flask, render_template, request
#from fastbook import *
from fastai.tabular.all import *
#from fastai.vision import *

scoring_columns = ["age","workclass","fnlwgt","education","education-num","marital-status","occupation","relationship","race","sex","capital-gain","capital-loss","hours-per-week","native-country"]

# build the path for the trained model
path = Path(os.getcwd())
full_path = os.path.join(path,'adult_sample_model.pkl')
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
    # load the scoring parameter values into a dictionary indexed by the column names expected by the model
    score_values_dict = {}
    # bring the URL argument values into a Python dictionary
    for column in scoring_columns:
        # use input from home.html for scoring
        score_values_dict[column] = request.args.get(column)
    for value in score_values_dict:
        print("value for "+value+" is: "+str(score_values_dict[value]))
    # create and load scoring parameters dataframe (containing the scoring parameters)that will be fed into the pipelines
    score_df = pd.DataFrame(columns=scoring_columns)
    # df = df.astype({"a": int, "b": complex})
    print("score_df before load is "+str(score_df))
    for col in scoring_columns:
        score_df.at[0,col] = score_values_dict[col]
    # ensure columns have the correc types    
    score_df = score_df.astype({"age":np.int64,"fnlwgt":np.int64,"education-num":np.float64,"capital-gain":np.int64,"capital-loss":np.int64,"hours-per-week":np.int64})
    # print details about scoring parameters
    print("score_df: ",score_df)
    print("score_df.dtypes: ",score_df.dtypes)
    print("score_df.iloc[0]",score_df.iloc[0])
    print("shape of score_df.iloc[0] is: ",score_df.iloc[0].shape)
    pred_class,pred_idx,outputs = learner.predict(score_df.iloc[0])
    for col in scoring_columns:
        print("pred_class "+str(col)+" is: "+str(pred_class[col]))
    print("pred_idx is: "+str(pred_idx))
    print("outputs is: "+str(outputs))
    # get a result string from the value of the model's prediction
    if outputs[0] >= outputs[1]:
        predict_string = "Prediction is: individual has income less than 50k"
    else:
        predict_string = "Prediction is: individual has income greater than 50k"
    # build parameter to pass on to show-prediction.html
    prediction = {'prediction_key':predict_string}
    # render the page that will show the prediction
    return(render_template('show-prediction.html',prediction=prediction))
    
    
    
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')