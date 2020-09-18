# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 15:21:28 2020

@author: eroynab
"""

from flask import Flask, make_response, request, jsonify
from flasgger import Swagger
import service_functions as sf
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import collections 

app = Flask(__name__)
app.debug = True
Swagger(app)

###################################### ROUTES ##########################################
########### ROUTE 1  #####################
@app.route("/api/v0.1/format/match",methods=['POST'])
@app.route("/api/v0.1/train/format/match",methods=['POST'])
def string_match():
    """
    This is the Forecasting Data Fetcher & Cleaner API
    Call this api & get "kube-system" container names
    ---
    tags:
      - Forecasting Data Fecther & Cleaner API
    responses:
        404:
            description: Error in the Getting the Container Names from Prometheus!
        200:
            description: List of data for "kube-system" Container Names.
    """
    try:
        if request.method == "POST":
            body = request.json
            data_set = json_parser(body)
            #vectorizer = pickle.load(open("vect.sav","rb"))
            vectorizer = TfidfVectorizer(analyzer=sf.ngrams)
            dt_model = pickle.load(open("decision_tree.sav","rb"))
            lev_score_list = [round(lev_score(list(data_set["source_format"])[i],list(data_set["target_format"])[i]),2)*100 for i in range(0,len(data_set["source_format"]))]
            data_set["lev_score"] = lev_score_list
            data_set["fuzz_lev_score"] = sf.calculate_fuzzy_score(list(data_set["source_format"]),list(data_set["target_format"]))
            data_set["fuzz_abb_score"] = sf.calculate_fuzzy_abbv_score(list(data_set["source_format"]),list(data_set["target_format"]))
            vect_score = [round(sf.calculate_cosine_similarity(list(data_set["source_format"])[i],list(data_set["target_format"])[i],vectorizer),2)*100 for i in range(0,len(data_set["source_format"]))]
            data_set["vect_score"] = vect_score
            match_score = dt_model.predict_proba(data_set[['lev_score', 'fuzz_lev_score','fuzz_abb_score', 'vect_score']])
            confidence = [score[1] for score in match_score]
            print(confidence)                        
            final_response = json_response(body,confidence)
            return make_response(jsonify(final_response), 200, headers)
    except Exception() as e:
        return make_response(jsonify({"Response":"Error in Prediction of /container_names {0}".format(e)}), 404, headers)
        #print("Error in Prediction of /container_names",e)
    
# ########### ROUTE 2 train/match/ #####################
@app.route("/api/v0.1/train/format/learn",methods=['POST'])
def route_container_daily_values():
    """
    This is the Forecasting Data Fetcher & Cleaner API
    Call this api & get "kube-system" container names
    ---
    tags:
      - Forecasting Data Fecther & Cleaner API
    responses:
        404:
            description: Error in the Getting the Container Names from Prometheus!
        200:
            description: List of data for "kube-system" Container Names.
    """
    try:
        if request.method == "POST":
            body = request.json
            #print(json_parser_train(body))
            #print(body)
            data_set = json_parser_train(body)
            vectorizer = TfidfVectorizer(analyzer=sf.ngrams)
            dt_model = pickle.load(open("decision_tree.sav","rb"))
            lev_score_list = [round(lev_score(list(data_set["source_format"])[i],list(data_set["target_format"])[i]),2)*100 for i in range(0,len(data_set["source_format"]))]
            data_set["lev_score"] = lev_score_list
            data_set["fuzz_lev_score"] = sf.calculate_fuzzy_score(list(data_set["source_format"]),list(data_set["target_format"]))
            data_set["fuzz_abb_score"] = sf.calculate_fuzzy_abbv_score(list(data_set["source_format"]),list(data_set["target_format"]))
            vect_score = [round(sf.calculate_cosine_similarity(list(data_set["source_format"])[i],list(data_set["target_format"])[i],vectorizer),2)*100 for i in range(0,len(data_set["source_format"]))]
            data_set["vect_score"] = vect_score
            prev_dataset = pd.read_csv("trainer.csv")
            if collections.Counter(list(prev_dataset.columns)) == collections.Counter(list(data_set.columns)): 
                final_train_data = append_data(prev_dataset,data_set)
                X = final_train_data[['lev_score', 'fuzz_lev_score','fuzz_abb_score', 'vect_score']]
                y = final_train_data["match"] 
                dt_model.fit(X,y)
                print("Model is learnt on the new data")
                pickle.dump(dt_model,open("decision_tree.sav","wb"))
                return make_response(jsonify({"sourceformatName":body["source"]["formatName"],"targetformatName":body["target"]["formatName"],"Message" : "Learned the mappings"}), 200, headers)
            else:
                return make_response(jsonify({"Message" : "Issue in Learning the Mappings || Difference in the learning columns"}), 505, headers)
    except Exception() as e:
        return make_response(jsonify({"Response":"Error in Prediction of /container_names {0}".format(e)}), 404, headers)
        #print("Error in Prediction of /container_names",e)


        
#### Json Parsers ####
def json_parser(json_body):
    try:
        source_data = json_body["source"]["formatFields"]
        target_data = json_body["target"]["formatFields"]
        cleaned_source_data = [sf.clean_strings(data) for data in source_data]
        cleaned_target_data = [sf.clean_strings(data) for data in target_data]
        data_frame = pd.DataFrame({"source_format":cleaned_source_data,"target_format":cleaned_target_data})
        return data_frame
    except Exception as e:
        print("Error Occured while parsing the json body ::",e)

#### Response former ####
def json_response(json_body,confidence_list):
    try:
        source_name = json_body["source"]["formatName"]
        target_name = json_body["target"]["formatName"]
        source_data = json_body["source"]["formatFields"]
        target_data = json_body["target"]["formatFields"]
        overall_confidence = round(sum(confidence_list)/len(confidence_list),2)*100
        mapping_dict_list= [{"sourceField":source_data[i],"targetField":target_data[i],"confidence":round(confidence_list[i],2)*100} for i in range(0,len(source_data))]
        response_dict = {"sourceformatName":source_name,"targetformatName":target_name,"overallConfidence":overall_confidence,"mappings":mapping_dict_list}
        return response_dict
    except Exception as e:
        print("Error in forming the json response body :: ",e)

#### Json Parsers ####
def json_parser_train(json_body):
    try:
        source_data = [map_list["sourceField"] for map_list in json_body["mappings"]]
        target_data = [map_list["targetField"] for map_list in json_body["mappings"]]
        cleaned_source_data = [sf.clean_strings(data) for data in source_data]
        cleaned_target_data = [sf.clean_strings(data) for data in target_data]
        default_match = [1 for i in range(0,len(cleaned_source_data))]
        data_frame = pd.DataFrame({"source_format":cleaned_source_data,"target_format":cleaned_target_data,"match":default_match})
        return data_frame
    except Exception as e:
        print("Error Occured while parsing the json body ::",e)

#### While training, we can append with old data for enhanced training ####
def append_data(prev_dataset,current_data):
    try:
        if (prev_dataset.empty == False) and (current_data.empty == False):
            dataset = prev_dataset.append(current_data)
            return dataset
    except Exception as e:
        print("Error Occured while appending the dataframes :: ",e)

#### Find the levenstien score #####
def lev_score(string_1,string_2):
    try:
        score = sf.levenshtein_rate(string_1,string_2)
        return score
    except Exception as e:
        print("Error Occured in find the Lev_Score ::: ",e)
                

################# MAIN FUNCTION TO RUN THE WHOLE PROGRAM ###########
if __name__=='__main__':
    """
    DocString: Main function for the whole program operation.
    """
    headers = {"Content-Type": "application/json"}
    app.run(debug=False,host= "0.0.0.0",port=5005,threaded=True)