from flask import Flask
from flask import request
from flask import jsonify
from series2graph import *
import pandas as pd
from s2g_wrapper import *
from pymongo import MongoClient
import json
import numpy as np


app = Flask(__name__, static_url_path='', static_folder='./')


@app.route('/graph')
def static_file():
    print('in static file')
    return app.send_static_file('./graph.html')


@app.route('/s2g')
def s2g():
    args = request.args
    qlen = request.args.get("qlen", 100, type=int)
    plen = request.args.get("plen", 75, type=int)   
    limit = request.args.get("limit", 5000, type=int)
    delta = request.args.get("delta", 0, type=int)
    print("qlen=",qlen, " plen=",plen , " limit = ",limit, " delta=",delta);
    
    client = MongoClient()
    db = client.hack
    collection = db.materna
    df = pd.DataFrame(list(collection.find().limit(limit).skip(delta)))
   

    df = df.drop(columns=['_id','date'])
    s2gw = S2gWrapper(df, qlen, plen)
    scores = s2gw.calc('y')
    lscores = scores.tolist()
    l = len(scores)
    print("len scores= ",l, " len df = ",df['y'].size)
    df2 = df.iloc[:l]
    print("after truncate len df2 = ",df2['y'].size)

    j1 = df2.to_json(orient="records")

    df2['scores'] = scores
# create anom values and scores
    threshold = 0.6
    anom_val = [];
    anom_score = [];
    for i in range(len(scores)):
        if(scores[i] > threshold):
            anom_val.append({ 'x': int(df2['x'].values[i]), 'y': df2['y'].values[i]})
            anom_score.append({'x':  int( df2['x'].values[i]), 'y':df2['scores'].values[i]})
        else:
                 anom_val.append({ 'x': int(df2['x'].values[i]), 'y': None})     
                 anom_score.append({'x':  int( df2['x'].values[i]), 'y': None})
         

    df2 = df2.drop(columns=['y'])
    df2 = df2.rename(columns= {'scores' : 'y'})
    j2 = df2.to_json(orient="records")
    p1 = json.loads(j1)
    p2 = json.loads(j2)
    d = {'values' : p1, 'scores' : p2, 'anom_val' : anom_val, 'anom_score': anom_score}
    return(d)


#def calc(df,qlen,plen)


@app.route('/data')
def data():
    args = request.args
    limit = request.args.get("limit", 100, type=int)
    skip = request.args.get("skip", 100, type=int)

    client = MongoClient()
    db = client.hack
    collection = db.cpu
    df = pd.DataFrame(list(collection.find().limit(500)))
    #df = df.iloc[8000:]
    df2 = df.drop(columns=['_id','date'])
    json =df2.to_json(orient="records")
    print("/data  done ...")
    return(json)


    