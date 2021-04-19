from flask import Flask
from flask import request
from flask import jsonify
from series2graph import *
import pandas as pd
from s2g_wrapper import *

app = Flask(__name__)




@app.route('/s2g')
def s2g():
    args = request.args
    filename = args['filename']
    index = int(args['index'])
    qlen = int(args['qlen'])
    plen = int(args['plen'])
    s = "filename=" + filename + " index=" + str(index) + " plen=" + str(plen) + " qlen=" + str(qlen)
    print(s)
    path = "../../DATA/"+ filename;
    ucols = []
    ucols.append(index)
    df = pd.read_csv(path,header=None,usecols=ucols)
    print(df)
    s2gw = S2gWrapper(df,qlen,plen)
    scores = s2gw.calc(index)
    lscores = scores.tolist()
    values = df[index].values.tolist()
    result = { 'values': values, 'scores': lscores}
    return(jsonify(result))
