#!/usr/bin/env python
# coding: utf-8

# # Series2Graph Demo
# 
# This notebook describe and display all the step that series2graph preforms in order to detect abnormal subsequences in a time series.

# In[ ]:


import matplotlib.pyplot as plt

from series2graph import series2graph as sg, series2graph_tools as sg_tools
import  numpy as np
import pandas as pd

# ## Demo on an ECG time series
# 
# The full process of Time2graph applied on a time series corresponding to an electro-cardiogram from the ATM physiobank dataset (record 803).

# In[Setup log function]
import sys
from datetime import datetime
import logging

log_file = "logfile.txt"
def write_log(*args): 
   now = datetime.now()
   current_time = now.strftime("%H:%M:%S")

   #logger = get_logger()
   line = ' '.join([str(a) for a in args])
   line = f'{current_time} ' + line 
   f_log = open(log_file,'a', encoding='utf-8')
   f_log.write(line+'\n')
   f_log.close()
   print(line)
# In[Compute graph for one serie]:
def graph_one_serie(df, column_name):
   # ## Parameters setting
   
   pattern_length = 75
   query_length = 100
   
   
   
   # In[ ]:
   
   # ## Computing the Graph
   write_log("define model Series2Graph")
   s2g = sg.Series2Graph(pattern_length=pattern_length)
   write_log("fit the model")
   s2g.fit(df)
   
   
   # In[ ]:
   
   write_log("fit the model Graph Statistics:")
   print("Graph Statistics:")
   print("Number of nodes: {}".format(s2g.graph['Graph'].number_of_nodes()))
   print("Number of edges: {}".format(s2g.graph['Graph'].number_of_edges()))
   
   
   
   # In[ ]:
   
   # ### Visualization of the embedding space
   # write_log("plot graph projection A")
   # plt.figure(figsize=(10,10))
   # plt.plot(s2g.graph['proj_A']['0'],s2g.graph['proj_A']['1'])
   # plt.title("SProj(T,l,lambda)")
   
   
   
   # In[ ]:
   
   
   # ### Visualization of the graph
   write_log("Visualization of the graph:")
   # ==> !!!! s2g.plot_graph()
   
   
   
   # In[ ]:
   
   
   # ## Anomalies detection
   write_log("Anomalies detection")
   s2g.score(query_length)
   
   
   
   # In[ ]:
   
   
   # ### Visualization of the full time series
   # write_log("Visualization of the full time series:")
   # fig,ax = plt.subplots(2,1,figsize=(20,4))
   # ax[0].plot(df[0].values[0:len(s2g.all_score)])
   # ax[1].plot(s2g.all_score)
   # ax[0].set_xlim(0,len(s2g.all_score))
   # ax[1].set_xlim(0,len(s2g.all_score))
   
   # In[]
   threshold = 0.5
   anom_val = []
   anom_score = []
   for i in range(len(s2g.all_score)):
       if(s2g.all_score[i] > threshold):
           anom_val.append(df[0].values[i])
           anom_score.append(s2g.all_score[i])
       else: 
           anom_val.append(np.nan)
           anom_score.append(np.nan)
   len(anom_val)        
   
   # In[Plot anomalies]
   # fig, ax = plt.subplots(2,1,figsize=(20,4))
   # ax[0].plot(df[0].values[0:len(s2g.all_score)])
   # ax[0].plot(anom_val[0:len(anom_val)],"r")
   # ax[1].plot(anom_score)
   # ax[0].set_xlim(0,len(anom_score))
   # ax[1].set_xlim(0,len(anom_score))
   # ax[0].title.set_text(column_name)
   
   # In[ ]:
   
   
   # ### Visualization of a snippet
   # write_log("Visualization of a snippet")
   # fig,ax = plt.subplots(2,1,figsize=(20,4))
   # ax[0].title.set_text(column_name)
   # ax[0].plot(df[0].values[0:len(s2g.all_score)])
   # ax[0].plot(anom_val[0:len(anom_val)],"ro")
   # ax[1].plot(s2g.all_score)
   # ax[0].set_xlim(len(s2g.all_score)*.9,len(s2g.all_score))
   # ax[1].set_xlim(len(s2g.all_score)*.9,len(s2g.all_score))

   return anom_val

# In[Main Program]
# write_log("read_csv file /DATA/ATM_ECG_803.ts")
# df = pd.read_csv("../DATA/ATM_ECG_803.ts",header=None)[:100000]
# df = pd.read_csv('https://query1.finance.yahoo.com/v7/finance/download/BKHYY?period1=1201824000&period2=1997753600&interval=1d&events=history&includeAdjustedClose=true',)
# Stock='BKHYY'
# df = pd.read_csv('https://query1.finance.yahoo.com/v7/finance/download/BLMIF?period1=1199145600&period2=1619308800&interval=1d&events=history&includeAdjustedClose=true')
# Stock='BLMIF'
df = pd.read_csv('https://query1.finance.yahoo.com/v7/finance/download/JPM?period1=1199145600&period2=1619308800&interval=1d&events=history&includeAdjustedClose=true')
Stock='JP Morgan'
print("Time Series Statistics:")
print("Number of points: {}".format(len(df)))
print("Data frame: {}",df)
columns= ['Open','High','Low','Close','Adj Close','Volume']

# In[compute graph for each serie]
anomal_values=pd.DataFrame([])
for i in columns:
   tmp = pd.DataFrame(df[i])
   tmp.rename(columns = {i:0}, inplace = True)
   anomal_values[i]=graph_one_serie(tmp, i)
# In[Visualize all datasets]
# ### Visualization of a snippet
write_log("Visualization of a snippet")
fig,ax = plt.subplots(len(columns),1,figsize=(20,20))
for i in range(len(columns)):
   ax[i].title.set_text(f'{Stock} - {columns[i]}')
   ax[i].plot(df[columns[i]].values[0:len(anomal_values)])
   ax[i].plot(anomal_values[columns[i]],"ro")
   #ax[i].set_xlim(len(s2g.all_score)*.9,len(s2g.all_score))
plt.show()
# In[ ]:

write_log("Done!")
