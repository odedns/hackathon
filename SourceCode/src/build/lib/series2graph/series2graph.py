# Authors: Paul Boniol, Themis Palpanas
# Date: 08/07/2020
# copyright retained by the authors
# algorithms protected by patent application FR2005261
# code provided as is, and can be used only for research purposes
#
# Reference using:
#
# P. Boniol and T. Palpanas, Series2Graph: Graph-based Subsequence Anomaly Detection in Time Series, PVLDB (2020)
#
# P. Boniol and T. Palpanas and M. Meftah and E. Remy, GraphAn: Graph-based Subsequence Anomaly Detection, demo PVLDB (2020)
#

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from scipy.stats import gaussian_kde
from scipy.signal import argrelextrema
from sklearn.decomposition import PCA
from scipy.signal import argrelextrema
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout


from .series2graph_tools import *


class Series2Graph():

	def __init__(self,pattern_length,latent=None,rate=30):
		self.pattern_length = pattern_length
		self.rate = rate
		if latent is not None:
			self.latent = latent
		else:
			self.latent = self.pattern_length//3

		self.graph = {}

	def fit(self,ts):
		min_df_r = min(ts[0].values)
		max_df_r = max(ts[0].values)

		df_ref = []
		for i in np.arange(min_df_r, max_df_r,(max_df_r-min_df_r)/100):
			tmp = []
			T = [i]*self.pattern_length
			for j in range(self.pattern_length - self.latent):
				tmp.append(sum(x for x in T[j:j+self.latent]))
			df_ref.append(tmp)
		df_ref = pd.DataFrame(df_ref)
		
		phase_space_1 = build_phase_space(ts[0].values,self.latent,self.pattern_length)

		pca_1 = PCA(n_components=3)
		pca_1.fit(phase_space_1)
		reduced = pd.DataFrame(pca_1.transform(phase_space_1),columns=[str(i) for i in range(3)])
		reduced_ref = pd.DataFrame(pca_1.transform(df_ref),columns=[str(i) for i in range(3)])

		v_1 = reduced_ref.values[0]

		R = get_rotation_matrix(v_1,[0.0, 0.0, 1.0])
		A = np.dot(R,reduced.T)
		A_ref = np.dot(R,reduced_ref.T)
		A = pd.DataFrame(A.T,columns=['0','1','2'])
		A_ref = pd.DataFrame(A_ref.T,columns=['0','1','2'])
	
		res_point,res_dist = get_intersection_from_radius(A,'0','1',rate=self.rate)
		nodes_set,node_weight = nodes_extraction(A,'0','1',res_point,res_dist,self.rate)
		list_edge,node_evo,time_evo = edges_extraction(A,'0','1',nodes_set,rate=self.rate)
	
		G = nx.DiGraph(list_edge)
	
		dict_edge = {}
		for edge in list_edge:
			if str(edge) not in dict_edge.keys():
				dict_edge[str(edge)] = list_edge.count(edge)

	 
		result = {
			"Graph": G,
			"list_edge": list_edge,
			"edge_in_time": time_evo,
			"pca_proj": pca_1,
			"rotation_matrix": R,
			"edge_weigth": dict_edge,
			"proj_A":A,
			"node_weigth":node_weight,
			"node_set": nodes_set,
			}

		self.graph = result

	def score(self,query_length):
		all_score = []
		degree = nx.degree(self.graph["Graph"])
		for i in range(0,len(self.graph["edge_in_time"])-int(query_length)):
			P_edge = self.graph["list_edge"][self.graph["edge_in_time"][i]:self.graph["edge_in_time"][i+int(query_length)]]
			score,len_score = score_P_degree(self.graph["edge_weigth"],P_edge,degree)
			all_score.append(score)

		all_score = [-score for score in all_score]
		all_score = np.array(all_score)
		all_score = (all_score - min(all_score))/(max(all_score) - min(all_score))
		#all_score = running_mean(all_score,self.pattern_length)

		self.all_score = all_score

	#to optimize
	def plot_graph(self):
		edge_size = []
		for edge in self.graph["Graph"].edges():
			edge_size.append(self.graph["list_edge"].count([edge[0],edge[1]]))
		edge_size_b = [float(1+(e - min(edge_size)))/float(1+max(edge_size) - min(edge_size)) for e in edge_size]
		edge_size = [min(e*50,30) for e in edge_size_b]
		pos = nx.nx_agraph.graphviz_layout(nx.Graph(self.graph["list_edge"]),prog="fdp")
		
		plt.figure(figsize=(30,30))
		dict_node = []
		for node in self.graph["Graph"].nodes():
			dict_node.append(1000*self.graph["node_weigth"][int(node.split("_")[0])][int(node.split("_")[1])])
		nx.draw(self.graph["Graph"],pos, node_size=dict_node,with_labels=True,width=edge_size)

