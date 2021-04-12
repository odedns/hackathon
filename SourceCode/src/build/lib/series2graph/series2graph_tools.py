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
import math
from scipy.stats import norm
from scipy.stats import gaussian_kde
from scipy.signal import argrelextrema
from sklearn.decomposition import PCA
from scipy.signal import argrelextrema
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout


def build_phase_space(T,latent,m,rate=1):
	tmp_glob = []
	current_seq = [0]*m
	first = True
	for i in range(int((len(T) - m - latent)/rate)):
		tmp = []
		it_rate = i*rate
		if first:
			first = False
			for j in range(m - latent):
				tmp.append(sum(x for x in T[it_rate+j:it_rate+j+latent]))
			tmp_glob.append(tmp)
			current_seq = tmp
		else:
			tmp = current_seq[1:]
			tmp.append(sum(x for x in T[it_rate+m-latent:it_rate+m]))
			tmp_glob.append(tmp)
			current_seq = tmp
			
	return pd.DataFrame(tmp_glob)

def get_rotation_matrix(i_v, unit):
	curve_vec_1 = i_v
	curve_vec_2 = unit
	a,b = (curve_vec_1/ np.linalg.norm(curve_vec_1)).reshape(3), (curve_vec_2/ np.linalg.norm(curve_vec_2)).reshape(3)
	v = np.cross(a,b)
	c = np.dot(a,b)
	s = np.linalg.norm(v)
	I = np.identity(3)
	vXStr = '{} {} {}; {} {} {}; {} {} {}'.format(0, -v[2], v[1], v[2], 0, -v[0], -v[1], v[0], 0)
	k = np.matrix(vXStr)
	r = I + k + k@k * ((1 -c)/(s**2))
	return r


#####################################################################
####################### NODES EXTRACTION ############################
#####################################################################

def distance(a,b):
	return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def det(a, b):
	return a[0] * b[1] - a[1] * b[0]

def line_intersection(line1, line2):
	xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
	ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

	div = det(xdiff, ydiff)
	if div == 0:
		return None,None

	max_x_1 = max(line1[0][0],line1[1][0])
	max_x_2 = max(line2[0][0],line2[1][0])
	max_y_1 = max(line1[0][1],line1[1][1])
	max_y_2 = max(line2[0][1],line2[1][1])
	
	min_x_1 = min(line1[0][0],line1[1][0])
	min_x_2 = min(line2[0][0],line2[1][0])
	min_y_1 = min(line1[0][1],line1[1][1])
	min_y_2 = min(line2[0][1],line2[1][1])
	
	
	d = (det(*line1), det(*line2))
	x = det(d, xdiff) / div
	y = det(d, ydiff) / div
	if not(((x <= max_x_1) and (x >= min_x_1)) and ((x <= max_x_2) and (x >= min_x_2))):
		return None,None
	if not(((y <= max_y_1) and (y >= min_y_1)) and ((y <= max_y_2) and (y >= min_y_2))):
		return None,None
	return [x, y], distance(line1[0],[x,y])

def find_tuple_interseted(proj,line):
	result = []
	dist_l = []
	for i in range(len(proj)-1):
		intersect,dist = line_intersection(line, [proj[i],proj[i+1]])
		if intersect is not None:
			result.append(intersect)
			dist_l.append(dist)
	return [result,dist_l]

def PointsInCircum(r,n=500):
	return [(math.cos(2*np.pi/n*x)*r,math.sin(2*np.pi/n*x)*r) for x in range(0,n)]


def find_closest_node(list_maxima_ind,point):
	result_list = [np.abs(maxi - point) for maxi in list_maxima_ind]
	result_list_sorted = sorted(result_list)
	return result_list.index(result_list_sorted[0])
	

def find_theta_to_check(proj,k,rate):
	k_0 = proj[k][0]
	k_1 = proj[k][1]
	k_1_0 = proj[k+1][0]
	k_1_1 = proj[k+1][1]
	dist_to_0 = np.sqrt(k_0**2 + k_1**2)
	dist_to_1 = np.sqrt(k_1_0**2 + k_1_1**2)
	theta_point = np.arctan2([k_1/dist_to_0],[k_0/dist_to_0])[0]
	theta_point_1 = np.arctan2([k_1_1/dist_to_1],[k_1_0/dist_to_1])[0]
	if theta_point < 0:
		theta_point += 2*np.pi    
	if theta_point_1 < 0:
		theta_point_1 += 2*np.pi    
	theta_point = int(theta_point/(2.0*np.pi) * (rate)) 
	theta_point_1 = int(theta_point_1/(2.0*np.pi) * (rate))
	diff_theta = abs(theta_point - theta_point_1)
	if diff_theta > rate//2:
		if theta_point_1 > rate//2:
			diff_theta = abs(theta_point - (-rate + theta_point_1))
		elif theta_point > rate//2:
			diff_theta = abs((-rate + theta_point) - theta_point_1)
	diff_theta = min(diff_theta,rate//2)
	theta_to_check = [(theta_point + lag) % rate for lag in range(-diff_theta-1,diff_theta+1)]
	return theta_to_check
		

def get_intersection_from_radius(A,col1,col2,rate=100):
	
	max_1 = max(max(A[col1].values),abs(min(A[col1].values)))
	max_2 = max(max(A[col2].values),abs(min(A[col2].values)))
	set_point = PointsInCircum(np.sqrt(max_1**2 + max_2**2),n=rate)
	previous_node = "not_defined"

	result = [[] for i in range(len(set_point))]
	result_dist = [[] for i in range(len(set_point))]

	proj = A[[col1,col2]].values
	for k in range(0,len(A)-1):	
		theta_to_check = find_theta_to_check(proj,k,rate)
		was_found = False
		for i in theta_to_check:
			intersect,dist = line_intersection(
				[[0,0],set_point[i]],
				[proj[k],proj[k+1]])
			if intersect is not None:
				was_found = True
				result[i].append(intersect)
				result_dist[i].append(dist)
	new_result_dist = []
	for i,res_d in enumerate(result_dist):
		new_result_dist += res_d
	return result,new_result_dist




def get_intersection_from_radius_(A,col1,col2,rate=100):
	result = []
	result_dist = []
	max_1 = max(max(A[col1].values),abs(min(A[col1].values)))
	max_2 = max(max(A[col2].values),abs(min(A[col2].values)))
	set_point = PointsInCircum(np.sqrt(max_1**2 + max_2**2),n=rate)
	proj = A[[col1,col2]].values
	for i in range(len(set_point)):
		intersect = find_tuple_interseted(proj,[[0,0],set_point[i]])
		result.append(intersect[0])
		result_dist += intersect[1]
	return result,result_dist


def kde_scipy(x, x_grid):
	kde = gaussian_kde(x, bw_method='scott')
	return list(kde.evaluate(x_grid))


def nodes_extraction(A,col1,col2,res_point,res_dist,rate=100):
	max_all = max(max(max(A[col1].values),max(A[col2].values)),max(-min(A[col1].values),-min(A[col2].values)))
	max_all = max_all*1.2
	list_maxima = []
	list_maxima_val = []
	for segment in range(rate):
		pos_start = sum(len(res_point[i]) for i in range(segment))
		dist_on_segment = kde_scipy(res_dist[pos_start:pos_start+len(res_point[segment])], 
									np.arange(0, max_all, max_all/250.0)) 
		dist_on_segment = (dist_on_segment - min(dist_on_segment))/(max(dist_on_segment) - min(dist_on_segment))
		maxima = argrelextrema(np.array(dist_on_segment), np.greater)[0]
		maxima_ind = [np.arange(0, max_all, max_all/250.0)[val] for val in list(maxima)]
		maxima_val = [dist_on_segment[val] for val in list(maxima)]
		list_maxima.append(maxima_ind)
		list_maxima_val.append(maxima_val)
	return list_maxima,list_maxima_val
	


	
#####################################################################
####################### EDGES EXTRACTION ############################
#####################################################################
	
def edges_extraction(A,col1,col2,set_nodes,rate=100):
	
	list_edge = []
	new_node_list = []
	node_evo = []
	edge_in_time = []
	
	max_1 = max(max(A[col1].values),abs(min(A[col1].values)))
	max_2 = max(max(A[col2].values),abs(min(A[col2].values)))
	set_point = PointsInCircum(np.sqrt(max_1**2 + max_2**2),n=rate)
	previous_node = "not_defined"

	proj = A[[col1,col2]].values
	for k in range(0,len(A)-2):
		
		theta_to_check = find_theta_to_check(proj,k,rate)
		was_found = False
		for i in theta_to_check:
			to_add = find_tuple_interseted(proj[k:k+2],[[0,0],set_point[i]])[1]
			if to_add == [] and not was_found:
				continue
			elif to_add == [] and was_found:
				break
			else:
				was_found = True
				node_in = find_closest_node(set_nodes[i],to_add[0])
				if "{}_{}".format(i,node_in) not in new_node_list:
					new_node_list.append("{}_{}".format(i,node_in))
				
				if previous_node == "not_defined":
					previous_node = "{}_{}".format(i,node_in)
				else:
					list_edge.append([previous_node,"{}_{}".format(i,node_in)])
					
					previous_node = "{}_{}".format(i,node_in)
		#if (not was_found) and (previous_node != "not_defined"):
		#	list_edge.append([previous_node,previous_node])
		edge_in_time.append(len(list_edge))		
		node_evo.append(len(new_node_list))

	return list_edge,node_evo,edge_in_time


def get_nodes_from_P(G,node_set,P,latent,length_pattern,pca_1,R,skip=1,rate=100):
	P_space = build_phase_space(P,latent,length_pattern,skip)
	reduced_P = pd.DataFrame(pca_1.transform(P_space),columns=[str(i) for i in range(3)])
	A = np.dot(R,reduced_P.T)
	A = pd.DataFrame(A.T,columns=['0','1','2'])
	list_edge,_,_ = edges_extraction(A,'0','1',node_set,rate)
	return list_edge,A



def score_P_degree(dict_edge,list_edge_P,node_degree):

	score = np.sum(dict_edge[str(edge)]*(node_degree[edge[0]]-1) for edge in list_edge_P)/float(0.00000001+len(list_edge_P))
	return score,len(list_edge_P)
