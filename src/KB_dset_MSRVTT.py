from dl_utils.misc import check_dir
from collections import OrderedDict
import sys
import os
import json
from pdb import set_trace
from torch import optim
from utils import get_datetime_stamp, asMinutes,get_w2v_vec
from time import time
from gensim.models import KeyedVectors

import options
import my_models
import train
import data_loader
from semantic_parser import tuplify
from get_metrics_final import compute_dset_fragment_scores
import pickle

from sklearn.linear_model import LogisticRegressionCV
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

from build_LR_dataset import build_LR_dataset
from Logistic_Regression import train_LR, train_svm, train_mlp, train_rbf

args = options.load_arguments()

#Use MSRVTT for VG dataset
if args.dataset == 'MSVD':

    json_path = f"MSRVTT_final.json"

    with open(json_path) as f: json_data=json.load(f)
    inds,classes,relations,json_data_list = json_data['inds'],json_data['classes'],json_data['relations'],json_data['dataset']

    for dp in json_data_list:
        dp['pruned_atoms_with_synsets'] = [tuplify(a) for a in dp['pruned_atoms_with_synsets']]
        
    json_data_dict = {dp['video_id']:dp for dp in json_data_list}

    attribute_stats = {}
    relationship_stats = {}
    sub_count_c = {}
    sub_count_r = {}

    for i in range(len(json_data_dict)):
        for dp in json_data_dict[i]['pruned_atoms_with_synsets']:
            if len(dp) == 3:
                relationship_stats[(dp[0][1],dp[1][1],dp[2][1])] = relationship_stats.get((dp[0][1],dp[1][1],dp[2][1]),0) + 1
                sub_count_r[(dp[1][1],dp[2][1])] = sub_count_r.get((dp[1][1],dp[2][1]),0) + 1

            elif len(dp) == 2:
                attribute_stats[(dp[0][1],dp[1][1])] = attribute_stats.get((dp[0][1],dp[1][1]),0) + 1
                sub_count_c[dp[1][1]] = sub_count_c.get(dp[1][1],0) + 1


    with open('msrvtt_attribute_dict.pickle', 'wb') as handle:
        pickle.dump(attribute_stats , handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('msrvtt_relationship_dict.pickle', 'wb') as handle:
        pickle.dump(relationship_stats , handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('msrvtt_sub_count_c.pickle', 'wb') as handle:
        pickle.dump(sub_count_c , handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('msrvtt_sub_count_r.pickle', 'wb') as handle:
        pickle.dump(sub_count_r , handle, protocol=pickle.HIGHEST_PROTOCOL)


#Use MSRVTT for VG dataset
if args.dataset == 'MSRVTT':

    json_path = f"MSVD_final.json"

    with open(json_path) as f: json_data=json.load(f)
    inds,classes,relations,json_data_list = json_data['inds'],json_data['classes'],json_data['relations'],json_data['dataset']

    for dp in json_data_list:
        dp['pruned_atoms_with_synsets'] = [tuplify(a) for a in dp['pruned_atoms_with_synsets']]
        
    json_data_dict = {dp['video_id']:dp for dp in json_data_list}

    attribute_stats = {}
    relationship_stats = {}
    sub_count_c = {}
    sub_count_r = {}

    for i in range(len(json_data_dict)):
        for dp in json_data_dict[i]['pruned_atoms_with_synsets']:
            if len(dp) == 3:
                relationship_stats[(dp[0][1],dp[1][1],dp[2][1])] = relationship_stats.get((dp[0][1],dp[1][1],dp[2][1]),0) + 1
                sub_count_r[(dp[1][1],dp[2][1])] = sub_count_r.get((dp[1][1],dp[2][1]),0) + 1

            elif len(dp) == 2:
                attribute_stats[(dp[0][1],dp[1][1])] = attribute_stats.get((dp[0][1],dp[1][1]),0) + 1
                sub_count_c[dp[1][1]] = sub_count_c.get(dp[1][1],0) + 1


    with open('msvd_attribute_dict.pickle', 'wb') as handle:
        pickle.dump(attribute_stats , handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('msvd_relationship_dict.pickle', 'wb') as handle:
        pickle.dump(relationship_stats , handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('msvd_sub_count_c.pickle', 'wb') as handle:
        pickle.dump(sub_count_c , handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('msvd_sub_count_r.pickle', 'wb') as handle:
        pickle.dump(sub_count_r , handle, protocol=pickle.HIGHEST_PROTOCOL)

        

        


        

        

