import torch
import numpy as np
from time import time
import utils
import torch.nn.functional as F
import sys
from dl_utils.tensor_funcs import numpyify
torch.manual_seed(0)
from collections import defaultdict
import csv
from utils import asMinutes
import os
import random
#from scipy.integrate import quad
import math 
from dl_utils.misc import check_dir

def convert_atoms_to_ids_only(atoms_list):
    return [tuple([item[1] for item in atom]) for atom in atoms_list]


def build_dataset(dl, encoder, dataset_dict, attribute_stats, relationship_stats, sub_count_c, sub_count_r, use_i3d):
    
    train_LR = {}

    for d in dl:
        video_tensor = d[0].float().transpose(0,1).to('cuda')
        video_ids = d[4].to('cuda')
        i3d = d[5].float().to('cuda')
        encodings, enc_hidden = encoder(video_tensor)
        if use_i3d: encodings = torch.cat([encodings,i3d],dim=-1)

        ind_dict = dataset_dict['ind_dict']
        pred_mlps  = dataset_dict['mlp_dict']

        json_data_dict = dataset_dict['dataset']
        dpoint = json_data_dict[video_ids.item()]
        atoms = dpoint['pruned_atoms_with_synsets']
        lcwa_temp = dpoint['lcwa']
        lcwa = []
        for _ in range(0,len(atoms)):
            lcwa.append(lcwa_temp[random.randint(0,len(lcwa_temp)-1)])  
        atoms = convert_atoms_to_ids_only(atoms)
        lcwa = convert_atoms_to_ids_only(lcwa)

        count_atom, count_lcwa = 0, 0
        with torch.no_grad():
            for i in range(len(atoms)):
                if len(atoms[i]) == 3:
                    rel_id, sub_id, obj_id = atoms[i][0], atoms[i][1], atoms[i][2]
                    rel_mlp = pred_mlps['relations'][atoms[i][0]]
                    subj_val = ind_dict[atoms[i][1]]
                    obj_val = ind_dict[atoms[i][2]]
                    context_embedding = torch.cat([encodings[0], subj_val, obj_val])
                    v = relationship_stats.get((rel_id, sub_id, obj_id),0)
                    if v or sub_count_r.get((sub_id,obj_id),0):
                        D, N = v, sub_count_r.get((sub_id,obj_id))
                        prob = (D+1)/(N+2)
                        train_LR.update({(rel_id,sub_id,obj_id):[numpyify(F.sigmoid(rel_mlp(context_embedding))), prob, 1]})
                        count_atom += 1
                else:
                    class_id, sub_id = atoms[i][0], atoms[i][1]
                    class_mlp = pred_mlps['classes'][atoms[i][0]]
                    subj_val = ind_dict[atoms[i][1]]
                    context_embedding = torch.cat([encodings[0], subj_val])
                    v = attribute_stats.get((class_id, sub_id),0)
                    if v or sub_count_c.get(sub_id,0):
                        D, N = v, sub_count_c.get(sub_id)
                        prob = (D+1)/(N+2)
                        train_LR.update({(class_id,sub_id):[numpyify(F.sigmoid(class_mlp(context_embedding))), prob, 1]})
                        count_atom += 1

            for i in range(len(lcwa)):
                if len(lcwa[i]) == 3:
                    rel_id, sub_id, obj_id = lcwa[i][0], lcwa[i][1], lcwa[i][2]
                    rel_mlp = pred_mlps['relations'][lcwa[i][0]]
                    subj_val = ind_dict[lcwa[i][1]]
                    obj_val = ind_dict[lcwa[i][2]]
                    context_embedding = torch.cat([encodings[0], subj_val, obj_val])
                    v = relationship_stats.get((rel_id, sub_id, obj_id),0)
                    if (v or sub_count_r.get((sub_id,obj_id),0)) and count_lcwa < count_atom:
                        D, N = v, sub_count_r.get((sub_id,obj_id))
                        prob = (D+1)/(N+2)
                        train_LR.update({(rel_id,sub_id,obj_id):[numpyify(F.sigmoid(rel_mlp(context_embedding))), prob, 0]})
                        count_lcwa += 1
                else:
                    class_id, sub_id = lcwa[i][0], lcwa[i][1]
                    class_mlp = pred_mlps['classes'][lcwa[i][0]]
                    subj_val = ind_dict[lcwa[i][1]]
                    context_embedding = torch.cat([encodings[0], subj_val])
                    v = attribute_stats.get((class_id, sub_id),0) 
                    if (v or sub_count_c.get(sub_id,0)) and count_lcwa < count_atom:
                        D, N = v, sub_count_c.get(sub_id)
                        prob = (D+1)/(N+2)
                        train_LR.update({(class_id,sub_id):[numpyify(F.sigmoid(class_mlp(context_embedding))), prob, 0]})
                        count_lcwa += 1
    return train_LR

        
def build_LR_dataset(epoch, dl, encoder, dataset_dict, attribute_stats, relationship_stats, sub_count_c, sub_count_r, use_i3d, fragment_name):

    print("Building LR dataset")

    t1 = time()
    train_LR = build_dataset(dl, encoder, dataset_dict, attribute_stats, relationship_stats, sub_count_c, sub_count_r, use_i3d)
    print("Time Taken: ",{asMinutes(time()-t1)})

    #dir = '../data/train_LR/MSRVTT/epoch' + str(epoch)
    #dir = '../data/train_LR/MSVD'
    dir = '../data/MSVD/train_LR/epoch' + str(epoch)
    check_dir(dir)
    filename = fragment_name + '_LR' + '.csv'
    #filename = os.path.join('../data/train_LR',filename)
    filename = os.path.join(dir,filename)
    
    with open(filename, 'w') as f:

        writer = csv.writer(f)
        for key, value in train_LR.items():
            row = [float(value[0]), value[1], value[2]]
            writer.writerow(row)
