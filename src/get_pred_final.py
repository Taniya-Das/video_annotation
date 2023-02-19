import torch
import numpy as np
import time
import utils
from utils import acc_f1_from_binary_confusion_mat
import torch.nn.functional as F
from pdb import set_trace
from dl_utils.label_funcs import unique_labels
from statistics import mean
from dl_utils.misc import show_gpu_memory
from dl_utils.tensor_funcs import numpyify
import matplotlib.pyplot as plt
torch.manual_seed(0)
from time import time
from utils import asMinutes
import operator
import pickle
import os
import csv
from scipy.integrate import quad
import math

def plot_class(count, figname):
    plt.plot(count)
    plt.show(block=True)
    plt.savefig(figname)

def topK_classifications(ind_multiclassifications, class_multiclassifications, rel_multiclassifications, top):
    ind_softmax = F.softmax(ind_multiclassifications)
    class_softmax = F.softmax(class_multiclassifications)
    rel_softmax = F.softmax(rel_multiclassifications)

    ind_prob = np.expand_dims(ind_softmax[0].detach().cpu().numpy(), axis=0)
    class_prob = np.expand_dims(class_softmax[0].detach().cpu().numpy(), axis = 1)
    rel_prob = np.expand_dims(rel_softmax[0].detach().cpu().numpy(), axis = 1)

    classes_mat = class_prob*ind_prob #dim 129*285

    temp = np.ones((len(ind_prob[0]), (len(ind_prob[0]))))
    np.fill_diagonal(temp, 0)
    sub_obj = ind_prob.T * ind_prob * temp
    sub_obj = rel_prob * sub_obj.reshape((1,len(ind_prob[0])*len(ind_prob[0])))
    relations_mat = sub_obj.reshape((len(rel_prob),len(ind_prob[0]),len(ind_prob[0])))
 
    class_mat = torch.from_numpy(classes_mat)
    value_class, i = torch.topk(class_mat.flatten(), top)
    index_class = np.array(np.unravel_index(i.numpy(), classes_mat.shape)).T #Use indexes by index_class[0], etc.

    rel_mat = torch.from_numpy(relations_mat)
    value_rel, i = torch.topk(rel_mat.flatten(), top)
    index_rel = np.array(np.unravel_index(i.numpy(), rel_mat.shape)).T
    
    return index_class, index_rel 

def compute_f1_for_thresh(annotation, atoms, lcwa):
    tp,fp,fn,tn = 0,0,0,0
    for a in atoms:
        if a in annotation:
            tp += 1
        else:
            fn += 1
    for a in lcwa:
        if a in annotation:
            fp += 1
        else:
            tn += 1
    acc,f1 = acc_f1_from_binary_confusion_mat(tp,fp,tn,fn)
    #print("tp: {}, fp: {}, fn: {}, tn: {}, f1: {}".format(tp,fp,fn,tn, f1))
    return tp, fp, fn, tn, f1, acc

def mlp_value_dict_compute(ind_dict, index_class, index_rel, pred_mlps, encodings):
    mlp_value_dict = {}
    with torch.no_grad():
        for i in range(len(index_class)):
            sub_id, subj_val = tuple(list(ind_dict.items())[index_class[i][1]])
            class_id, class_mlp = tuple(list(pred_mlps['classes'].items())[index_class[i][0]])
            context_embedding = torch.cat([encodings[0], subj_val])
            mlp_value_dict.update({(class_id, sub_id):numpyify(F.sigmoid(class_mlp(context_embedding)))})

        for i in range(len(index_rel)):
            sub_id, subj_val = tuple(list(ind_dict.items())[index_rel[i][1]])
            rel_id, rel_mlp = tuple(list(pred_mlps['relations'].items())[index_rel[i][0]])
            obj_id, obj_val = tuple(list(ind_dict.items())[index_rel[i][2]])
            context_embedding = torch.cat([encodings[0], subj_val, obj_val])
            mlp_value_dict.update({(rel_id,sub_id,obj_id):numpyify(F.sigmoid(rel_mlp(context_embedding)))})
            
    return mlp_value_dict


def get_pred_loss(video_ids, encodings, dataset_dict, testing, margin=1, device='cuda'):
    loss = torch.tensor([0.], device=device)
    num_atoms = 0
    if testing: pos_predictions, neg_predictions = [],[]
    json_data_dict,ind_dict,pred_mlps = dataset_dict['dataset'],dataset_dict['ind_dict'],dataset_dict['mlp_dict']
    for video_id, encoding in zip(video_ids,encodings):
        dpoint = json_data_dict[video_id.item()]
        atoms = dpoint['pruned_atoms_with_synsets']
        lcwa = dpoint['lcwa'][:len(atoms)]
        try: neg_weight = float(len(atoms))/len(lcwa)
        except ZeroDivisionError: pass # Don't define neg_weight because it shouldn't be needed
        truth_values = [True]*len(atoms) + [False]*len(lcwa)
        if truth_values == []: continue
        num_atoms += len(truth_values)
        # Use GT inds to test predictions for GT atoms
        for tfatom,truth_value in zip(atoms+lcwa,truth_values):
            arity = len(tfatom)-1
            if arity == 1:
                # Unary predicate
                predname,subname = tfatom
                context_embedding = torch.cat([encoding, ind_dict[subname[1]]])
            elif arity == 2:
                # Binary predicate
                predname,subname,objname = tfatom
                context_embedding = torch.cat([encoding, ind_dict[subname[1]],ind_dict[objname[1]]])
            else: set_trace()
            mlp = pred_mlps['classes' if arity==1 else 'relations'][predname[1]]
            prediction = mlp(context_embedding)
            if testing:
                if truth_value: pos_predictions.append(prediction.item())
                else: neg_predictions.append(prediction.item())
            else:
                if truth_value: loss += F.relu(-prediction+margin)
                else: loss += neg_weight*F.relu(prediction+margin)
    if testing:return pos_predictions, neg_predictions
    else: return loss if num_atoms == 0 else loss/num_atoms

def convert_atoms_to_ids_only(atoms_list):
    return [tuple([item[1] for item in atom]) for atom in atoms_list]


def inference(video_ids, encodings, ind_multiclassifications, class_multiclassifications, rel_multiclassifications, dataset_dict, dataset, threshold, threshold_VG, fragment_name, attribute_stats, relationship_stats, sub_count_c, sub_count_r, model, VG):
    json_data_dict = dataset_dict['dataset']
    dpoint = json_data_dict[video_ids.item()]
    atoms = dpoint['pruned_atoms_with_synsets']

    annotation = []
    annotations_by_id = {}

    if len(atoms)==0:
        acc = -1
        f1 = -1
        pos = -1
        neg = -1

    else: 
        ind_dict = dataset_dict['ind_dict']
        pred_mlps  = dataset_dict['mlp_dict']
        annotations_by_id = {}
        
        print('evaluating', video_ids.item())


        top = 1000 #Change if required

        """
        #To get top k values and built mlp_value_dict containing facts and corresponding prediction
        #from predicate-MLPs

        index_class, index_rel = topK_classifications(ind_multiclassifications, class_multiclassifications, rel_multiclassifications, top)
        mlp_value_dict = mlp_value_dict_compute(ind_dict, index_class, index_rel, pred_mlps, encodings)
        mlp_value_dict_path = '../data/MSRVTT/mlp_value_dict/'+str(video_ids.item())+'.pickle'
        with open(mlp_value_dict_path, 'wb') as handle:
            pickle.dump(mlp_value_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        """
        mlp_value_dict_path = '../data/'+ str(dataset) +'/mlp_value_dict/'+str(video_ids.item())+'.pickle'
        with open(mlp_value_dict_path, 'rb') as handle:
            mlp_value_dict = pickle.load(handle)

        lcwa = dpoint['lcwa']
        atoms = convert_atoms_to_ids_only(atoms)
        lcwa = convert_atoms_to_ids_only(lcwa)

        if VG:
            for key, value in mlp_value_dict.items():
                if len(key) == 2:
                    v = attribute_stats.get(key,0)
                    if v or sub_count_c.get(key[1],0):
                        D = v 
                        N = sub_count_c.get(key[1],0)
                        prob = (D+1)/(N+2)
                        X_test = [float(value), prob]
                        X_test = np.array(X_test).reshape(1, -1)
                        Y_pred = model.predict(X_test) 
                        Y_prob = model.predict_proba(X_test)
                        if Y_prob[0][1] > threshold_VG:
                            annotation.append(key)
                            flag = 1
                    elif value > threshold_VG:
                        annotation.append(key)
                else:
                    v = relationship_stats.get(key,0)
                    if v or sub_count_r.get(tuple(key[1:]),0):
                        prob = []
                        D = v 
                        N = sub_count_r.get(tuple(key[1:]),0)
                        prob = (D+1)/(N+2)
                        X_test = [float(value), prob]
                        X_test = np.array(X_test).reshape(1, -1)
                        Y_pred = model.predict(X_test) 
                        Y_prob = model.predict_proba(X_test)
                        if Y_prob[0][1] > threshold_VG:
                        #if Y_pred:
                            annotation.append(key)
                            flag = 1
                    elif value > threshold_VG:
                        annotation.append(key)
        else:
            annotation = [key for key, value in mlp_value_dict.items() if value > threshold]
        
        tp, fp, fn, tn, f1, acc = compute_f1_for_thresh(annotation, atoms, lcwa)
        print("tp: {}, fp: {}, fn: {}, tn: {}, f1: {}".format(tp,fp,fn,tn,f1))
        pos = tp/(tp + fn)
        neg = tn/(tn + fp)
    
    annotations_by_id[video_ids[0]] = {'annotation': annotation,
                                       'acc': acc,
                                       'f1': f1,
                                       'pos': pos,
                                       'neg': neg}
    
    """
    annotations_by_id[video_ids[0]] = {'annotation': 0,
                                       'acc': 0,
                                       'f1': 0,
                                       'pos': 0,
                                       'neg': 0}
    """
    
    return annotations_by_id
                            
def compute_probs_for_dataset(dl,encoder,multiclassifier,multiclassifier_class,multiclassifier_rel,dataset_dict,use_i3d, dataset, threshold,threshold_VG,fragment_name, attribute_stats, relationship_stats, sub_count_c, sub_count_r, model, VG):
    all_accs = []
    all_f1s = []
    all_pos = []
    all_neg = []

    for d in dl:
        video_tensor = d[0].float().transpose(0,1).to('cuda')
        video_ids = d[4].to('cuda')
        i3d = d[5].float().to('cuda')
        encodings, enc_hidden = encoder(video_tensor)
        if use_i3d: encodings = torch.cat([encodings,i3d],dim=-1)
        multiclassif = multiclassifier(encodings)
        multiclassif_class = multiclassifier_class(encodings)
        multiclassif_rel = multiclassifier_rel(encodings)
        
        annotations_by_id = inference(video_ids, encodings, multiclassif, multiclassif_class, multiclassif_rel, dataset_dict, dataset, threshold, threshold_VG, fragment_name, attribute_stats, relationship_stats, sub_count_c, sub_count_r, model, VG)
        
        all_accs += [vid['acc'] for vid in annotations_by_id.values()]
        all_f1s += [vid['f1'] for vid in annotations_by_id.values()]
        all_pos += [vid['pos'] for vid in annotations_by_id.values()]
        all_neg += [vid['neg'] for vid in annotations_by_id.values()]
        

    total_acc = np.array([x for x in all_accs if x>=0]).mean()
    total_f1 = np.array([x for x in all_f1s if x>=0]).mean()
    total_pos = np.array([x for x in all_pos if x>=0]).mean()
    total_neg = np.array([x for x in all_neg if x>=0]).mean()
    
    return total_acc, total_f1, total_pos, total_neg

