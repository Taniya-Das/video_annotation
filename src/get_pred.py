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


def plot_class(count, figname):
    plt.plot(count)
    plt.show(block=True)
    plt.savefig(figname)

"""
def check_classifier(video_id, inds_to_use, class_to_use, rel_to_use, atoms, ind_multiclassifications, class_multiclassifications, rel_multiclassifications):
    ind_softmax = F.softmax(ind_multiclassifications)
    class_softmax = F.softmax(class_multiclassifications)
    rel_softmax = F.softmax(rel_multiclassifications)

    ind_prob = ind_softmax[0].detach().cpu().numpy()
    class_prob = 

    breakpoint()
    ind_prob_dict, class_prob_dict, rel_prob_dict = {}, {}, {}

    for i in range(len(inds_to_use)):
        ind_prob_dict.update({inds_to_use[i][0]: numpyify(ind_prob[0][i])})
    for i in range(len(class_to_use)):
        class_prob_dict.update({class_to_use[i][0]: numpyify(class_prob[0][i])})
    for i in range(len(rel_to_use)):
        rel_prob_dict.update({rel_to_use[i][0]: numpyify(rel_prob[0][i])})

    attribute, relation = [], []
    for subj_id, subj_val in ind_prob_dict.items():
        for class_id, class_val in class_prob_dict.items():
            attribute.append(((class_id, subj_id), subj_val*class_val))
        for obj_id, obj_val in ind_prob_dict.items():
            if obj_id==subj_id: continue # Assume no reflexive predicates
            for rel_id, rel_val in rel_prob_dict.items():
                relation.append(((rel_id,subj_id,obj_id),subj_val*rel_val*obj_val))


    attribute = sorted(attribute, key=lambda x: x[1], reverse=True)
    relation = sorted(relation, key=lambda x: x[1], reverse=True)

    c, r = -1, -1
    for i in range(len(attribute)):
        if attribute[i][0] in atoms:
            c = i
            break
        if i == 200:
            break

    for i in range(len(relation)):
        if relation[i][0] in atoms:
            r = i
            break
        if i == 200:
            break
    print("VidoeID: {}, class: {}, rel: {} ".format(video_id, c, r))
    return c, r
"""  

def topK_classifications(ind_multiclassifications, class_multiclassifications, rel_multiclassifications):
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
    sub_obj = rel_prob * sub_obj.reshape((1,285*285))
    relations_mat = sub_obj.reshape((150,285,285))
    """
    relations_mat = np.zeros((150,285,285)) #dim 150*285*285
    t1 = time()
    for i in range(150):
        for j in range(285):
            for k in range(285):
                if j==k:continue
                relations_mat[i,j,k] = rel_prob[i]*ind_prob[0][j]*ind_prob[0][k]

    print("Time for relations_mat: ", {asMinutes(time()-t1)})
    """

    class_mat = torch.from_numpy(classes_mat)
    value_class, i = torch.topk(class_mat.flatten(), 200)
    index_class = np.array(np.unravel_index(i.numpy(), classes_mat.shape)).T #Use indexes by index_class[0], etc.

    rel_mat = torch.from_numpy(relations_mat)
    value_rel, i = torch.topk(rel_mat.flatten(), 200)
    index_rel = np.array(np.unravel_index(i.numpy(), rel_mat.shape)).T
    
    return index_class, index_rel 


def mlp_output_dict(encoding, inds_to_use, class_to_use, rel_to_use):
    with torch.no_grad():
        mlp_value_dict = {}
        for subj_id, subj_vector in inds_to_use:
            for class_id, class_mlp in class_to_use:
                context_embedding = torch.cat([encoding, subj_vector])
                mlp_value_dict.update({(class_id,subj_id):numpyify(class_mlp(context_embedding))[0]})
            for obj_id, obj_vector in inds_to_use:
                if obj_id==subj_id: continue # Assume no reflexive predicates
                for rel_id, rel_mlp in rel_to_use:
                    #show_gpu_memory()
                    context_embedding = torch.cat([encoding, subj_vector, obj_vector])
                    mlp_value_dict.update({(rel_id,subj_id,obj_id):numpyify(rel_mlp(context_embedding))[0]}) 
    return mlp_value_dict


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

def inference(video_ids, encodings, ind_multiclassifications, class_multiclassifications, rel_multiclassifications, dataset_dict):
    json_data_dict = dataset_dict['dataset']

    dpoint = json_data_dict[video_ids.item()]
    atoms = dpoint['pruned_atoms_with_synsets']

    annotation = []
    annotations_by_id = {}

    if len(atoms)==0:
        acc = -1
        f1 = -1

    else: 
        ind_dict = dataset_dict['ind_dict']
        pred_mlps  = dataset_dict['mlp_dict']
        annotations_by_id = {}
        
        print('evaluating', video_ids.item())
        t1 = time()
        index_class, index_rel = topK_classifications(ind_multiclassifications, class_multiclassifications, rel_multiclassifications)
        print("Time to get top 200: ", {asMinutes(time()-t1)})

        t1 = time()
        with torch.no_grad():
            for i in range(len(index_class)):
                sub_id, subj_val = tuple(list(ind_dict.items())[index_class[i][1]])
                class_id, class_mlp = tuple(list(pred_mlps['classes'].items())[index_class[i][0]])
                context_embedding = torch.cat([encodings[0], subj_val])
                if class_mlp(context_embedding) > 0:
                    annotation.append((class_id, sub_id))

            for i in range(len(index_rel)):
                sub_id, subj_val = tuple(list(ind_dict.items())[index_rel[i][1]])
                rel_id, rel_mlp = tuple(list(pred_mlps['relations'].items())[index_rel[i][0]])
                obj_id, obj_val = tuple(list(ind_dict.items())[index_rel[i][2]])
                context_embedding = torch.cat([encodings[0], subj_val, obj_val])
                if rel_mlp(context_embedding) > 0:
                    annotation.append((rel_id, sub_id, obj_id))

        print("MLP calculations:", {asMinutes(time()-t1)})        

        # Compare to GT and compute scores
        lcwa = dpoint['lcwa']
        atoms = convert_atoms_to_ids_only(atoms)
        lcwa = convert_atoms_to_ids_only(lcwa)
        tp = len([a for a in atoms if a in annotation])
        fp = len([a for a in lcwa if a in annotation])
        fn = len([a for a in atoms if a not in annotation])
        tn = len([a for a in lcwa if a not in annotation])
        acc,f1 = acc_f1_from_binary_confusion_mat(tp,fp,tn,fn)
        print("tp: {} ,fp: {} ,tn: {}, fn: {}, f1: {}, acc: {}\n".format(tp,fp,tn,fn,f1,acc))

    annotations_by_id[video_ids[0]] = {'annotation': annotation,
                                    'acc': acc,
                                    'f1': f1}

    return annotations_by_id



def compute_probs_for_dataset(dl,encoder,multiclassifier,multiclassifier_class,multiclassifier_rel,dataset_dict,use_i3d):
    pos_classifications, neg_classifications, pos_classifications_class, neg_classifications_class, pos_classifications_rel, neg_classifications_rel, pos_predictions, neg_predictions, perfects = [],[],[],[],[],[],[],[],{}
    all_accs = []
    all_f1s = []
    all_count_c = []
    all_count_r = []

    start = time()

    for d in dl:
        video_tensor = d[0].float().transpose(0,1).to('cuda')
        multiclass_inds = d[1].to('cuda')
        multiclass_class = d[2].to('cuda')
        multiclass_rel = d[3].to('cuda')
        #print(multiclass_inds)

        video_ids = d[4].to('cuda')
        i3d = d[5].float().to('cuda')
        encodings, enc_hidden = encoder(video_tensor)
        if use_i3d: encodings = torch.cat([encodings,i3d],dim=-1)
        multiclassif = multiclassifier(encodings)
        multiclassif_class = multiclassifier_class(encodings)
        multiclassif_rel = multiclassifier_rel(encodings)
        
        #assert (multiclass_inds==1).sum() + (multiclass_inds==0).sum() == multiclass_inds.shape[1]
        assert unique_labels(multiclass_inds).issubset(set([0,1]))
        assert unique_labels(multiclass_class).issubset(set([0,1]))
        assert unique_labels(multiclass_rel).issubset(set([0,1]))

        multiclass_inds = multiclass_inds.type(torch.bool)
        multiclass_class = multiclass_class.type(torch.bool)
        multiclass_rel = multiclass_rel.type(torch.bool)

        new_pos_classifications,new_neg_classifications = multiclassif[multiclass_inds], multiclassif[~multiclass_inds]
        new_pos_classifications_class,new_neg_classifications_class = multiclassif_class[multiclass_class], multiclassif_class[~multiclass_class]
        new_pos_classifications_rel,new_neg_classifications_rel = multiclassif_rel[multiclass_rel], multiclassif_rel[~multiclass_rel]
        
        #To evaluate predicate MLPs by comparing with GT individuals. 
        new_pos_predictions, new_neg_predictions = get_pred_loss(video_ids, encodings, dataset_dict, testing=True)
        
        annotations_by_id = inference(video_ids, encodings, multiclassif, multiclassif_class, multiclassif_rel, dataset_dict)

        all_accs += [vid['acc'] for vid in annotations_by_id.values()]
        all_f1s += [vid['f1'] for vid in annotations_by_id.values()]
  
        if (new_pos_classifications>0).all() and (new_neg_classifications<0).all() and all([p>0 for p in new_pos_predictions]) and all([p<0 for p in new_neg_predictions]): perfects[int(video_ids[0].item())] = len(new_pos_predictions)
        #if all([p>0 for p in new_pos_predictions]) and all([p<0 for p in new_neg_predictions]): perfects[video_ids[0].item()] = len(new_pos_predictions)
        
        pos_predictions += new_pos_predictions
        neg_predictions += new_neg_predictions

        pos_classifications += new_pos_classifications.tolist()
        neg_classifications += new_neg_classifications.tolist()
        
        pos_classifications_class += new_pos_classifications_class.tolist()
        neg_classifications_class += new_neg_classifications_class.tolist()

        pos_classifications_rel += new_pos_classifications_rel.tolist()
        neg_classifications_rel += new_neg_classifications_rel.tolist()

        
    total_acc = np.array([x for x in all_accs if x!=-1]).mean()
    total_f1 = np.array([x for x in all_f1s if x!=-1]).mean()

    return pos_classifications, neg_classifications, pos_classifications_class, neg_classifications_class, pos_classifications_rel, neg_classifications_rel, pos_predictions, neg_predictions, perfects, total_acc, total_f1

