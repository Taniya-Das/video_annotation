"""Compute scores for results from a specified experiment. Metrics computed are
tp,fp,fn,tn,f1,accuracy. Each are computed twice, once using a fixed threshold
of 0.5 and once using the best available threshold. Also computed are the
average probability assigned to positive facts and negative facts respectively.
"""

import json
import numpy as np
#from get_pred_ablation_comb import compute_probs_for_dataset # To perform ablation on combining technique
from get_pred_final_fullrun import compute_probs_for_dataset
from dl_utils.misc import check_dir
from utils import acc_f1_from_binary_confusion_mat


def compute_dset_fragment_scores(dl, encoder, multiclassifier, multiclassifier_class, multiclassifier_rel, dataset_dict, fragment_name, ARGS, threshold, threshold_VG, attribute_stats, relationship_stats, sub_count_c, sub_count_r, model):
    """Compute performance metrics for a train/val/test dataset fragment. First
    executes forward pass of network to get outputs corresponding to true and
    false individuals and predicates; then thresholds and computes metrics.
    """
    
    acc, f1, pos, neg = compute_probs_for_dataset(dl, encoder, multiclassifier, multiclassifier_class, multiclassifier_rel, dataset_dict, ARGS.i3d, ARGS.dataset, threshold, threshold_VG, fragment_name, attribute_stats, relationship_stats, sub_count_c, sub_count_r, model, ARGS.VG)

    return acc, f1, pos, neg 


def compute_scores_for_thresh(positive_probs, negative_probs, thresh):
    tp = len([p for p in positive_probs if p>thresh])
    fp = len([p for p in negative_probs if p>thresh])
    fn = len([p for p in positive_probs if p<thresh])
    tn = len([p for p in negative_probs if p<thresh])

    acc, f1 = acc_f1_from_binary_confusion_mat(tp,fp,tn,fn)

    return tp, fp, fn, tn, f1, acc

def find_best_thresh_from_probs(positive_probs, negative_probs):
    """Compute accuracy and f1 by thresholding probabilities for positive and
    negative atoms. Use both a fixed threshold of 0.5 (reported in the paper)
    and also search for the threshold that gives the highest f1.
    """

    avg_pos_prob = sum(positive_probs)/len(positive_probs)
    avg_neg_prob = sum(negative_probs)/len(negative_probs)
    tphalf, fphalf, fnhalf, tnhalf, f1half, acchalf = compute_scores_for_thresh(positive_probs, negative_probs, 0.0)

    best_f1 = -1
    for thresh in np.linspace(avg_neg_prob, avg_pos_prob, num=10):
        tp, fp, fn, tn, f1, acc = compute_scores_for_thresh(positive_probs, negative_probs, thresh)
        if f1>best_f1:
            best_thresh = thresh
            best_tp, best_fp, best_fn, best_tn = tp, fp, fn, tn
            best_f1 = f1
            best_acc = acc

    return {'thresh': best_thresh,
            'tp':best_tp,
            'fp':best_fp,
            'fn':best_fn,
            'tn':best_tn,
            'f1':best_f1,
            'best_acc':best_acc,
            'tphalf':tphalf,
            'fphalf':fphalf,
            'fnhalf':fnhalf,
            'tnhalf':tnhalf,
            'f1half':f1half,
            'acchalf':acchalf,
            'avg_pos_prob':avg_pos_prob,
            'avg_neg_prob':avg_neg_prob
            }
