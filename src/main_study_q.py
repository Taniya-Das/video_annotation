"""
This studies the effect to changing q on F1 score and time-taken for MSVD* and MSR-VTT* dataset.
The model is fully trained and reloaded from saved checkpoint. The prediction only uses predicate-MLPs for
output as it is an investigative study.
"""

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


def main(args):
    print(args)

    global LOAD_START_TIME; LOAD_START_TIME = time()
    if args.mini:
        args.exp_name = 'try'
    exp_name = get_datetime_stamp() if args.exp_name == "" else args.exp_name
    exp_dir = os.path.join('../experiments/',exp_name)
    check_dir(exp_dir)
    
    w2v_path = os.path.join(args.data_dir,'w2v_vecs.bin')
    if args.mini:
        splits = [4,6,11]
        json_path = f"{args.dataset}_10dp.json"
        args.batch_size = min(2, args.batch_size)
        args.enc_size, args.dec_size = 50, 51
        args.enc_layers = args.dec_layers = 1
        args.no_chkpt = True
        if args.max_epochs == 1000:
            args.max_epochs = 1
        w2v = KeyedVectors.load_word2vec_format(w2v_path,binary=True,limit=2000)
    else:
        splits = [1200,1300,1970] if args.dataset=='MSVD' else [6517,7010,10000]
        json_path = f"{args.dataset}_final.json"
        print('Loading w2v model...')
        w2v = KeyedVectors.load_word2vec_format(w2v_path,binary=True,limit=args.w2v_limit)
    with open(json_path) as f: json_data=json.load(f)

    inds,classes,relations,json_data_list = json_data['inds'],json_data['classes'],json_data['relations'],json_data['dataset']
    
    for dp in json_data_list:
        dp['pruned_atoms_with_synsets'] = [tuplify(a) for a in dp['pruned_atoms_with_synsets']]
        dp['lcwa'] = [tuplify(a) for a in dp['lcwa']]
    json_data_dict = {dp['video_id']:dp for dp in json_data_list}
    video_data_dir = os.path.join(args.data_dir,args.dataset)
    train_dl, val_dl, test_dl = data_loader.get_split_dls(json_data_list,splits,args.batch_size,args.shuffle,args.i3d,video_data_dir=video_data_dir)
    print(next(iter(train_dl))[1].shape, next(iter(val_dl))[1].shape, next(iter(test_dl))[1].shape)
    
    print('Initializing new networks...')
    encoder = my_models.EncoderRNN(args, args.device).to(args.device)
    encoding_size = args.enc_size + 4096 if args.i3d else args.enc_size
    multiclassifier = my_models.MLP(encoding_size,args.classif_size,len(inds)).to(args.device)
    multiclassifier_class = my_models.MLP(encoding_size, args.classif_size, len(classes)).to(args.device)
    multiclassifier_rel = my_models.MLP(encoding_size, args.classif_size, len(relations)).to(args.device)
        
    mlp_dict = {}
    class_dict = {c[1]: my_models.MLP(encoding_size + args.ind_size,args.mlp_size,1).to(args.device) for c in classes}
    relation_dict = {r[1]: my_models.MLP(encoding_size + 2*args.ind_size,args.mlp_size,1).to(args.device) for r in relations}
    # order the dicts so can lookup by index at inference
    mlp_dict = OrderedDict({'classes':class_dict, 'relations':relation_dict})
    ind_dict = OrderedDict({ind[1]: torch.nn.Parameter(torch.tensor(get_w2v_vec(ind[0],w2v),device=args.device,dtype=torch.float32)) for ind in inds})
    dataset_dict = {'dataset':json_data_dict,'ind_dict':ind_dict,'mlp_dict':mlp_dict} 

    if args.dataset == 'MSVD':
        print("\nLoading trained model from MSVD_full_.pt")
        checkpoint_path ='../data/checkpoints/MSVD_full_.pt'
        checkpoint = torch.load(checkpoint_path)
        encoder = checkpoint['encoder']
        multiclassifier = checkpoint['multiclassifier']
        multiclassifier_class = checkpoint['multiclassifier_class']
        multiclassifier_rel = checkpoint['multiclassifier_rel']
        dataset_dict['ind_dict'] = checkpoint['ind_dict']
        dataset_dict['mlp_dict'] = checkpoint['mlp_dict']

        train_threshold = 0.45 # This value gave best performance in train set

    else:
        print("\nLoading trained model from MSRVTT_full_.pt")
        checkpoint_path ='../data/checkpoints/MSRVTT_full_.pt'
        checkpoint = torch.load(checkpoint_path)
        encoder = checkpoint['encoder']
        multiclassifier = checkpoint['multiclassifier']
        multiclassifier_class = checkpoint['multiclassifier_class']
        multiclassifier_rel = checkpoint['multiclassifier_rel']
        dataset_dict['ind_dict'] = checkpoint['ind_dict']
        dataset_dict['mlp_dict'] = checkpoint['mlp_dict']

        train_threshold = 0.5 # This value gave best performance in train set


    encoder.batch_size=1
    train_dl, val_dl, test_dl = data_loader.get_split_dls(json_data['dataset'],splits,batch_size=1,shuffle=False,i3d=args.i3d,video_data_dir=video_data_dir)

    for top in [200,1000,3000,6000,9000,12000,30000]:
        test_time = time()
        test_acc, test_f1, test_pos, test_neg = compute_dset_fragment_scores(test_dl,encoder,multiclassifier,multiclassifier_class,multiclassifier_rel,dataset_dict,'test',args, train_threshold)
        print("Test Eval Time:",{asMinutes(time()-test_time)})

    summary_filename = os.path.join(exp_dir,'{}_summary.txt'.format(exp_name, exp_name))

    with open(summary_filename, 'w') as summary_file:
        summary_file.write('Final Scores for Test set\n')
        summary_file.write('\n\n')
        summary_file.write(f'With Visual Genome: {args.VG}\n')
        summary_file.write(f'train threshold: {train_threshold},  train_threshold_VG: {train_threshold_VG}\n\n')
        if args.VG:
            summary_file.write(f'LR weights [pred-MLP, KB]: {coeff}\n')
        summary_file.write(f'test acc: {test_acc}\n')
        summary_file.write(f'test f1: {test_f1}\n')
        summary_file.write(f'test positive accuracy: {test_pos}\n')
        summary_file.write(f'test negative accuracy: {test_neg}\n')
        summary_file.write(f'test time: {asMinutes(time()-test_time)}\n')
    
        summary_file.write('\n\nParameters:\n')
        for key in options.IMPORTANT_PARAMS:
            summary_file.write(str(key) + ": " + str(vars(args)[key]) + "\n")

    print(f'Total Time: {asMinutes(time()-LOAD_START_TIME)}')

def get_user_yesno_answer(question):
    answer = input(question+'(y/n)')
    if answer == 'y': return True
    elif answer == 'n': return False
    else:
        print("Please answer 'y' or 'n'")
        return(get_user_yesno_answer(question))

if __name__=="__main__":
    ARGS = options.load_arguments()

    import torch
    torch.manual_seed(ARGS.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ['CUDA_VISIBLE_DEVICES'] = ARGS.cuda_visible_devices
    import numpy as np
    np.random.seed(ARGS.seed)
    main(ARGS)
