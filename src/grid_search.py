import os
import sys
import utils
import options
import models
import torch
import numpy as np
from data_loader import load_data, load_data_lookup, video_lookup_table_from_range


SUMMARY_DIR = '../data/logs'

def get_best_dev_file_path(dev):
    return os.path.join(SUMMARY_DIR, "best_on_device{}.txt".format(dev))


class HyperParamSet():
    def __init__(self, param_dict):
        self.ind_size = 50
        self.dec_size = param_dict['dec_size']
        self.num_frames = 8
        self.batch_size = param_dict['batch_size']
        self.learning_rate = param_dict['lr']
        self.optimizer = param_dict['opt']
        self.max_length = 29
        self.weight_decay = param_dict['weight_decay']
        self.dropout = 0
        self.shuffle = True
        self.max_epochs = 200
        self.patience = 5
        self.vgg_layers_to_freeze = 17
        self.quick_run = False
        

def train_with_hyperparams(model, train_table, val_table, param_dict, exp_name=None, best_val_loss=0, checkpoint_path=None, device="cuda"):

    print(device)
    args = HyperParamSet(param_dict)
    if exp_name == None:
        exp_name = utils.get_datetime_stamp()
    if mini:
        h5_train_generator = load_data_lookup('../data/mini/train_data.h5', video_lookup_table=train_table, batch_size=args.batch_size, shuffle=args.shuffle)
        h5_val_generator = load_data_lookup('../data/mini/val_data.h5', video_lookup_table=val_table, batch_size=args.batch_size, shuffle=args.shuffle)
    else:
        h5_train_generator = load_data_lookup('../data/rdf_video_captions/train_50d.h5', video_lookup_table=train_table, batch_size=args.batch_size, shuffle=args.shuffle)
        h5_val_generator = load_data_lookup('../data/rdf_video_captions/val_50d.h5', video_lookup_table=val_table, batch_size=args.batch_size, shuffle=args.shuffle)

    if model == 'seq2seq':
        encoder = models.EncoderRNN(args, device).to(device)
        decoder = models.DecoderRNN(args, device).to(device)
        val_loss = models.train_iters_seq2seq(args, encoder, decoder, train_generator=h5_train_generator, val_generator=h5_val_generator, exp_name=exp_name, device=device)
    elif model == 'reg':
        if checkpoint_path == None: 
            print("\nPath to the encoder weights checkpoints needed\n")
            sys.exit()
        checkpoint = torch.load(checkpoint_path)
        print("\n CHECKPOINT \n")
        print(checkpoint)
        encoder = checkpoint['encoder']
        decoder = checkpoint['decoder']
        regressor = models.NumIndRegressor(args, device).to(device)
        val_loss = models.train_iters_reg(args, encoder, decoder, regressor, train_generator=h5_train_generator, val_generator=h5_val_generator, exp_name=exp_name, device=device)
    elif model == 'eos':
        if checkpoint_path == None: 
            print("\nPath to the encoder weights checkpoints needed\n")
            sys.exit()
        checkpoint = torch.load(checkpoint_path)
        encoder = checkpoint['encoder']
        decoder = checkpoint['decoder']
        eos = models.NumIndEOS(args, device).to(device)
        val_loss = models.train_iters_eos(args, encoder, decoder, eos, train_generator=h5_train_generator, val_generator=h5_val_generator, exp_name=exp_name, device=device)

    summary_file_path = os.path.join(SUMMARY_DIR, "{}.txt".format(exp_name))
    summary_file = open(summary_file_path, 'w')
    summary_file.write("Val loss:" + ': ' + str(round(val_loss, 3)))
    for key in sorted(param_dict.keys()):
        summary_file.write(key + ': ' +  str(param_dict[key]))
   
    if val_loss < best_val_loss:
        best_dev_file_path = get_best_dev_file_path(device)
        with open(best_dev_file_path, 'w') as summary_file:
            summary_file.write("Val loss:" + ': ' + str(val_loss))
            for key in sorted(param_dict.keys()):
                summary_file.write(key + ': ' +  str(param_dict[key]))

    return val_loss, exp_name

def grid_search(model, dec_sizes, batch_sizes, lrs, opts, weight_decays, checkpoint_path=None):
    cuda_devs = ["cuda:{}".format(i) for i in range(torch.cuda.device_count())]
    print(cuda_devs)
    print("Available devices:")
    if cuda_devs == []:
        cuda_devs = ['cpu']
    for d in cuda_devs:
        print(d)
    if cuda_devs == []:
        cuda_devs = ['cpu']
    it = 0
 
    
    print("Loading video lookup tables..")
    if mini:
        train_table = video_lookup_table_from_range(1,11)
        val_table = video_lookup_table_from_range(1201,1211)
    else:
        train_table = video_lookup_table_from_range(1,1201)
        val_table = video_lookup_table_from_range(1201,1301)

    best_val_loss = float('inf')
    best_exp_name = None 
    for dec_size in dec_sizes:
        for batch_size in batch_sizes:
            for lr in lrs:
                for opt in opts:
                    for weight_decay in weight_decays:
                        param_dict = {
                            'dec_size': dec_size,
                            'batch_size': batch_size,
                            'lr': lr,
                            'opt': opt,
                            'weight_decay': weight_decay}
                        next_available_device = cuda_devs[it%len(cuda_devs)]
                        print("Executing run on {}".format(next_available_device))
                        new_val_loss, new_exp_name = train_with_hyperparams(model, train_table, val_table, param_dict, it, best_val_loss, checkpoint_path=checkpoint_path, device=next_available_device)
                        if new_val_loss < best_val_loss:
                            best_it = it
                            best_val_loss = new_val_loss
                            best_exp_name = new_exp_name
                        it += 1
    return best_exp_name

if __name__=="__main__":

    #dec_sizes = [1,2,3,4]
    dec_sizes = [4]
    batch_sizes = [64]
    lrs = [1e-5]
    opts = ['Adam']
    weight_decays = [0]

    if len(sys.argv) == 1:
        mini = False
        print("running grid search on full dataset")
    elif sys.argv[1] == 'm':
        mini = True
        print("running grid search on mini dataset")
    else:
        print("Unrecognized argument")
        sys.exit()

    best_exp_name = grid_search('seq2seq', dec_sizes, batch_sizes, lrs, opts, weight_decays)
    grid_search('eos', dec_sizes, batch_sizes, lrs, opts, weight_decays, checkpoint_path='../checkpoints/chkpt{}.pt'.format(best_exp_name))

    with open("checkpoit.out", 'w') as f:
        f.write('../checkpoints/chkpt{}.pt'.format(best_exp_name))

    ckpt_path = '../checkpoints/chkpt{}.pt'.format(best_exp_name)

    test_table = video_lookup_table_from_range(1301,1970)
    num_lines = 1900 - 1301
    h5_test_generator = load_data_lookup('../data/rdf_video_captions/test_50d.h5', video_lookup_table=test_table, batch_size=num_lines, shuffle=False)
    print(h5_test_generator)
    for t in h5_test_generator:
        print(t)
    models.get_test_output(ckpt_path, h5_test_generator[0], num_datapoints= num_lines, ind_size=50, use_eos=True, device='cuda')


