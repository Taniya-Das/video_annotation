import csv
import sys
import os
import json
import utils
import options
import models
import train
import data_loader 
from get_output import write_outputs_get_info


#torch.manual_seed(0)

def run_experiment(exp_name, ARGS, train_table, val_table, test_table, i3d_train_table, i3d_val_table, i3d_test_table):
    """Cant' just pass generators as need to re-init with batch_size=1 when testing.""" 
    
    dataset = '{}-{}d'.format(ARGS.dataset, ARGS.ind_size)

    if ARGS.mini:
        ARGS.batch_size = min(2, ARGS.batch_size)
        ARGS.enc_size = 51
        ARGS.dec_size = 50
        ARGS.enc_layers = ARGS.dec_layers = 1
        ARGS.no_chkpt = True
        if ARGS.max_epochs == 1000:
            ARGS.max_epochs = 1
        train_file_path = val_file_path = test_file_path = '/data1/louis/data/rdf_video_captions/{}-{}d-6dp.h5'.format(ARGS.dataset, ARGS.ind_size)
        assert os.path.isfile(train_file_path), f"No train file found at {train_file_path}"
        print('Using dataset at: {}'.format(train_file_path))
    else:
        train_file_path = os.path.join('/data1/louis/data/rdf_video_captions', '{}-train.h5'.format(dataset))
        val_file_path = os.path.join('/data1/louis/data/rdf_video_captions', '{}-val.h5'.format(dataset))
        test_file_path = os.path.join('/data1/louis/data/rdf_video_captions', '{}-test.h5'.format(dataset))
        assert os.path.isfile(train_file_path)
        print('Using dataset at: {}'.format(train_file_path))

    train_generator = data_loader.load_data_lookup(train_file_path, video_lookup_table=train_table, i3d_lookup_table=i3d_train_table, batch_size=ARGS.batch_size, shuffle=ARGS.shuffle)
    val_generator = data_loader.load_data_lookup(val_file_path, video_lookup_table=val_table, i3d_lookup_table=i3d_val_table, batch_size=ARGS.batch_size, shuffle=ARGS.shuffle)
     
    print(ARGS)
   
    encoder = None
    encoder_optimizer = None
    decoder = None
    decoder_optimizer = None
    transformer = None
    transformer_optimizer = None
    regressor = None
    regressor_optimizer = None

    if ARGS.setting in ['embeddings', 'preds', 'eos']:
        if ARGS.reload:
            if exp_name.startswith('jade'):
                reload_file_path= '../jade_checkpoints/{}.pt'.format(ARGS.reload)
            else:
                #reload_file_path = '/data2/louis/checkpoints/{}.pt'.format(ARGS.reload)
                reload_file_path = '/data1/louis/checkpoints/{}.pt'.format(ARGS.reload)
            reload_file_path = ARGS.reload
            print('Reloading model from {}'.format(reload_file_path))
            saved_model = torch.load(reload_file_path)
            encoder = saved_model['encoder']
            decoder = saved_model['decoder']
            encoder.batch_size = ARGS.batch_size
            decoder.batch_size = ARGS.batch_size
            encoder_optimizer = saved_model['encoder_optimizer']
            decoder_optimizer = saved_model['decoder_optimizer']
            if ARGS.setting == 'eos' and not ARGS.eos_reuse_decoder:
                decoder = models.DecoderRNN_openattn(ARGS).to(ARGS.device)
                decoder_optimizer = None
        else: 
            encoder = models.EncoderRNN(ARGS, ARGS.device).to(ARGS.device)
            #decoder = models.DecoderRNN(ARGS, ARGS.device).to(ARGS.device)
            decoder = models.DecoderRNN_openattn(ARGS).to(ARGS.device)
            encoder_optimizer = None
            decoder_optimizer = None
      
    elif ARGS.setting == 'transformer':
            transformer = RegTransformer(4096,10, num_layers=ARGS.transformer_layers, num_heads=ARGS.transformer_heads)
            transformer = torch.nn.DataParallel(transformer, device_ids=[0,1])
    print('\nTraining the model')
    train_info, _ = train.train(ARGS, encoder, decoder, transformer, train_generator=train_generator, val_generator=val_generator, exp_name=exp_name, device=ARGS.device, encoder_optimizer=encoder_optimizer, decoder_optimizer=decoder_optimizer)
       
    train_generator = data_loader.load_data_lookup(train_file_path, video_lookup_table=train_table, i3d_lookup_table=i3d_train_table, batch_size=1, shuffle=False)
    val_generator = data_loader.load_data_lookup(val_file_path, video_lookup_table=val_table, i3d_lookup_table=i3d_val_table, batch_size=1, shuffle=False)
    test_generator = data_loader.load_data_lookup(test_file_path, video_lookup_table=test_table, i3d_lookup_table=i3d_test_table, batch_size=1, shuffle=False)

      
    if ARGS.no_chkpt:
        print("\nUsing final (likely overfit) version of network for outputs because no checkpoints were saved")
    else:
        print("Reloading best network version for outputs")
        if exp_name.startswith('jade'):
            filename = '../jade_checkpoints/{}.pt'.format(exp_name[:-2])
        else:
            filename = '/data1/louis/checkpoints/{}.pt'.format(exp_name)
        print(filename)
        checkpoint = torch.load(filename)
        encoder = checkpoint['encoder']
        decoder = checkpoint['decoder']

    gt_forcing = (ARGS.setting == 'eos')
    #gt_forcing = False
    print('\nComputing outputs on val set')
    val_output_info = write_outputs_get_info(encoder, decoder, transformer, ARGS, gt_forcing=gt_forcing, data_generator=val_generator, exp_name=exp_name, dset_fragment='val', setting=ARGS.setting)
    val_output_info['dset_fragment'] = 'val'
    print('\nComputing outputs on train set')
    train_output_info = write_outputs_get_info(encoder, decoder, transformer, ARGS, gt_forcing=gt_forcing, data_generator=train_generator, exp_name=exp_name, dset_fragment='train', setting=ARGS.setting)
    train_output_info['dset_fragment'] = 'train'
    fixed_thresh = ((train_output_info['thresh']*1200)+(val_output_info['thresh']*100))/1300
    print('\nComputing outputs on test set')
    test_output_info = write_outputs_get_info(encoder, decoder, transformer, ARGS, gt_forcing=gt_forcing, data_generator=test_generator, exp_name=exp_name, dset_fragment='test', fixed_thresh=fixed_thresh, setting=ARGS.setting)
    test_output_info['dset_fragment'] = 'test'

    summary_filename = '../experiments/{}/{}_summary.txt'.format(exp_name, exp_name) 
    with open(summary_filename, 'w') as summary_file:
        summary_file.write('Experiment name: {}\n'.format(exp_name))
        summary_file.write('\tTrain\tVal\tTest\n')
        for k in ['dset_fragment', 'l2_distance', 'cos_similarity', 'thresh', 'legit_f1', 'best_acc', 'acchalf', 'f1half', 'avg_pos_prob', 'avg_neg_prob']: 
            summary_file.write(k+'\t'+str(train_output_info[k])+'\t'+str(val_output_info[k])+'\t'+str(test_output_info[k])+'\n')
        summary_file.write('\nParameters:\n')
        for key in options.IMPORTANT_PARAMS:
            summary_file.write(str(key) + ": " + str(vars(ARGS)[key]) + "\n")
    summary_csvfilename = '../experiments/{}/{}_summary.csv'.format(exp_name, exp_name) 
    fieldnames = list(sorted(vars(ARGS).keys())) + list(sorted(val_output_info.keys()))
    with open(summary_csvfilename, 'w') as csvfile:
        val_output_info.update(vars(ARGS))
        dictwriter = csv.DictWriter(csvfile, fieldnames=fieldnames)
        dictwriter.writeheader()
        dictwriter.writerow(val_output_info)


    print(val_output_info)
    return val_output_info


def get_user_yesno_answer(question):
    answer = input(question+'(y/n)')
    if answer == 'y':
        return True
    elif answer == 'n':
        return False
    else:
        print("Please answer 'y' or 'n'")
        return(get_user_yesno_answer(question))
    

def main():
    if ARGS.mini:
        ARGS.exp_name = 'try'
    
    exp_name = utils.get_datetime_stamp() if ARGS.exp_name == "" else ARGS.exp_name
    if not os.path.isdir('../experiments/{}'.format(exp_name)):
        os.mkdir('../experiments/{}'.format(exp_name))
    elif ARGS.exp_name == 'try' or ARGS.exp_name.startswith('jade'):
        pass
    else:
        try:
            overwrite = get_user_yesno_answer('An experiment with name {} has already been run, do you want to overwrite?'.format(exp_name))
        except OSError:
            overwrite = ARGS.overwrite
        if not overwrite:
            print('Please rerun command with a different experiment name')
            sys.exit()

    if ARGS.enc_dec_hidden_init and (ARGS.enc_size != ARGS.dec_size):
        print("Not applying enc_dec_hidden_init because the encoder and decoder are different sizes")
        ARGS.enc_dec_hidden_init = False

    if ARGS.mini:
        #train_table = val_table = test_table = data_loader.video_lookup_table_from_ids([1218,1337,1571,1443,1833,1874], cnn=ARGS.enc_cnn)
        if ARGS.dataset == 'MSVD':
            train_table = val_table = test_table = data_loader.video_lookup_table_from_range(1,7, dataset=ARGS.dataset)
        elif ARGS.dataset == 'MSRVTT':
            train_table = val_table = test_table = data_loader.video_lookup_table_from_range(0,6, dataset=ARGS.dataset)
        print(train_table.keys())
        if ARGS.i3d:
            i3d_train_table = i3d_val_table = i3d_test_table = data_loader.i3d_lookup_table_from_range(1,7)
        else:
            i3d_train_table = i3d_val_table = i3d_test_table = None
    else:
        print('\nLoading lookup tables\n')
        if ARGS.dataset == 'MSVD':
            train_table = data_loader.video_lookup_table_from_range(1,1201, dataset='MSVD')
            val_table = data_loader.video_lookup_table_from_range(1201,1301, dataset='MSVD')
            test_table = data_loader.video_lookup_table_from_range(1301,1971, dataset='MSVD')
        
        elif ARGS.dataset == 'MSRVTT':
            train_table = data_loader.video_lookup_table_from_range(0,6513, dataset='MSRVTT')
            val_table = data_loader.video_lookup_table_from_range(6513,7010, dataset='MSRVTT')
            test_table = data_loader.video_lookup_table_from_range(7010,10000, dataset='MSRVTT')
            #train_table=val_table=test_table=data_loader.video_lookup_table_from_range(6513,7010, dataset='MSRVTT')
        
        if ARGS.i3d:
            i3d_train_table = data_loader.i3d_lookup_table_from_range(1,1201)
            i3d_val_table = data_loader.i3d_lookup_table_from_range(1201,1301)
            i3d_test_table = data_loader.i3d_lookup_table_from_range(1301,1971)
        else:
            i3d_train_table = i3d_val_table = i3d_test_table = None
    run_experiment( exp_name, 
                    ARGS,
                    train_table=train_table,
                    val_table=val_table,
                    test_table=test_table,
                    i3d_train_table=i3d_train_table,
                    i3d_val_table=i3d_val_table,
                    i3d_test_table=i3d_test_table)


if __name__=="__main__":
    ARGS = options.load_arguments()

    import torch
    torch.manual_seed(ARGS.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ARGS.cuda_visible_devices
    import numpy as np
    np.random.seed(ARGS.seed)
    main()
