import sys
import argparse


def load_arguments():
    argparser = argparse.ArgumentParser(sys.argv[0])

    argparser.add_argument("--overwrite", 
            action="store_true",
            default=False,
            help = "whether to overwrite existing experiment of same name"
        )    
    argparser.add_argument("--loss_func", 
            type=str,
            choices=['mse', 'cos'],
            default='mse',
            help = "loss function to train on"
        )    
    argparser.add_argument("--dec_init", 
            type=str,
            choices=['zeroes', 'unit', 'learned', 'unit_learned'],
            default='zeroes',
            help = "how to initialize decoder rnn"
        )    
    argparser.add_argument("--enc_init", 
            type=str,
            choices=['zeroes', 'unit', 'learned', 'unit_learned'],
            default='zeroes',
            help = "how to initialize encoder rnn"
        )    
    #argparser.add_argument("--rrn_init", 
    #        type=str,
    #        choices=['det', 'rand'],
    #        default='det',
    #        help = "whether rnn embeddings are trained with deterministic or random inits"
    #    )    
    argparser.add_argument("--norm_threshold", 
            type=float,
            default=1.0,
            help = "value below which the norm is penalized"
        )    
    argparser.add_argument("--device", 
            type=str,
            default = 'cuda',
            help = "whether to train on gpu (cuda) or cpu"
        )    
    argparser.add_argument("--chkpt", 
            action="store_true",
            default = False,
            help = "whether to write a checkpoint"
        )    
    argparser.add_argument("--exp_name", 
            type=str,
            default = "",
            help = "name of this experiment, defaults to a timestamp"
        )    
    argparser.add_argument("--enc_cnn", 
            choices = ["vgg", "nasnet", "vgg_old"],
            default = "vgg",
            help = "which cnn to use as first part of encoder"
        )    
    argparser.add_argument("--dec_rnn", 
            choices = ["gru", "lstm"],
            default = "gru",
            help = "which rnn to use for the decoder"
        )    
    argparser.add_argument("--enc_rnn", 
            choices = ["gru", "lstm"],
            default = "gru",
            help = "which rnn to use for the encoder"
        )
    argparser.add_argument("--enc_dec_hidden_init", 
            action = "store_true",
            default = False,
            help = "whether to init decoder rnn hidden with encoder rnn hidden, otherwise zeroes"
        )
    argparser.add_argument("--reload_path", 
            default = None,
            help = "path of checkpoint to reload from, None means random init"
        )
    argparser.add_argument("--dec_size", 
            default = 1500,
            type=int,
            help = "number of units in decoder rnn"
        )
    argparser.add_argument("--enc_size", 
            default = 2000,
            type=int,
            help = "number of units in encoder rnn"
        )
    argparser.add_argument("--enc_layers", 
            default = 2,
            type=int,
            help = "number of layers in encoder rnn"
        )
    argparser.add_argument("--dec_layers", 
            default = 2,
            type=int,
            help = "number of layers in decoder rnn"
        )
    argparser.add_argument("--quick_run", "-q",
            default = False,
            action = "store_true",
            help = "whether to use mini-dataset, so doesn't exceed ram when running locally"
        )
    argparser.add_argument("--mini", "-m",
            default = False,
            action = "store_true",
            help = "whether to use mini-dataset, so doesn't exceed ram when running locally"
        )
    argparser.add_argument("--cnn_layers_to_freeze",
            type = int,
            default = 17,
            help = "how many of the CNN's layers to freeze during training"
        )
    argparser.add_argument("--weight_decay",
            type = int,
            default = 0,
            help = "optimzer"
        )
    argparser.add_argument("--optimizer",
            choices = ['SGD', 'Adam', 'RMS'],
            default = 'Adam',
            help = "optimzer"
        )
    argparser.add_argument("--model",
            choices = ['seq2seq', 'reg', 'eos'],
            default = 'seq2seq',
            help = "which subnetwork to train"
        )
    argparser.add_argument("--verbose",
            action = "store_true",
            default = False,
            help = "whether to print network info before starting training"
        )
    argparser.add_argument("--max_epochs",
            type = int,
            default = 1000,
            help = "maximum number of epochs"
        )
    argparser.add_argument("--dropout",
            type = float,
            default = 0.1,
            help = "dropout probability"
        )
    argparser.add_argument("--learning_rate",
            type = float,
            default = 1e-3,
            help = "learning rate"
        )
    argparser.add_argument("--num_frames",
            type = int,
            default = 8,
            help = "number of frames"
        )
    argparser.add_argument("--frame_width",
            type = int,
            default = 256,
            help = "width of a single frame"
        )
    argparser.add_argument("--frame_height",
            type = int,
            default = 256,
            help = "height of a single frame"
        )
    argparser.add_argument("--ind_size",
            type = int,
            default = 10,
            help = "size of the individuals embeddings"
        )
    argparser.add_argument("--teacher_forcing_ratio",
            type = float,
            default = 1.0,
            help = "teacher forcing ratio"
        )
    argparser.add_argument("--lmbda",
            type = float,
            default = 1.0,
            help = "scalar multiplying the norm loss"
        )
    argparser.add_argument("--batch_size",
            type = int,
            default = 100,
            help = "number of training examples in each batch"
        )
    argparser.add_argument("--shuffle",
            #type = bool,
            action = "store_false",
            default = True,
            help = "whether to shuffle that data at each epoch"
        )
    argparser.add_argument("--patience",
            type = int,
            default = 15,
            help = "number of epochs to allow without improvement before early-stopping"
        )
    argparser.add_argument("--output_cnn_size",
            type = int,
            default = 4096,
            help = "size of the output of the cnn layers"
        )





    args = argparser.parse_args()
    return args
