import random
import re
import string
import time
import unicodedata
import utils

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, models, transforms


class EncoderRNN(nn.Module):
    def __init__(self, args, device):
        super(EncoderRNN, self).__init__()
        self.hidden_size = args.enc_size
        self.device = device
                
        #Use VGG, NOTE: param.requires_grad are set to True by default
        self.vgg = models.vgg19(pretrained=True)
        num_ftrs = self.vgg.classifier[6].in_features
        #Changes the size of VGG output to the GRU size
        self.vgg.classifier[6] = nn.Linear(num_ftrs, args.enc_size)

        self.gru = nn.GRU(args.enc_size, args.enc_size)

    def forward(self, input, hidden):
        embedded = self.vgg(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, args, device):
        super(AttnDecoderRNN, self).__init__()
        self.device = device
        self.hidden_size = args.ind_size
        self.output_size = args.ind_size
        self.dropout_p = args.dropout
        self.max_length = args.max_length

        self.attn = nn.Linear(self.hidden_size * 2, args.num_frames)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.output_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):

        drop_input = self.dropout(input.view(1, 1, -1))

        attn_weights = F.softmax(
            self.attn(torch.cat((drop_input[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((drop_input[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        #output = F.log_softmax(self.out(output[0]), dim=1)

        output = self.out(output[0])

        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)


class NumIndRegressor(nn.Module):
    def __init__(self, args, device):
        super(NumIndRegressor, self).__init__()
        self.device = device
        self.output_size = 1
        self.input_size = args.ind_size
        self.out = nn.Linear(self.input_size, self.output_size)

    # The input is supposed to be all the outputs of the encoder
    def forward(self, input):
        emb_sum = torch.sum(input, dim=0)
        output = self.out(emb_sum)
        return output


def train(args, input_tensor, target_tensor, target_number, encoder, decoder, regressor, encoder_optimizer, decoder_optimizer, regressor_optimizer, dec_criterion, reg_criterion):
    
    teacher_forcing_ratio = args.teacher_forcing_ratio

    encoder_hidden = encoder.initHidden()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    regressor_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(args.num_frames, encoder.hidden_size, device=encoder.device)
    
    dec_loss = 0
    reg_loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    #decoder_input = torch.tensor([[SOS_token]], device=device)
    decoder_input = torch.zeros(args.ind_size, device=decoder.device).cuda()
    decoder_hidden = encoder_hidden

    if torch.cuda.is_available():
        encoder_hidden = encoder_hidden.cuda()
        encoder_outputs = encoder_outputs.cuda()


    regressor_output = regressor(encoder_outputs)
    reg_loss += reg_criterion(regressor_output, target_number)

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    #REMOVE WHEN FIGURED OUT HOW TO PREDICT NUMBER!!!
    #use_teacher_forcing = True

    if use_teacher_forcing:

        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            dec_loss += dec_criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing
 
    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(int(regressor_output.item())):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            
            d_o = decoder_output
            decoder_input = d_o.detach()  # detach from history as input

            dec_loss += dec_criterion(decoder_output, target_tensor[di])
            

    loss = dec_loss + args.lmbda * reg_loss
    
    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()
    regressor_optimizer.step()

    return loss.item() / target_length


def tensorsFromTriplets(triplets, args):
	# given a couple video + set of embeddings of all individuals it returns a triplet:
	# (input_tensor, output_tensor, output_number)
	# where:
	# - input_tensor (num_frames, 1, 3, 224, 224)
	# - output_tensor (num_individuals, embedding_size)
    # - output_number (1)
	return (torch.rand(args.num_frames, 1, 3, 224, 224), torch.rand(args.max_length, args.ind_size), torch.rand(1))


def trainIters(args, encoder, decoder, regressor, print_every=1000, plot_every=1000):

    start = time.time()
    learning_rate = args.learning_rate
    n_iters = args.max_epochs

    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    regressor_optimizer = optim.SGD(regressor.parameters(), lr=learning_rate)

    triplets = "dummy"

    training_triplets = [tensorsFromTriplets(random.choice(triplets), args)
                      for i in range(n_iters)]
    dec_criterion = nn.MSELoss()
    reg_criterion = nn.MSELoss()

    #TO DO: add early stopping, add epochs and shuffle after epochs
    for iter in range(1, n_iters + 1):
        training_triplet = training_triplets[iter - 1]
        input_tensor = training_triplet[0]
        target_tensor = training_triplet[1]
        target_number = training_triplet[2]

        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()
            target_tensor = target_tensor.cuda()
            target_number = target_number.cuda()
        loss = train(args, input_tensor, target_tensor, target_number, encoder,
                     decoder, regressor, encoder_optimizer, decoder_optimizer, regressor_optimizer, dec_criterion, reg_criterion)

        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (utils.timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    utils.showPlot(plot_losses)


