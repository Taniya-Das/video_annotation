from pdb import set_trace
import time
import utils
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import torch
import torch.nn as nn
from early_stopper import EarlyStopper
from get_pred import get_pred_loss


def train_on_batch(ARGS, training_example, encoder, multiclassifier, multiclassifier_class, multiclassifier_rel, dataset_dict, optimizer, criterion, device, train):
    input_tensor = training_example[0].float().transpose(0,1).to(device)
    multiclass_inds = training_example[1].float().to(device)
    multiclass_class = training_example[2].float().to(device)
    multiclass_rel = training_example[3].float().to(device)
    video_ids = training_example[4].to(device)
    i3d = training_example[5].float().to(device)
    if train:
        encoder.train()
        multiclassifier.train()
        multiclassifier_class.train()
        multiclassifier_rel.train()
    else:
        encoder.eval()
        multiclassifier.eval()
        multiclassifier_class.eval()
        multiclassifier_rel.eval()
    encoding, encoder_hidden = encoder(input_tensor)
    if ARGS.i3d: encoding = torch.cat([encoding,i3d],dim=-1)
    if ARGS.bottleneck: encoding = torch.randn_like(encoding)

    multiclassif_logits = multiclassifier(encoding)
    multiclass_loss = criterion(multiclassif_logits,multiclass_inds)

    multiclassifclass_logits = multiclassifier_class(encoding)
    multiclassclass_loss = criterion(multiclassifclass_logits,multiclass_class)

    multiclassifrel_logits = multiclassifier_rel(encoding)
    multiclassrel_loss = criterion(multiclassifrel_logits,multiclass_rel)

    pred_loss = get_pred_loss(video_ids,encoding,dataset_dict,testing=False)
    loss = multiclass_loss + ARGS.lmbda*pred_loss
    loss_class = multiclassclass_loss + ARGS.lmbda*pred_loss
    loss_rel = multiclassrel_loss + ARGS.lmbda*pred_loss

    if train: loss.backward(retain_graph=True); loss_class.backward(retain_graph=True); loss_rel.backward(); optimizer.step(); optimizer.zero_grad()
    return round(multiclass_loss.item(),5), round(multiclassclass_loss.item(),5), round(multiclassrel_loss.item(),5), round(pred_loss.item(),5)

def train(ARGS, encoder, multiclassifier, multiclassifier_class, multiclassifier_rel, dataset_dict, train_dl, val_dl, optimizer, exp_name, device, train):
    EarlyStop = EarlyStopper(patience=ARGS.patience)

    criterion = nn.BCEWithLogitsLoss()
    for epoch_num in range(ARGS.max_epochs):
        epoch_start_time = time.time()
        batch_train_multiclass_losses = []
        batch_train_multiclassclass_losses = []
        batch_train_multiclassrel_losses = []
        batch_train_pred_losses = []
        print("Epoch:", epoch_num+1)
        for iter_, training_example in enumerate(train_dl):
            #if iter_==0: print(training_example[4])
            new_train_multiclass_loss, new_train_multiclassclass_loss, new_train_multiclassrel_loss, new_train_pred_loss = train_on_batch(
                ARGS,
                training_example,
                encoder=encoder,
                multiclassifier=multiclassifier,
                multiclassifier_class=multiclassifier_class,
                multiclassifier_rel=multiclassifier_rel,
                dataset_dict=dataset_dict,
                optimizer=optimizer,
                criterion=criterion,
                device=device, train=True)
            print('Batch:', iter_, 'multiclass loss:', new_train_multiclass_loss, 'multiclass class loss:', new_train_multiclassclass_loss, 'mulriclass relations loss:', new_train_multiclassrel_loss, 'pred loss:', new_train_pred_loss)
            batch_train_multiclass_losses.append(new_train_multiclass_loss)
            batch_train_multiclassclass_losses.append(new_train_multiclassclass_loss)
            batch_train_multiclassrel_losses.append(new_train_multiclassrel_loss)
            batch_train_pred_losses.append(new_train_pred_loss)
            if ARGS.quick_run:
                break
        batch_val_multiclass_losses = []
        batch_val_multiclassclass_losses = []
        batch_val_multiclassrel_losses = []
        batch_val_pred_losses = []

        for iter_, valing_triplet in enumerate(val_dl):
            new_val_multiclass_loss, new_val_multiclassclass_loss, new_val_multiclassrel_loss, new_val_pred_loss = train_on_batch(ARGS, training_example, encoder=encoder, multiclassifier=multiclassifier, multiclassifier_class=multiclassifier_class, multiclassifier_rel=multiclassifier_rel, optimizer=None, criterion=criterion, dataset_dict=dataset_dict, device=device, train=False)

            batch_val_multiclass_losses.append(new_val_multiclass_loss)
            batch_val_multiclassclass_losses.append(new_val_multiclassclass_loss)
            batch_val_multiclassrel_losses.append(new_val_multiclassrel_loss)
            batch_val_pred_losses.append(new_val_pred_loss)

            if ARGS.quick_run:
                break

        try:
            epoch_val_multiclass_loss = sum(batch_val_multiclass_losses)/len(batch_val_multiclass_losses)
            epoch_val_multiclassclass_loss = sum(batch_val_multiclassclass_losses)/len(batch_val_multiclassclass_losses)
            epoch_val_multiclassrel_loss = sum(batch_val_multiclassrel_losses)/len(batch_val_multiclassrel_losses)
            epoch_val_pred_loss = sum(batch_val_pred_losses)/len(batch_val_pred_losses)

        except ZeroDivisionError:
            print("\nIt seems the batch size might be larger than the number of data points in the validation set\n")
        save_dict = {'encoder':encoder, 'multiclassifier':multiclassifier, 'multiclassifier_class':multiclassifier_class, 'multiclassifier_rel':multiclassifier_rel,'ind_dict': dataset_dict['ind_dict'], 'mlp_dict': dataset_dict['mlp_dict'], 'optimizer': optimizer}
        #save = not ARGS.no_chkpt and new_epoch_val_loss < 0.01 and random.random() < 0.1
        EarlyStop(epoch_val_multiclass_loss+epoch_val_multiclassclass_loss+epoch_val_multiclassrel_loss+epoch_val_pred_loss, save_dict, exp_name=exp_name, save=not ARGS.no_chkpt)

        print('val_multiclass_loss', epoch_val_multiclass_loss, 'val_multiclass_class_loss', epoch_val_multiclassclass_loss, 'val_multiclass_relation_loss', epoch_val_multiclassrel_loss, 'val_pred_loss', epoch_val_pred_loss)
        if EarlyStop.early_stop:
            break

        print(f'Epoch time: {utils.asMinutes(time.time()-epoch_start_time)}')

def train_diffepoch(ARGS, epoch, encoder, multiclassifier, multiclassifier_class, multiclassifier_rel, dataset_dict, train_dl, val_dl, optimizer, exp_name, device, train):
    EarlyStop = EarlyStopper(patience=ARGS.patience)

    criterion = nn.BCEWithLogitsLoss()
    for epoch_num in range(epoch):
        epoch_start_time = time.time()
        batch_train_multiclass_losses = []
        batch_train_multiclassclass_losses = []
        batch_train_multiclassrel_losses = []
        batch_train_pred_losses = []
        print("Epoch:", epoch_num+1)
        for iter_, training_example in enumerate(train_dl):
            #if iter_==0: print(training_example[4])
            new_train_multiclass_loss, new_train_multiclassclass_loss, new_train_multiclassrel_loss, new_train_pred_loss = train_on_batch(
                ARGS,
                training_example,
                encoder=encoder,
                multiclassifier=multiclassifier,
                multiclassifier_class=multiclassifier_class,
                multiclassifier_rel=multiclassifier_rel,
                dataset_dict=dataset_dict,
                optimizer=optimizer,
                criterion=criterion,
                device=device, train=True)
            print('Batch:', iter_, 'multiclass loss:', new_train_multiclass_loss, 'multiclass class loss:', new_train_multiclassclass_loss, 'mulriclass relations loss:', new_train_multiclassrel_loss, 'pred loss:', new_train_pred_loss)
            batch_train_multiclass_losses.append(new_train_multiclass_loss)
            batch_train_multiclassclass_losses.append(new_train_multiclassclass_loss)
            batch_train_multiclassrel_losses.append(new_train_multiclassrel_loss)
            batch_train_pred_losses.append(new_train_pred_loss)
            if ARGS.quick_run:
                break
        batch_val_multiclass_losses = []
        batch_val_multiclassclass_losses = []
        batch_val_multiclassrel_losses = []
        batch_val_pred_losses = []

        for iter_, valing_triplet in enumerate(val_dl):
            new_val_multiclass_loss, new_val_multiclassclass_loss, new_val_multiclassrel_loss, new_val_pred_loss = train_on_batch(ARGS, training_example, encoder=encoder, multiclassifier=multiclassifier, multiclassifier_class=multiclassifier_class, multiclassifier_rel=multiclassifier_rel, optimizer=None, criterion=criterion, dataset_dict=dataset_dict, device=device, train=False)

            batch_val_multiclass_losses.append(new_val_multiclass_loss)
            batch_val_multiclassclass_losses.append(new_val_multiclassclass_loss)
            batch_val_multiclassrel_losses.append(new_val_multiclassrel_loss)
            batch_val_pred_losses.append(new_val_pred_loss)

            if ARGS.quick_run:
                break

        try:
            epoch_val_multiclass_loss = sum(batch_val_multiclass_losses)/len(batch_val_multiclass_losses)
            epoch_val_multiclassclass_loss = sum(batch_val_multiclassclass_losses)/len(batch_val_multiclassclass_losses)
            epoch_val_multiclassrel_loss = sum(batch_val_multiclassrel_losses)/len(batch_val_multiclassrel_losses)
            epoch_val_pred_loss = sum(batch_val_pred_losses)/len(batch_val_pred_losses)

        except ZeroDivisionError:
            print("\nIt seems the batch size might be larger than the number of data points in the validation set\n")
        save_dict = {'encoder':encoder, 'multiclassifier':multiclassifier, 'multiclassifier_class':multiclassifier_class, 'multiclassifier_rel':multiclassifier_rel,'ind_dict': dataset_dict['ind_dict'], 'mlp_dict': dataset_dict['mlp_dict'], 'optimizer': optimizer}
        #save = not ARGS.no_chkpt and new_epoch_val_loss < 0.01 and random.random() < 0.1
        EarlyStop(epoch_val_multiclass_loss+epoch_val_multiclassclass_loss+epoch_val_multiclassrel_loss+epoch_val_pred_loss, save_dict, exp_name=exp_name, save=not ARGS.no_chkpt)

        print('val_multiclass_loss', epoch_val_multiclass_loss, 'val_multiclass_class_loss', epoch_val_multiclassclass_loss, 'val_multiclass_relation_loss', epoch_val_multiclassrel_loss, 'val_pred_loss', epoch_val_pred_loss)
        if EarlyStop.early_stop:
            break

        print(f'Epoch time: {utils.asMinutes(time.time()-epoch_start_time)}')


def train_on_batch_onlyMLP(ARGS, training_example, encoder, dataset_dict, optimizer, criterion, device, train):
    input_tensor = training_example[0].float().transpose(0,1).to(device)
    multiclass_inds = training_example[1].float().to(device)
    video_ids = training_example[4].to(device)
    i3d = training_example[5].float().to(device)

    encoding, encoder_hidden = encoder(input_tensor)
    if ARGS.i3d: encoding = torch.cat([encoding,i3d],dim=-1)
    if ARGS.bottleneck: encoding = torch.randn_like(encoding)

    pred_loss = get_pred_loss(video_ids,encoding,dataset_dict,testing=False)

    return round(pred_loss.item(),5)

def train_onlyMLP(ARGS, epoch, encoder, dataset_dict, train_dl, val_dl, optimizer, exp_name, device, train):
    EarlyStop = EarlyStopper(patience=ARGS.patience)

    criterion = nn.BCEWithLogitsLoss()
    for epoch_num in range(epoch):
        epoch_start_time = time.time()
        batch_train_pred_losses = []
        print("Epoch:", epoch_num+1)
        for iter_, training_example in enumerate(train_dl):
            #if iter_==0: print(training_example[4])
            new_train_pred_loss = train_on_batch_onlyMLP(
                ARGS,
                training_example,
                encoder=encoder,
                dataset_dict=dataset_dict,
                optimizer=optimizer,
                criterion=criterion,
                device=device, train=True)
            print('Batch:', iter_, 'pred loss:', new_train_pred_loss)
            batch_train_pred_losses.append(new_train_pred_loss)
            if ARGS.quick_run:
                break

        batch_val_pred_losses = []

        for iter_, valing_triplet in enumerate(val_dl):
            new_val_pred_loss = train_on_batch_onlyMLP(ARGS, training_example, encoder=encoder, optimizer=None, criterion=criterion, dataset_dict=dataset_dict, device=device, train=False)
            batch_val_pred_losses.append(new_val_pred_loss)

            if ARGS.quick_run:
                break

        try:
            epoch_val_pred_loss = sum(batch_val_pred_losses)/len(batch_val_pred_losses)

        except ZeroDivisionError:
            print("\nIt seems the batch size might be larger than the number of data points in the validation set\n")
        save_dict = {'mlp_dict': dataset_dict['mlp_dict'], 'optimizer': optimizer}
        #save = not ARGS.no_chkpt and new_epoch_val_loss < 0.01 and random.random() < 0.1
        EarlyStop(epoch_val_pred_loss, save_dict, exp_name=exp_name, save=not ARGS.no_chkpt)

        print('val_pred_loss', epoch_val_pred_loss)
        if EarlyStop.early_stop:
            break

        print(f'Epoch time: {utils.asMinutes(time.time()-epoch_start_time)}')