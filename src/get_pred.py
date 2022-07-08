import torch
import torch.nn.functional as F
from pdb import set_trace

torch.manual_seed(0)

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

def compute_probs_for_dataset(dl,encoder,multiclassifier,multiclassifier_class,multiclassifier_rel,dataset_dict,use_i3d):
    pos_classifications, neg_classifications, pos_classifications_class, neg_classifications_class, pos_classifications_rel, neg_classifications_rel, pos_predictions, neg_predictions, perfects = [],[],[],[],[],[],[],[],{}
    for d in dl:
        video_tensor = d[0].float().transpose(0,1).to('cuda')
        multiclass_inds = d[1].to('cuda')
        multiclass_class = d[2].to('cuda')
        multiclass_rel = d[3].to('cuda')
        #print(multiclass_inds)
        video_ids = d[4].to('cuda')
        i3d = d[5].float().to('cuda')
        encoding, enc_hidden = encoder(video_tensor)
        if use_i3d: encoding = torch.cat([encoding,i3d],dim=-1)
        multiclassif = multiclassifier(encoding)
        multiclassif_class = multiclassifier_class(encoding)
        multiclassif_rel = multiclassifier_rel(encoding)

        assert (multiclass_inds==1).sum() + (multiclass_inds==0).sum() == multiclass_inds.shape[1]
        assert (multiclass_class==1).sum() + (multiclass_class==0).sum() == multiclass_class.shape[1]
        assert (multiclass_rel==1).sum() + (multiclass_rel==0).sum() == multiclass_rel.shape[1]

        multiclass_inds = multiclass_inds.type(torch.bool)
        multiclass_class = multiclass_class.type(torch.bool)
        multiclass_rel = multiclass_rel.type(torch.bool)

        new_pos_classifications,new_neg_classifications = multiclassif[multiclass_inds], multiclassif[~multiclass_inds]
        new_pos_classifications_class,new_neg_classifications_class = multiclassif_class[multiclass_class], multiclassif_class[~multiclass_class]
        new_pos_classifications_rel,new_neg_classifications_rel = multiclassif_rel[multiclass_rel], multiclassif_rel[~multiclass_rel]

        new_pos_predictions, new_neg_predictions = get_pred_loss(video_ids, encoding, dataset_dict, testing=True)
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


    return pos_classifications, neg_classifications, pos_classifications_class, neg_classifications_class, pos_classifications_rel, neg_classifications_rel, pos_predictions, neg_predictions, perfects
