import json
import torch
import numpy as np
import torch.nn.functional as F


torch.manual_seed(0)

def get_pred_loss(video_ids, embeddings, json_data_dict, mlp_dict, neg_weight, margin, device):

    #printout = np.random.rand()<0.05
    printout = np.random.rand()<0.00
    for batch_idx, video_id in enumerate(video_ids):
        dpoint = json_data_dict[int(video_id.item())]
        triples = dpoint['caption']
        ntriples = dpoint['negatives']
        if neg_weight == -1: 
            try:
                neg_weight = float(len(triples))/len(ntriples)
            except ZeroDivisionError:
                neg_weight = 0
        loss = torch.tensor([0.], device=device)
        for triple in triples:
            sub, relation, obj = triple.split()
            try:
                mlp = mlp_dict[relation].to(device)
            except KeyError:
                #print("Can't find mlp for relation {}".format(relation))
                continue
            sub_pos = dpoint['individuals'].index(sub)
            obj_pos = dpoint['individuals'].index(obj)
            sub_embedding = embeddings[batch_idx,sub_pos]
            obj_embedding = embeddings[batch_idx,obj_pos]
            sub_obj_concat = torch.cat([sub_embedding, obj_embedding])
            prediction = mlp(sub_obj_concat)
            #prediction = torch.min(prediction, torch.tensor([10], dtype=torch.float32, device=device))
            if printout:
                print('pos', prediction.item())
                print(F.relu(-prediction+margin).item())
            #loss += F.relu(-prediction+margin)**2
            loss += F.relu(-prediction+margin)

        if neg_weight == 0:
            return loss

        for ntriple in ntriples:
            sub, relation, obj = ntriple.split()
            try:
                mlp = mlp_dict[relation].to(device)
            except KeyError:
                #print("Can't find mlp for relation {}".format(relation))
                continue
            sub_pos = dpoint['individuals'].index(sub)
            obj_pos = dpoint['individuals'].index(obj)
            sub_embedding = embeddings[batch_idx, sub_pos]
            obj_embedding = embeddings[batch_idx, obj_pos]
            sub_obj_concat = torch.cat([sub_embedding, obj_embedding])
            mlp = mlp_dict[relation].to(device)
            prediction = mlp(sub_obj_concat)
            #prediction = torch.max(prediction, torch.tensor([-10], dtype=torch.float32, device=device))
            if printout:
                print('neg', prediction.item())
                print(F.relu(prediction+margin).item())
                print(neg_weight*F.relu(prediction+margin).item())
            #loss += neg_weight*F.relu(prediction+margin)**2
            loss += neg_weight*F.relu(prediction+margin)

    return loss
       

if __name__ == "__main__":

    with open('/data2/commons/rdf_video_captions/10d.dev.json', 'r') as f:
        json_data_dict = json.load(f)

    weight_dict = torch.load("mlp-weights.pickle")
    mlp_dict = {}
    for relation, weights in weight_dict.items():
        hidden_layer = nn.Linear(weights["hidden_weights"].shape[0], weights["hidden_bias"].shape[0])
        hidden_layer.weight = nn.Parameter(torch.FloatTensor(weights["hidden_weights"]), requires_grad=False)
        hidden_layer.bias = nn.Parameter(torch.FloatTensor(weights["hidden_bias"]), requires_grad=False)
        output_layer = nn.Linear(weights["output_weights"].shape[0], weights["output_bias"].shape[0])
        output_layer.weight = nn.Parameter(torch.FloatTensor(weights["output_weights"]), requires_grad=False)
        output_layer.bias = nn.Parameter(torch.FloatTensor(weights["output_bias"]), requires_grad=False)
        mlp_dict[relation] = nn.Sequential(hidden_layer, nn.ReLU(), output_layer, nn.Sigmoid()) 
