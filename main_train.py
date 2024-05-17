import os
import random
import argparse

import torch


import torch.nn as nn
import numpy as np
import torch.utils.data as Data
import net as net
from utils import load_data
from sklearn.metrics import f1_score
import pdb
import train_pruning as pruning
import copy
from scipy.sparse import coo_matrix
import warnings
warnings.filterwarnings('ignore')
import dgl
import scipy.sparse as sp
from pathlib import Path
from model.MLP import MLP
from utils_self_learning import *

class MLP(nn.Module):
    def __init__(self, input_size, n_class, n_hidden):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, n_hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(n_hidden, n_class)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def normalize_feature(mx):
    """Row-normalize sparse matrix or dense matrix
    Parameters
    ----------
    mx : scipy.sparse.csr_matrix or numpy.array
        matrix to be normalized
    Returns
    -------
    scipy.sprase.lil_matrix
        normalized matrix
    """
    if type(mx) is not sp.lil.lil_matrix:
        try:
            mx = mx.tolil()
        except AttributeError:
            pass
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    mx = torch.tensor(mx)
    return mx


def run_fix_mask(args, training_nodes_loss_weight, seed, topk_test_nodes_index, pseudo_labels, rewind_weight_mask):
    #######################################################
    #       this function use the modified graph          #
    #######################################################
    pruning.setup_seed(seed)
    #######################################################################################
    #   load the modified graph (here take the ptb_rate = 0.1 and seed = 0 as an example)
    #######################################################################################
    from pathlib import Path
    dataset_path = Path("./mod_graph/")
    output_graph = Path(dataset_path).joinpath(
        "ptb_graph",
        args["attack_name"],
        args["datasource"],
        f"ptb_rate_"+str(args["ptb_rate"]),
        args["dataset"],
    )

    with open(output_graph.joinpath("archive.txt"), 'r') as f:
        best_seed = f.readline().strip().split()[0]
    # set up the path with the best seed
    output_graph = output_graph.joinpath(f"seed_{best_seed}")
    print("Loading data from " + str(output_graph) + " ......")
    mod_data = dgl.data.CSVDataset(output_graph)
    mod_g = mod_data[0]

    #### get adj ####
    adj = mod_g.adjacency_matrix().to_dense()
    #### get train, val, and test sets ####
    idx_train = torch.nonzero(mod_g.ndata['train_mask']).squeeze().tolist()
    idx_val = torch.nonzero(mod_g.ndata['val_mask']).squeeze().tolist()
    idx_test = torch.nonzero(mod_g.ndata['test_mask']).squeeze().tolist()
    #### get features ####
    features_temp = mod_g.ndata["feat"]
    features = normalize_feature(features_temp)
    labels = mod_g.ndata["label"]


    adj = adj.cuda()
    features = features.cuda()
    labels = labels.cuda()
    pseudo_labels = pseudo_labels.cuda()
    loss_func = nn.CrossEntropyLoss()

    net_gcn = net.net_gcn(embedding_dim=args['embedding_dim'], adj=adj, features=features)
    pruning.add_mask(net_gcn)
    net_gcn = net_gcn.cuda()
    net_gcn.load_state_dict(rewind_weight_mask)
    adj_spar, wei_spar = pruning.print_sparsity(net_gcn) ### adj_spr - means how many edges are present.
    
    for name, param in net_gcn.named_parameters():
        if 'mask' in name:
            param.requires_grad = False

    optimizer = torch.optim.Adam(net_gcn.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    acc_test = 0.0
    best_val_acc = {'val_acc': 0, 'epoch' : 0, 'test_acc': 0}

    for epoch in range(args['total_epoch']):

        optimizer.zero_grad()
        output, loss_feat__ = net_gcn(features, adj)
        loss1 = loss_func(output[idx_train], labels[idx_train]) ### train nodes
        loss2 = loss_func(output[topk_test_nodes_index], pseudo_labels[topk_test_nodes_index]) ## test nodes topk
        
        if args['test_vs_traning_wei_opt']:
            print("alpha 1-alpha used")
            topk_test_nodes_loss_weight = 1 - training_nodes_loss_weight
            loss = training_nodes_loss_weight * loss1 + topk_test_nodes_loss_weight * loss2 #with iterations, training nodes are getting more importance and topk test nodes are getting less importance
        else:
            loss = args["alpha_fix_mask"]*loss1 + args['gamma_fix_mask']*loss2    
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            output, loss_feat_ = net_gcn(features, adj, val_test=True)
            acc_val = f1_score(labels[idx_val].cpu().numpy(), output[idx_val].cpu().numpy().argmax(axis=1), average='micro')
            acc_test = f1_score(labels[idx_test].cpu().numpy(), output[idx_test].cpu().numpy().argmax(axis=1), average='micro')
            if acc_val > best_val_acc['val_acc']:
                best_val_acc['val_acc'] = acc_val
                best_val_acc['test_acc'] = acc_test
                best_val_acc['epoch'] = epoch
 
        print("(Fix Mask) Epoch:[{}] Val:[{:.2f}] Test:[{:.2f}] | Final Val:[{:.2f}] Test:[{:.2f}] at Epoch:[{}]"
                 .format(epoch, acc_val * 100, acc_test * 100, 
                                best_val_acc['val_acc'] * 100, 
                                best_val_acc['test_acc'] * 100, 
                                best_val_acc['epoch']))

    return best_val_acc['val_acc'], best_val_acc['test_acc'], best_val_acc['epoch'], adj_spar, wei_spar

def run_get_mask(args, seed, imp_num, topk_test_nodes_index, pseudo_labels, rewind_weight_mask=None):
    
    print(args)
    pruning.setup_seed(seed)
    dataset_path = Path("./mod_graph/")
    output_graph = Path(dataset_path).joinpath(
        "ptb_graph",
        args["attack_name"],
        args["datasource"],
        f"ptb_rate_"+str(args["ptb_rate"]),
        args["dataset"],
        # f"seed_{seed}",
    )
    with open(output_graph.joinpath("archive.txt"), 'r') as f:
        best_seed = f.readline().strip().split()[0]
    output_graph = output_graph.joinpath(f"seed_{best_seed}")
    print("Loading data from " + str(output_graph) + " ......")
    mod_data = dgl.data.CSVDataset(output_graph)
    mod_g = mod_data[0]

    #### get adj ####
    adj = mod_g.adjacency_matrix().to_dense()
    #### get train, val, and test sets ####
    idx_train = torch.nonzero(mod_g.ndata['train_mask']).squeeze().tolist()
    idx_val = torch.nonzero(mod_g.ndata['val_mask']).squeeze().tolist()
    idx_test = torch.nonzero(mod_g.ndata['test_mask']).squeeze().tolist()
    #### get features ####
    features_temp = mod_g.ndata["feat"]
    features = normalize_feature(features_temp)
    ### No normalization
    #features = mod_g.ndata["feat"]
    #### get labels ####
    labels = mod_g.ndata["label"]

    adj = adj.cuda()
    features = features.cuda()
    labels = labels.cuda()
    pseudo_labels = pseudo_labels.cuda()
    
    
    loss_func = nn.CrossEntropyLoss()
    net_gcn = net.net_gcn(embedding_dim=args['embedding_dim'], adj=adj, features=features)  # the GNN is created. And for the GNN we also have the adj_mask created.
    pruning.add_mask(net_gcn)  ### creates the mask and initializes to 1 for the NN weights only.
    net_gcn = net_gcn.cuda()
    #net_gcn = net_gcn.to(device)
    for name, param in net_gcn.named_parameters():
        if param.requires_grad:
            print (name, param.data)


    if args['weight_dir']: 
        print("load : {}".format(args['weight_dir']))
        encoder_weight = {} ### for loading the weights.
        cl_ckpt = torch.load(args['weight_dir'], map_location='cuda')
        encoder_weight['weight_orig_weight'] = cl_ckpt['gcn.fc.weight'] 
        ori_state_dict = net_gcn.net_layer[0].state_dict()
        ori_state_dict.update(encoder_weight)
        net_gcn.net_layer[0].load_state_dict(ori_state_dict)  

    if rewind_weight_mask:
        net_gcn.load_state_dict(rewind_weight_mask)
        if not args['rewind_soft_mask'] or args['init_soft_mask_type'] == 'all_one':
            pruning.soft_mask_init(net_gcn, args['init_soft_mask_type'], seed)
        adj_spar, wei_spar = pruning.print_sparsity(net_gcn)
    else:
        pruning.soft_mask_init(net_gcn, args['init_soft_mask_type'], seed)  

   
    optimizer = torch.optim.Adam(net_gcn.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])

    acc_test = 0.0
    best_val_acc = {'val_acc': 0, 'epoch' : 0, 'test_acc':0}
    rewind_weight = copy.deepcopy(net_gcn.state_dict())  

    for epoch in range(args['total_epoch']):
        
        optimizer.zero_grad()
        output, loss_feat = net_gcn(features, adj)
        print('loss_feat', loss_feat)
        loss_ce = loss_func(output[idx_train], labels[idx_train])  
        loss_ce_test = loss_func(output[topk_test_nodes_index], pseudo_labels[topk_test_nodes_index])
        loss = args["alpha"]*loss_ce + args["beta"]*loss_feat + args["gamma"]*loss_ce_test

        loss.backward()  ####
        pruning.subgradient_update_mask(net_gcn, args) 
        optimizer.step()

        with torch.no_grad():
            output, loss_feat_ = net_gcn(features, adj, val_test=True)
            acc_val = f1_score(labels[idx_val].cpu().numpy(), output[idx_val].cpu().numpy().argmax(axis=1), average='micro')
            acc_test = f1_score(labels[idx_test].cpu().numpy(), output[idx_test].cpu().numpy().argmax(axis=1), average='micro')
            if acc_val > best_val_acc['val_acc']:
                best_val_acc['test_acc'] = acc_test
                best_val_acc['val_acc'] = acc_val
                best_val_acc['epoch'] = epoch
                best_epoch_mask = pruning.get_final_mask_epoch(net_gcn, adj_percent=args['pruning_percent_adj'], 
                                                                        wei_percent=args['pruning_percent_wei'])   ### adj and weight mask.

            print("(Get Mask) Epoch:[{}] Val:[{:.2f}] Test:[{:.2f}] | Best Val:[{:.2f}] Test:[{:.2f}] at Epoch:[{}]"
                 .format(epoch, acc_val * 100, acc_test * 100, 
                                best_val_acc['val_acc'] * 100,  
                                best_val_acc['test_acc'] * 100,
                                best_val_acc['epoch']))

    return best_epoch_mask, rewind_weight


def parser_loader():
    parser = argparse.ArgumentParser(description='GLT')
    ###### Unify pruning settings #######
    parser.add_argument('--s1', type=float, default=0.0001,help='scale sparse rate (default: 0.0001)')
    parser.add_argument('--s2', type=float, default=0.0001,help='scale sparse rate (default: 0.0001)')
    parser.add_argument('--total_epoch', type=int, default=300)
    parser.add_argument('--pruning_percent_wei', type=float, default=0.1)
    parser.add_argument('--pruning_percent_adj', type=float, default=0.1)
    parser.add_argument('--weight_dir', type=str, default='') ## the weight directory.
    parser.add_argument('--rewind_soft_mask', action='store_true')
    parser.add_argument('--init_soft_mask_type', type=str, default='', help='all_one, kaiming, normal, uniform')
    parser.add_argument('--dataset', type=str, default='citeseer')
    parser.add_argument('--embedding-dim', nargs='+', type=int, default=[3703,16,6])
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--attack_name', type=str, default='pgd_meta_adam', choices=['pgd_meta_adam', 'pgd', 'mettack'], help='attack types')
    parser.add_argument('--datasource', type=str, default='deeprobust_nettack_orilabel', choices=['planetoid_fakelabel', 'planetoid_orilabel', 'deeprobust_nettack_orilabel'], help='datasource')
    parser.add_argument('--ptb_rate', type=float, default=0.05,  help='pertubation rate')
    parser.add_argument('--beta', type=float, default=0.01,  help='coefficient for feature smoothness')
    parser.add_argument('--alpha', type=float, default=0.01,  help='coefficient for train nodes loss')
    parser.add_argument('--gamma', type=float, default=0.01,  help='coefficient for topk test nodes loss')
    parser.add_argument('--alpha_fix_mask', type=float, default=0.01,  help='coefficient for train nodes loss')
    parser.add_argument('--gamma_fix_mask', type=float, default=0.01,  help='coefficient for topk test nodes loss')
    parser.add_argument('--k', type=int, default=80, help='the number of pseudo labels')
    parser.add_argument('--test_vs_traning_wei_opt', action='store_true')
    return parser

def self_learning(args, seed):
    dataset_path = Path("./mod_graph/")
    output_graph = Path(dataset_path).joinpath(
        "ptb_graph",
        args["attack_name"],
        args["datasource"],
        f"ptb_rate_"+str(args["ptb_rate"]),
        args["dataset"],
        # f"seed_{seed}",
    )
    with open(output_graph.joinpath("archive.txt"), 'r') as f:
        best_seed = f.readline().strip().split()[0]
    # set up the path with the best seed
    output_graph = output_graph.joinpath(f"seed_{best_seed}")
    print("Loading data from " + str(output_graph) + " ......")

    mod_data = dgl.data.CSVDataset(output_graph)
    g = mod_data[0]

    perturbed_adj = g.adjacency_matrix()  # .to_dense()
    print("ptb number is : ", perturbed_adj.to_dense().nonzero().size(0))
    idx_train = torch.nonzero(g.ndata['train_mask']).squeeze().numpy()
    idx_val = torch.nonzero(g.ndata['val_mask']).squeeze().numpy()
    idx_test = torch.nonzero(g.ndata['test_mask']).squeeze().numpy()

    features = g.ndata["feat"]
    labels = g.ndata["label"]

    # Hyper-parameters
    epochs = 200
    n_hidden = 1024
    dropout = 0.5
    weight_decay = 5e-4
    lr = 1e-2
    loss = nn.CrossEntropyLoss()
    n_class = labels.max().item() + 1
    batch_size = 64

    train_dataset = Data.TensorDataset(features[idx_train], labels[idx_train])
    train_loader = Data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataset = Data.TensorDataset(features[idx_val], labels[idx_val])
    val_loader = Data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    test_dataset = Data.TensorDataset(features[idx_test], labels[idx_test])
    test_loader = Data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    model = MLP(features.shape[1], n_class, n_hidden)
    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


    acc = train_MLP(model, epochs, optimizer, train_loader, val_loader, test_loader, loss)
    print('Accuracy:%f' % acc)
    logits = model(features.cuda()).cpu()
    pseudo_labels = labels.clone()
    idx_train, pseudo_labels = get_psu_labels(logits, pseudo_labels, idx_train, idx_test, k=args["k"], append_idx=False)

    return idx_train, pseudo_labels 



if __name__ == "__main__":

    parser = parser_loader()
    args = vars(parser.parse_args())
    print(args)
    if args['rewind_soft_mask']:
        print('training and test nodes loss weight change is true')
    seed = random.randint(1, 10000)
    print(seed)
    rewind_weight = None  
    topk_test_nodes_index, pseudo_labels = self_learning(args, seed) 
    
    for p in range(20+1): 
        if p == 0:
            final_mask_dict, rewind_weight = run_get_mask(args, seed, p, topk_test_nodes_index, pseudo_labels, rewind_weight)

            best_acc_val, final_acc_test, final_epoch_list, adj_spar, wei_spar = run_fix_mask(args, p/20, seed, topk_test_nodes_index, pseudo_labels, rewind_weight) #p/20 = weight betweentopk test nodes loss and training nodes loss
            print("=" * 120)
            print("syd : Sparsity:[{}], Best Val:[{:.2f}] at epoch:[{}] | Final Test Acc:[{:.2f}] Adj:[{:.2f}%] Wei:[{:.2f}%]"
                .format(p + 1, best_acc_val * 100, final_epoch_list, final_acc_test * 100, adj_spar, wei_spar))
            print("=" * 120)
        else:
            final_mask_dict, rewind_weight = run_get_mask(args, seed, p, topk_test_nodes_index, pseudo_labels, rewind_weight)

            rewind_weight['adj_mask1_train'] = final_mask_dict['adj_mask']
            rewind_weight['adj_mask2_fixed'] = final_mask_dict['adj_mask']
            rewind_weight['net_layer.0.weight_mask_train'] = final_mask_dict['weight1_mask']
            rewind_weight['net_layer.0.weight_mask_fixed'] = final_mask_dict['weight1_mask']
            rewind_weight['net_layer.1.weight_mask_train'] = final_mask_dict['weight2_mask']
            rewind_weight['net_layer.1.weight_mask_fixed'] = final_mask_dict['weight2_mask']


            best_acc_val, final_acc_test, final_epoch_list, adj_spar, wei_spar = run_fix_mask(args, (p-1)/20, seed, topk_test_nodes_index, pseudo_labels, rewind_weight) #p/20 = weight between topk test nodes loss and training nodes loss
            print("=" * 120)
            print("syd : Sparsity:[{}], Best Val:[{:.2f}] at epoch:[{}] | Final Test Acc:[{:.2f}] Adj:[{:.2f}%] Wei:[{:.2f}%]"
                .format(p + 1, best_acc_val * 100, final_epoch_list, final_acc_test * 100, adj_spar, wei_spar))
            print("=" * 120)



