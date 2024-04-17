import argparse
import numpy as np
import pandas as pd
import random
from datetime import datetime
from tqdm import tqdm
import sys
from utils import *
import logging
import torch
from torch import nn, optim
import torch.nn.functional as F
from model import SCEModel

logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

def trainModel(model, train_dataloader, valid_dataloader, args):
    item_train_dataloader, expl_train_dataloader = train_dataloader["item"], train_dataloader["expl"]
    learning_rate = args.lr
    log_file = args.log_file
    epochs = args.epochs
    device = args.device
    save_file = args.save_file
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    enduration = 0
    prev_valid_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        avg_loss = 0
        for item_batch, expl_batch in tqdm(zip(item_train_dataloader, expl_train_dataloader), total=len(item_train_dataloader)):
            item_input, item_output = model.gather(item_batch, device)
            expl_input, expl_output = model.gather(expl_batch, device)
            optimizer.zero_grad()
            item_preds, expl_preds, loss_mi = model(item_input, item_output, expl_input, True) #(N, seqlen, nitems)
            loss_r = model.item_loss_fn(item_preds.view(-1, item_preds.size(-1)), item_output[:,:,0].view(-1))
            loss_e = model.expl_loss_fn(expl_preds.view(-1, expl_preds.size(-1)), expl_output.view(-1))
            loss = loss_r + model.lambda_value * loss_e + model.mu * loss_mi
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            avg_loss += loss.item()
        avg_loss /= len(item_train_dataloader)
        with open(log_file, "a") as f:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Get the current time
            f.write(f"Epoch {epoch+1}: [{current_time}] [lr: {learning_rate}] Loss = {avg_loss:.4f}\n")

        # checking learning rate
        current_valid_loss = validModel(model, valid_dataloader, device)
        if current_valid_loss > prev_valid_loss*1.005:
            learning_rate = learning_rate /2
            enduration += 1
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
        else:
            torch.save(model.state_dict(), save_file)
        prev_valid_loss = current_valid_loss
        if enduration  >= 5:
            break

def validModel(model, valid_dataloader, device):
    item_valid_dataloader, expl_valid_dataloader = valid_dataloader["item"], valid_dataloader["expl"]
    model.eval()
    with torch.no_grad():
        avg_loss = 0
        for item_batch, expl_batch in zip(item_valid_dataloader, expl_valid_dataloader):
            item_input, item_output = model.gather(item_batch, device)
            expl_input, expl_output = model.gather(expl_batch, device)
            item_preds, expl_preds = model(item_input, None, expl_input, False)
            if model.model_name == "LRURec":
                item_final_preds = item_preds[:,-1,:]
                expl_final_preds = expl_preds[:,-1,:]
            else:
                index = (item_input[:,:,0] != model.item_pad_idx).sum(-1)
                item_final_preds = torch.gather(item_preds, dim=1, index=index.view(-1, 1, 1).expand(-1, -1, item_preds.size(2))).squeeze()
                expl_final_preds = torch.gather(expl_preds, dim=1, index=index.view(-1, 1, 1).expand(-1, -1, expl_preds.size(2))).squeeze()  # (N, nitems)
            loss_r = model.item_loss_fn(item_final_preds, item_output.view(-1))
            loss_e = model.expl_loss_fn(expl_final_preds, expl_output.view(-1))
            loss =loss_r + model.lambda_value * loss_e
            avg_loss += loss.item()

        avg_loss /= len(item_valid_dataloader)
    return avg_loss



if __name__ == '__main__':
    random.seed(43)
    torch.manual_seed(43)

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', type=str, default='Amazon')
    argparser.add_argument('--test_mode', type=int, default=0)  
    argparser.add_argument('--log_file', type=str, default='log.txt')  
    argparser.add_argument('--embed_size', type=int, default=50)
    argparser.add_argument('--lr', type=float, default=0.001)
    argparser.add_argument('--epochs', type=int, default=50)
    argparser.add_argument('--nlayers', type=int, default=2)
    argparser.add_argument('--dropout_rate', type=float, default=0.5)
    argparser.add_argument('--save_file', type=str, default='model.pth')
    argparser.add_argument('--device', type=int, default=0)
    argparser.add_argument('--batch_size', type=int, default=512)
    argparser.add_argument('--lambda_value', type=float, default=0.5)
    argparser.add_argument('--alpha', type=float, default=0.9)
    argparser.add_argument('--mu', type=float, default=0.1)
    argparser.add_argument('--model_name', type=str, default='SASRec')  
    argparser.add_argument('--lower_bound', type=str, default='mine')
    argparser.add_argument('--negatives', type=str, default='all')



    args = argparser.parse_args()

    config = {
    "dataset": args.dataset,
    "test_mode": args.test_mode,
    "log_file": args.log_file,
    "embed_size": args.embed_size,
    "lr": args.lr,
    "epochs": args.epochs,
    "nlayers": args.nlayers,
    "dropout_rate": args.dropout_rate,
    "save_file": args.save_file,
    "device": args.device,
    "batch_size": args.batch_size,
    "lambda_value": args.lambda_value,
    "alpha": args.alpha,
    "mu": args.mu,
    "model_name": args.model_name,
    "lower_bound": args.lower_bound,
    "negatives": args.negatives, 
    "max_len": 50
    }

    with open(args.log_file, 'a') as log_file:
        arg_dict = vars(args)
        log_file.write(f"\nCurrent time: {datetime.now()}\n")
        for arg, value in arg_dict.items():
            log_file.write(f"--{arg} {value}\n")

    dataset = args.dataset
    data_path = './data/' + dataset + '/'
    if args.test_mode == 1:
        train_data = data_path + 'train_valid.csv'
        valid_data = data_path + 'valid.csv' # for early stopping
        test_data = data_path + 'test.csv'  # for evaluation
    else:
        train_data = data_path + 'train.csv'
        valid_data = data_path + 'valid.csv' # for early stopping.
        test_data = data_path + 'valid.csv'  # for tuning hyperparamters.
    train_df = pd.read_csv(train_data)
    valid_df = pd.read_csv(valid_data)
    test_df = pd.read_csv(test_data)
    user_groups = train_df.groupby('user_idx').agg({
    'item_idx': list,
    'exp_idx': list}).reset_index()
    user2action = {}
    for index, row in user_groups.iterrows():
        user_idx = row['user_idx']
        item_seq = row['item_idx']
        exp_seq = row['exp_idx']
        user2action[user_idx] = {'item_idx': item_seq, 'exp_idx': exp_seq}

    if args.test_mode == 1:
        input4test = pd.read_csv(data_path + 'train_valid.csv')
        user_groups = input4test.groupby('user_idx').agg({
            'item_idx': list,
            'exp_idx': list}).reset_index()
        user2action_4test = {}
        for index, row in user_groups.iterrows():
            user_idx = row['user_idx']
            item_seq = row['item_idx']
            exp_seq = row['exp_idx']
            user2action_4test[user_idx] = {'item_idx': item_seq, 'exp_idx': exp_seq}
    else:
        user2action_4test = user2action

    nusers = train_df.user_idx.max() + 1
    nitems = train_df.item_idx.max() + 1
    nexpls = train_df.exp_idx.max() + 1
    item_pad_idx = nitems
    expl_pad_idx = nexpls

    config["nitems"] = nitems
    config["nexpls"] = nexpls
    config["item_pad_idx"] = item_pad_idx
    config["expl_pad_idx"] = expl_pad_idx
    if args.model_name == "LRURec":
        train_dataloader, valid_dataloader, test_dataloader = construct_data_for_LRURec(user2action, user2action_4test, valid_df, test_df, item_pad_idx, expl_pad_idx, batch_size=args.batch_size)
    else:
        train_dataloader, valid_dataloader, test_dataloader = construct_data(user2action, user2action_4test, valid_df, test_df, item_pad_idx, expl_pad_idx, batch_size=args.batch_size)
    model = SCEModel(config)
    model = model.to(args.device)
    trainModel(model, train_dataloader, valid_dataloader, args)

    ## evaluation.
    # prepare ground_truth
    users_ground_item = {}
    users_ground_expl = {}
    # {0:[], 1:[],...}
    for i in range(len(test_df)):
        user = test_df.iloc[i]["user_idx"] 
        item = test_df.iloc[i]["item_idx"]
        users_ground_item[user] = [item] # list
        expl = test_df.iloc[i]["exp_idx"]
        users_ground_expl[user] = [expl]

    # prepare predictions
    users_ranked_item = []
    users_ranked_expl = []
    model.load_state_dict(torch.load(args.save_file))
    model.to(args.device)
    model.eval()
    with torch.no_grad():
        item_test_dataloader, expl_test_dataloader = test_dataloader["item"], test_dataloader["expl"]
        for item_batch, expl_batch in zip(item_test_dataloader, expl_test_dataloader):
            item_input = model.gather(item_batch, args.device, test=True)
            expl_input = model.gather(expl_batch, args.device, test=True)
            ranked_items, ranked_expls = model.rank_action(item_input, expl_input)
            ranked_items = ranked_items[:,:100]
            ranked_expls = ranked_expls[:,:100]
            users_ranked_item.extend(ranked_items.tolist())
            users_ranked_expl.extend(ranked_expls.tolist())

    users_ranked_item = {k: v for k, v in zip(range(len(users_ranked_item)), users_ranked_item)}
    users_ranked_expl = {k: v for k, v in zip(range(len(users_ranked_expl)), users_ranked_expl)}

    k = 5
    recommendation_score = compute_k(users_ground_item, users_ranked_item, k)
    explanation_score = compute_k(users_ground_expl, users_ranked_expl, k)
    with open(args.log_file, "a") as f:
        f.write(f"\nCurrent time: {datetime.now()}\n")
        f.write(f"[recommendation]: recall@{k}: {recommendation_score[0]}, ndcg@{k}: {recommendation_score[1]} \n")
        f.write(f"[explanation]: recall@{k}: {explanation_score[0]}, ndcg@{k}: {explanation_score[1]} \n")

    k = 10
    recommendation_score = compute_k(users_ground_item, users_ranked_item, k)
    explanation_score = compute_k(users_ground_expl, users_ranked_expl, k)
    with open(args.log_file, "a") as f:
        f.write(f"\nCurrent time: {datetime.now()}\n")
        f.write(f"[recommendation]: recall@{k}: {recommendation_score[0]}, ndcg@{k}: {recommendation_score[1]} \n")
        f.write(f"[explanation]: recall@{k}: {explanation_score[0]}, ndcg@{k}: {explanation_score[1]} \n")


