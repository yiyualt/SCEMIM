import argparse
import numpy as np
import pandas as pd
import random
from datetime import datetime
from tqdm import tqdm
import sys
sys.path.append('../../')  # Add the parent directory to the sys.path
from utils import *
import logging
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


def construct_data(train_df, valid_df, test_df, item_pad_idx, expl_pad_idx, attr_pad_idx, test_mode, batch_size, max_seq=25):
    """
    Note that train and test set are not affected by the argument "test_mode", only valid set are changed by test_mode.
    """
    user2itemlist = train_df.groupby('user_idx')['item_idx'].apply(list).to_dict()
    user2expllist = train_df.groupby('user_idx')['exp_idx'].apply(list).to_dict()
    user2attrlist = train_df.groupby('user_idx')['attribute_idx'].apply(list).to_dict()

    # Construct train data
    rec_input, rec_output, expl_input, expl_output, attr_seqs = [], [], [], [], []
    for user, item_actions in user2itemlist.items():
        expl_actions = user2expllist[user]
        attr_actions = user2attrlist[user]
        split_item_actions = [item_actions[i:i+max_seq] for i in range(0, len(item_actions), max_seq)]
        split_expl_actions = [expl_actions[i:i+max_seq] for i in range(0, len(expl_actions), max_seq)]
        split_attr_actions = [attr_actions[i:i+max_seq] for i in range(0, len(attr_actions), max_seq)]
        # Zip the split sequences and pad as necessary
        for item_seq, expl_seq, attr_seq in zip(split_item_actions, split_expl_actions, split_attr_actions):
            # Pad sequences if necessary
            if len(item_seq) < max_seq:
                item_seq = [item_pad_idx] * (max_seq - len(item_seq)) + item_seq
            if len(expl_seq) < max_seq:
                expl_seq = [expl_pad_idx] * (max_seq - len(expl_seq)) + expl_seq
            if len(attr_seq) < max_seq:
                attr_seq = [attr_pad_idx] * (max_seq - len(attr_seq)) + attr_seq 

            rec_input.append(list(zip(item_seq, expl_seq)))   # need shift for both
            rec_output.append(item_seq)
            expl_input.append(list(zip(item_seq, expl_seq)))   # need shift only for explanation
            expl_output.append(expl_seq)
            attr_seqs.append(attr_seq)

            
    rec_input = shift_right3d(torch.tensor(rec_input, dtype=torch.long), item_pad_idx, expl_pad_idx)    
    rec_output = torch.tensor(rec_output, dtype=torch.long)
    expl_input = shift_right3d(torch.tensor(expl_input, dtype=torch.long), item_pad_idx, expl_pad_idx)
    expl_input[:,:,0] = rec_output[:,:] 
    expl_output = torch.tensor(expl_output, dtype=torch.long)
    attr_seqs = torch.tensor(attr_seqs, dtype=torch.long) 
    expl_input = torch.cat((expl_input, attr_seqs.unsqueeze(-1)), dim=-1)

    # Combine item and explanation outputs
    train_dataset = TensorDataset(rec_input, rec_output, expl_input, expl_output)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # constrcut test.
    test_rec_input, test_rec_output, test_expl_input, test_expl_output, test_attr_seqs = [], [], [], [], []
    for user, item_actions in user2itemlist.items():
        target_item = test_df.loc[test_df['user_idx'] == user, 'item_idx'].values
        target_expl = test_df.loc[test_df['user_idx'] == user, 'exp_idx'].values
        if len(target_item) > 0 and len(target_expl) > 0:
            target_item = target_item[0]
            target_expl = target_expl[0]
        else:
            continue  

        expl_actions = user2expllist[user]
        attr_actions = user2attrlist[user]
        item_seq = item_actions[-max_seq:]
        expl_seq = expl_actions[-max_seq:]  # use the latest
        attr_seq = attr_actions[-max_seq:]
        if len(item_seq) < max_seq:         # padding
            item_seq = [item_pad_idx] * (max_seq - len(item_seq)) + item_seq
        if len(expl_seq) < max_seq:
            expl_seq = [expl_pad_idx] * (max_seq - len(expl_seq)) + expl_seq 
        if len(attr_seq) < max_seq:
            attr_seq = [attr_pad_idx] * (max_seq - len(attr_seq)) + attr_seq
        test_rec_input.append(list(zip(item_seq, expl_seq)))
        test_rec_output.append(target_item)
        test_expl_input.append(list(zip(item_seq[1:] + [target_item],  expl_seq)))
        test_expl_output.append(target_expl)
        test_attr_seqs.append(attr_seq)

    test_rec_input = torch.tensor(test_rec_input, dtype=torch.long)  # Shape: (all, seq_len)
    test_rec_output = torch.tensor(test_rec_output, dtype=torch.long)  # Shape: (all,)
    test_expl_input = torch.tensor(test_expl_input, dtype=torch.long)    # Shape: (all, seq_len)
    test_expl_output = torch.tensor(test_expl_output, dtype=torch.long)  # Shape: (all,)
    test_attr_seqs = torch.tensor(test_attr_seqs, dtype=torch.long) 
    test_expl_input = torch.cat((test_expl_input, test_attr_seqs.unsqueeze(-1)), dim=-1)
    test_dataset = TensorDataset(test_rec_input, test_rec_output, test_expl_input, test_expl_output)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # construct valid:
    if test_mode == 0:
        valid_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)  # Same as `test_dataset` when test_mode is False, since they are both from vlaid.csv
    elif test_mode == 1:
        valid_rec_input, valid_rec_output, valid_expl_input, valid_expl_output, valid_attr_seqs = [], [], [], [], []
        for user, item_actions in user2itemlist.items():
            target_item = valid_df.loc[valid_df['user_idx'] == user, 'item_idx'].values
            target_expl = valid_df.loc[valid_df['user_idx'] == user, 'exp_idx'].values
            if len(target_item) > 0 and len(target_expl) > 0:
                target_item = target_item[0]
                target_expl = target_expl[0]
            else:
                continue  
            expl_actions = user2expllist[user]
            attr_actions = user2attrlist[user]
            item_seq = item_actions[-max_seq-1:-1]  # Use the latest max_seq items, excluding the last one
            expl_seq = expl_actions[-max_seq-1:-1] 
            attr_seq = attr_actions[-max_seq-1:-1]
            if len(item_seq) < max_seq:
                item_seq = [item_pad_idx] * (max_seq - len(item_seq)) + item_seq
            if len(expl_seq) < max_seq:
                expl_seq = [expl_pad_idx] * (max_seq - len(expl_seq)) + expl_seq
            if len(attr_seq) < max_seq:
                attr_seq = [attr_pad_idx] * (max_seq - len(attr_seq)) + attr_seq
            valid_rec_input.append(list(zip(item_seq, expl_seq)))  
            valid_rec_output.append(target_item) 
            valid_expl_input.append(list(zip(item_seq[1:] + [target_item], expl_seq)))  
            valid_expl_output.append(target_expl) 
            valid_attr_seqs.append(attr_seq)
        valid_rec_input = torch.tensor(valid_rec_input, dtype=torch.long)       # Shape: (all, seq_len, 2)
        valid_rec_output = torch.tensor(valid_rec_output, dtype=torch.long)     # Shape: (all,)
        valid_expl_input = torch.tensor(valid_expl_input, dtype=torch.long)     # Shape: (all, seq_len, 2)
        valid_expl_output = torch.tensor(valid_expl_output, dtype=torch.long)   # Shape: (all,)
        valid_attr_seqs = torch.tensor(valid_attr_seqs, dtype=torch.long)
        valid_expl_input = torch.cat((valid_expl_input, valid_attr_seqs.unsqueeze(-1)), dim=-1)
        valid_dataset = TensorDataset(valid_rec_input, valid_rec_output, valid_expl_input, valid_expl_output)
        valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, valid_dataloader, test_dataloader


class SCEMIMa(nn.Module):
    def __init__(self, nusers, nitems, nexpls, nattrs, embed_size, dropout_rate, alpha, nlayers, attr_semantics):
        super(SCEMIMa, self).__init__()
        self.nusers = nusers
        self.nitems = nitems
        self.nexpls = nexpls
        self.embed_size = embed_size
        self.item_pad_idx = nitems
        self.expl_pad_idx = nexpls
        self.attr_pad_idx = nattrs
        self.alpha = alpha 
        self.item_embed = nn.Embedding(nitems+1, embed_size)  # leave one for pad
        self.item_bias = nn.Embedding(nitems+1, 1)            # leave one for pad
        self.expl_embed = nn.Embedding(nexpls+1, embed_size)  
        self.expl_bias = nn.Embedding(nexpls+1, 1)
        pad_embedding = torch.randn((1, 768), dtype=torch.float)
        attr_semantics_with_pad = torch.cat([attr_semantics, pad_embedding], dim=0)
        self.attr_embed = nn.Embedding.from_pretrained(attr_semantics_with_pad)
        self.attr_bias = nn.Embedding(nattrs+1, 1)
        self.proj = nn.Linear(768, embed_size)
        self.position_embedding = nn.Embedding(25, embed_size) # 25 is seq_len
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_size, nhead=2, dropout=dropout_rate, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.layernorm = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.W1 = nn.Linear(embed_size, embed_size)
        self.W2 = nn.Linear(embed_size, embed_size)
        self.apply(self.init_weights)
        self.lossr_fn = nn.CrossEntropyLoss(ignore_index=self.item_pad_idx)
        self.losse_fn = nn.CrossEntropyLoss(ignore_index=self.expl_pad_idx)
        self.lossa_fn = nn.CrossEntropyLoss(ignore_index=self.attr_pad_idx)

    def init_weights(self, module):
        initrange = 0.1
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -initrange, initrange)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.uniform_(m.weight, -initrange, initrange)

    def forward(self, rec_input, expl_input):
        # rec_input and expl_input in shape (N, seqlen, 2)
        device = rec_input.device
        seq_len = rec_input.size(1)
        position_ids = torch.arange(seq_len, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand_as(rec_input[:,:,0])
        position_embedding = self.position_embedding(position_ids)
        
        # Item prediction
        rec_embeddings_item = self.item_embed(rec_input[:,:,0])   # rec_input_items
        rec_embeddings_expl = self.expl_embed(rec_input[:,:,1])   # rec_input_expls
        rec_embeddings = self.alpha * rec_embeddings_item + (1-self.alpha) * rec_embeddings_expl
        rec_embeddings = self.dropout(self.layernorm(rec_embeddings + position_embedding))
        rec_mask = generate_square_mask(rec_input[:,:,0])  # causal mask for item input
        rec_digits = self.encoder(src=rec_embeddings, mask=rec_mask)
        candidate_item_embeddings = self.item_embed.weight  # (num_items, embed_dim)
        candidate_item_bias = self.item_bias.weight.squeeze()  # (num_items,)
        rec_output = torch.einsum('nsd, cd->nsc', rec_digits, candidate_item_embeddings) + candidate_item_bias
    
        # Explanation prediction
        expl_embeddings_item = self.item_embed(expl_input[:,:,0])
        expl_embeddings_expl = self.expl_embed(expl_input[:,:,1])
        expl_embeddings = self.alpha * expl_embeddings_expl + (1-self.alpha) * expl_embeddings_item
        expl_embeddings = self.dropout(self.layernorm(expl_embeddings + position_embedding))
        expl_mask = generate_square_mask(expl_input[:,:,0]) 
        expl_digits = self.encoder(src=expl_embeddings, mask=expl_mask)
        candidate_expl_embeddings = self.expl_embed.weight 
        candidate_expl_bias = self.expl_bias.weight.squeeze()  # (num_expls,)
        expl_output = torch.einsum('nsd, ed->nse', expl_digits, candidate_expl_embeddings) + candidate_expl_bias
        mi_output1 = torch.einsum('nsd, cd->nsc', self.W1(expl_digits), candidate_item_embeddings) + candidate_item_bias
        projected_candidate_attr_embeddings = self.proj(self.attr_embed.weight)
        mi_output2 = torch.einsum('nsd, cd->nsc', self.W2(expl_digits), projected_candidate_attr_embeddings) + self.attr_bias.weight.squeeze()
        return rec_output, expl_output, mi_output1, mi_output2

    def gather(self, batch, device): 
        item_input, item_output, expl_input, expl_output = batch
        item_input = item_input.to(device)
        item_output = item_output.to(device)
        expl_input = expl_input.to(device)
        expl_output = expl_output.to(device)
        return item_input, item_output, expl_input, expl_output

    def rank_action(self, rec_input, expl_input):
        rec_output, expl_output, _, _ = self.forward(rec_input, expl_input)
        rec_output = rec_output[:,-1,:]
        expl_output = expl_output[:,-1,:]
        return rec_output.topk(self.nitems, dim=-1).indices, expl_output.topk(self.nexpls, dim=-1).indices
        

def trainModel(model, train_dataloader, valid_dataloader, args):
    learning_rate = args.lr
    log_file = args.log_file
    epochs = args.epochs
    device = args.device
    save_file = args.save_file
    lambda_value = args.lambda_value
    gamma = args.gamma
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    enduration = 0
    prev_valid_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        avg_loss = 0
        for batch in tqdm(train_dataloader, total=len(train_dataloader)):
            rec_input, rec_output, expl_input, expl_output = model.gather(batch, device)
            optimizer.zero_grad()
            preds_rec, preds_expls, mi_output1, mi_output2 = model(rec_input, expl_input)      # (N, seqlen, nitems), (N, seqlen, nexpls)
            rec_loss = model.lossr_fn(preds_rec.view(-1, preds_rec.size(-1)), rec_output.view(-1))
            expl_loss = model.losse_fn(preds_expls.view(-1, preds_expls.size(-1)), expl_output.view(-1))
            mi_loss1 = model.lossr_fn(mi_output1.view(-1, mi_output1.size(-1)), expl_input[:, :, 0].view(-1))
            mi_loss2 = model.lossa_fn(mi_output2.view(-1, mi_output2.size(-1)), expl_input[:, :, 2].view(-1))
            loss = rec_loss + lambda_value * expl_loss + args.gamma * (mi_loss1 + mi_loss2)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            avg_loss += loss.item()
        avg_loss /= len(train_dataloader)
        with open(log_file, "a") as f:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Get the current time
            f.write(f"Epoch {epoch+1}: [{current_time}] [lr: {learning_rate}] Loss = {avg_loss:.4f}\n")

        # checking learning rate
        current_valid_loss = validModel(model, valid_dataloader, device)
        if current_valid_loss > prev_valid_loss:
            learning_rate /= 2.0
            enduration += 1
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate
        else:
            torch.save(model.state_dict(), save_file)
        prev_valid_loss = current_valid_loss
        if enduration  >= 5:
            break


def validModel(model, valid_dataloader, device):
    model.eval()
    with torch.no_grad():
        avg_loss = 0
        for batch in valid_dataloader:
            item_input, item_output, expl_input, expl_output = model.gather(batch, device)
            preds_items, preds_expls, _, _ = model(item_input, expl_input)  # Get both item and explanation predictions
            final_preds_items = preds_items[:, -1, :]  # (N, nitems)
            final_preds_expls = preds_expls[:, -1, :]  # (N, nexpls)
            loss_items = model.lossr_fn(final_preds_items, item_output.view(-1))
            loss_expls = model.losse_fn(final_preds_expls, expl_output.view(-1))
            loss = loss_items + loss_expls  # Combine both losses
            avg_loss += loss.item()
        avg_loss /= len(valid_dataloader)
        return avg_loss


if __name__ == '__main__':
    random.seed(43)
    torch.manual_seed(43)

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset', type=str, default='AM-Movies')
    argparser.add_argument('--lr', type=float, default=0.001)
    argparser.add_argument('--dropout_rate', type=float, default=0.5)
    argparser.add_argument('--lambda_value', type=float, default=0.5)
    argparser.add_argument('--alpha', type=float, default=0.5)
    argparser.add_argument('--gamma', type=float, default=0.1)
    argparser.add_argument('--test_mode', type=int, default=1)  
    argparser.add_argument('--log_file', type=str, default='log.txt')  # Add log_file argument with a default value.
    argparser.add_argument('--embed_size', type=int, default=50)
    argparser.add_argument('--epochs', type=int, default=50)
    argparser.add_argument('--nlayers', type=int, default=2)
    argparser.add_argument('--save_file', type=str, default='model.pth')
    argparser.add_argument('--device', type=int, default=0)
    argparser.add_argument('--batch_size', type=int, default=512)

    args = argparser.parse_args()
    with open(args.log_file, 'a') as log_file:
        arg_dict = vars(args)
        log_file.write(f"\nCurrent time: {datetime.now()}\n")
        for arg, value in arg_dict.items():
            log_file.write(f"--{arg} {value}\n")

    dataset = args.dataset
    data_path = '../../Processed_data/' + dataset + '/'
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
    nusers = train_df.user_idx.max() + 1
    nitems = train_df.item_idx.max() + 1
    nexpls = train_df.exp_idx.max() + 1
    nattrs = train_df.attribute_idx.max() + 1
    item_pad_idx = nitems
    expl_pad_idx = nexpls
    attr_pad_idx = nattrs
    attr_semantics = np.load(data_path + "attr_semantics.npy")
    attr_semantics= torch.tensor(attr_semantics, dtype=torch.float)

    if args.dataset == "Yelp":
        train_dataloader, valid_dataloader, test_dataloader = construct_data(train_df, valid_df, test_df, item_pad_idx, expl_pad_idx, attr_pad_idx, args.test_mode, 200)
    else:
        train_dataloader, valid_dataloader, test_dataloader = construct_data(train_df, valid_df, test_df, item_pad_idx, expl_pad_idx, attr_pad_idx, args.test_mode, args.batch_size)
    model = SCEMIMa(nusers, nitems, nexpls, nattrs, args.embed_size, args.dropout_rate, args.alpha, args.nlayers, attr_semantics)
    model = model.to(args.device)
    trainModel(model, train_dataloader, valid_dataloader, args)

    users_ground_item = {}
    users_ground_expl = {}
    # {0:[], 1:[],...}
    for i in tqdm(range(len(test_df))):
        user = test_df.iloc[i]["user_idx"] 
        item = test_df.iloc[i]["item_idx"]
        expl = test_df.iloc[i]["exp_idx"]
        users_ground_item[user] = [item] # list
        users_ground_expl[user] = [expl]


    # prepare predictions
    users_ranked_item = []
    users_ranked_expl = []
    model.load_state_dict(torch.load(args.save_file))
    model.to(args.device)
    model.eval()
    with torch.no_grad():
        for batch in test_dataloader:
            item_input, _, expl_input, _ = model.gather(batch, args.device)
            ranked_items, ranked_expls  = model.rank_action(item_input, expl_input)
            ranked_items = ranked_items[:,:100]  # (N, 100)
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
        f.write(f"[Explanation]: recall@{k}: {explanation_score[0]}, ndcg@{k}: {explanation_score[1]} \n")

    # compute scores, k = 10
    k = 10
    recommendation_score = compute_k(users_ground_item, users_ranked_item, k)
    explanation_score = compute_k(users_ground_expl, users_ranked_expl, k)
    with open(args.log_file, "a") as f:
        f.write(f"\nCurrent time: {datetime.now()}\n")
        f.write(f"[recommendation]: recall@{k}: {recommendation_score[0]}, ndcg@{k}: {recommendation_score[1]} \n")
        f.write(f"[Explanation]: recall@{k}: {explanation_score[0]}, ndcg@{k}: {explanation_score[1]} \n")

