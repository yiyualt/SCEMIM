import torch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from tqdm import tqdm


def shift_right(input_ids, pad_id):
    pad_token_id = pad_id
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
    shifted_input_ids[..., 0] = pad_id
    return shifted_input_ids


def construct_data(user2action, user2action_4test, valid_df, test_df, item_pad_idx, expl_pad_idx, batch_size=512, max_len=25):
    train_item_actions = []
    train_expl_actions = []
    for user, action in user2action.items():
        item_action = action["item_idx"] # should have same length with expl_actions
        expl_action = action["exp_idx"]
        split_item_actions = [item_action[i:i+25] for i in range(0, len(action), 25)]
        split_expl_actions = [expl_action[i:i+25] for i in range(0, len(action), 25)]
        for split_item_action, split_expl_action in zip(split_item_actions, split_expl_actions):
            if len(split_item_action) < max_len:
                split_item_action = split_item_action + [item_pad_idx] * (25 - len(split_item_action))
            if len(split_expl_action) < max_len:
                split_expl_action = split_expl_action + [expl_pad_idx] * (25 - len(split_expl_action))
            train_item_actions.append(split_item_action)
            train_expl_actions.append(split_expl_action)
    
    train_item_output = torch.tensor(train_item_actions, dtype=torch.long)
    train_expl_output = torch.tensor(train_expl_actions, dtype=torch.long)
    train_item_input = shift_right(train_item_output, item_pad_idx)
    train_expl_input = shift_right(train_expl_output, expl_pad_idx)
    
    # combine previous explanation actions for recommendaiton
    train_item_inputs = torch.concat((train_item_input.unsqueeze(-1), train_expl_input.unsqueeze(-1)), dim=-1)
    # combine item actions for explanation
    train_expl_inputs = torch.concat((train_item_output.unsqueeze(-1), train_expl_input.unsqueeze(-1)), dim=-1) # note that the used item actions are shifted to the right.
    train_item_dataset = TensorDataset(train_item_inputs, torch.stack((train_item_output, train_expl_output),dim=2))
    train_item_dataloader = DataLoader(train_item_dataset, batch_size=batch_size, shuffle=True)
    train_expl_dataset = TensorDataset(train_expl_inputs, train_expl_output)
    train_expl_dataloader = DataLoader(train_expl_dataset, batch_size=batch_size, shuffle=True)
    
    # construct for valid
    valid_item_output = valid_df.groupby('user_idx')['item_idx'].last().tolist()
    valid_expl_output = valid_df.groupby('user_idx')['exp_idx'].last().tolist()
    valid_item_inputs = []
    valid_expl_inputs = []
    for user, action in user2action.items():
        item_action = action["item_idx"] # should have same length with expl_actions
        expl_action = action["exp_idx"]
        split_item_actions = [item_action[i:i+25] for i in range(0, len(action), 25)]
        split_expl_actions = [expl_action[i:i+25] for i in range(0, len(action), 25)]
        # use the most recent actions as input for each user.
        latest_item_action = split_item_actions[-1]
        latest_expl_action = split_expl_actions[-1]
        if len(latest_item_action) < 25:
            valid_item_action4rec = [item_pad_idx] + latest_item_action + [item_pad_idx] * (24 - len(latest_item_action))
            valid_item_action4expl = latest_item_action + [valid_item_output[user]] +  [item_pad_idx] * (24 - len(latest_item_action))
        
        else:
            valid_item_action4rec = [item_pad_idx] + latest_item_action[1:]
            valid_item_action4expl = latest_item_action[1:] + [valid_item_output[user]]

        if len(latest_expl_action) < 25:
            latest_expl_action = [expl_pad_idx] + latest_expl_action + [expl_pad_idx] * (24 - len(latest_expl_action))
        else:
            latest_expl_action[0] = expl_pad_idx
        valid_item_inputs.append([valid_item_action4rec, latest_expl_action])
        valid_expl_inputs.append([valid_item_action4expl, latest_expl_action])

    valid_item_input = torch.tensor(valid_item_inputs, dtype=torch.long).permute(0, 2, 1)
    valid_item_output = torch.tensor(valid_item_output, dtype=torch.long)
    valid_item_dataset = TensorDataset(valid_item_input, valid_item_output)
    valid_item_dataloader = DataLoader(valid_item_dataset, batch_size=batch_size, shuffle=True)
    valid_expl_input = torch.tensor(valid_expl_inputs, dtype=torch.long).permute(0, 2, 1)
    valid_expl_output = torch.tensor(valid_expl_output, dtype=torch.long)
    valid_expl_dataset = TensorDataset(valid_expl_input, valid_expl_output)
    valid_expl_dataloader = DataLoader(valid_expl_dataset, batch_size=batch_size, shuffle=True)

    # construct for test. similar to valid only without shuffle and labels. 
    test_item_output = test_df.groupby('user_idx')['item_idx'].last().tolist()
    test_item_inputs = []
    test_expl_inputs = []
    for user, action in user2action_4test.items():
        item_action = action["item_idx"] # should have same length with expl_actions
        expl_action = action["exp_idx"]
        split_item_actions = [item_action[i:i+25] for i in range(0, len(action), 25)]
        split_expl_actions = [expl_action[i:i+25] for i in range(0, len(action), 25)]
        # use the most recent actions as input for each user.
        latest_item_action = split_item_actions[-1]
        latest_expl_action = split_expl_actions[-1]
        if len(latest_item_action) < 25:
            test_item_action4rec = [item_pad_idx] + latest_item_action + [item_pad_idx] * (24 - len(latest_item_action))
            test_item_action4expl = latest_item_action + [test_item_output[user]] +  [item_pad_idx] * (24 - len(latest_item_action))
        
        else:
            test_item_action4rec = [item_pad_idx] + latest_item_action[1:]
            test_item_action4expl = latest_item_action[1:] + [test_item_output[user]]

        if len(latest_expl_action) < 25:
            latest_expl_action = [expl_pad_idx] + latest_expl_action + [expl_pad_idx] * (24 - len(latest_expl_action))

        valid_item_inputs.append([test_item_action4rec, latest_expl_action])
        valid_expl_inputs.append([test_item_action4expl, latest_expl_action])
    valid_item_input = torch.tensor(valid_item_inputs, dtype=torch.long).permute(0, 2, 1)
    valid_item_dataset = TensorDataset(valid_item_input)
    test_item_dataloader = DataLoader(valid_item_dataset, batch_size=batch_size, shuffle=False)

    valid_expl_input = torch.tensor(valid_expl_inputs, dtype=torch.long).permute(0, 2, 1)
    valid_expl_dataset = TensorDataset(valid_expl_input)
    test_expl_dataloader = DataLoader(valid_expl_dataset, batch_size=batch_size, shuffle=False)
    train_dataloader = {"item": train_item_dataloader, "expl": train_expl_dataloader}
    valid_dataloader = {"item": valid_item_dataloader, "expl": valid_expl_dataloader}
    test_dataloader = {"item": test_item_dataloader, "expl": test_expl_dataloader}
    return train_dataloader, valid_dataloader, test_dataloader


# LRURec model needs left-padding. 
def construct_data_for_LRURec(user2action, user2action_4test, valid_df, test_df, item_pad_idx, expl_pad_idx, batch_size=512, max_len=25):
    train_item_actions = []
    train_expl_actions = []
    for user, action in user2action.items():
        item_action = action["item_idx"] # should have same length with expl_actions
        expl_action = action["exp_idx"]
        split_item_actions = [item_action[i:i+25] for i in range(0, len(action), 25)]
        split_expl_actions = [expl_action[i:i+25] for i in range(0, len(action), 25)]
        for split_item_action, split_expl_action in zip(split_item_actions, split_expl_actions):
            if len(split_item_action) < max_len:
                split_item_action =  [item_pad_idx] * (25 - len(split_item_action)) + split_item_action
                split_expl_action =  [expl_pad_idx] * (25 - len(split_expl_action)) + split_expl_action
            train_item_actions.append(split_item_action)
            train_expl_actions.append(split_expl_action)
    train_item_output = torch.tensor(train_item_actions, dtype=torch.long)
    train_expl_output = torch.tensor(train_expl_actions, dtype=torch.long)
    train_item_input = shift_right(train_item_output, item_pad_idx)
    train_expl_input = shift_right(train_expl_output, expl_pad_idx)

    # combine previous explanation actions for recommendaiton
    train_item_inputs = torch.concat((train_item_input.unsqueeze(-1), train_expl_input.unsqueeze(-1)), dim=-1)
    # combine item actions for explanation
    train_expl_inputs = torch.concat((train_item_output.unsqueeze(-1), train_expl_input.unsqueeze(-1)), dim=-1) # note that the used item actions are shifted to the right.
    train_item_dataset = TensorDataset(train_item_inputs, torch.stack((train_item_output, train_expl_output),dim=2))
    train_item_dataloader = DataLoader(train_item_dataset, batch_size=batch_size, shuffle=True)
    train_expl_dataset = TensorDataset(train_expl_inputs, train_expl_output)
    train_expl_dataloader = DataLoader(train_expl_dataset, batch_size=batch_size, shuffle=True)
    
    # construct for valid
    valid_item_output = valid_df.groupby('user_idx')['item_idx'].last().tolist()
    valid_expl_output = valid_df.groupby('user_idx')['exp_idx'].last().tolist()
    valid_item_inputs = []
    valid_expl_inputs = []
    for user, action in user2action.items():
        item_action = action["item_idx"] # should have same length with expl_actions
        expl_action = action["exp_idx"]
        split_item_actions = [item_action[i:i+25] for i in range(0, len(action), 25)]
        split_expl_actions = [expl_action[i:i+25] for i in range(0, len(action), 25)]
        # use the most recent actions as input for each user.
        latest_item_action = split_item_actions[-1]
        latest_expl_action = split_expl_actions[-1]
        if len(latest_item_action) < 25:
            valid_item_action4rec = [item_pad_idx] * (25 - len(latest_item_action)) + latest_item_action
            valid_item_action4expl = [item_pad_idx] * (24 - len(latest_item_action)) + latest_item_action + [valid_item_output[user]]
        
        else:
            valid_item_action4expl = latest_item_action[1:] + [valid_item_output[user]]

        if len(latest_expl_action) < 25:
            latest_expl_action = [expl_pad_idx] * (25 - len(latest_expl_action)) + latest_expl_action
        valid_item_inputs.append([valid_item_action4rec, latest_expl_action])
        valid_expl_inputs.append([valid_item_action4expl, latest_expl_action])

    valid_item_input = torch.tensor(valid_item_inputs, dtype=torch.long).permute(0, 2, 1)
    valid_item_output = torch.tensor(valid_item_output, dtype=torch.long)
    valid_item_dataset = TensorDataset(valid_item_input, valid_item_output)
    valid_item_dataloader = DataLoader(valid_item_dataset, batch_size=batch_size, shuffle=True)
    valid_expl_input = torch.tensor(valid_expl_inputs, dtype=torch.long).permute(0, 2, 1)
    valid_expl_output = torch.tensor(valid_expl_output, dtype=torch.long)
    valid_expl_dataset = TensorDataset(valid_expl_input, valid_expl_output)
    valid_expl_dataloader = DataLoader(valid_expl_dataset, batch_size=batch_size, shuffle=True)

    test_item_output = test_df.groupby('user_idx')['item_idx'].last().tolist()
    test_items_inputs = []
    test_expl_inputs = []
    for user, action in user2action_4test.items():
        item_action = action["item_idx"] # should have same length with expl_actions
        expl_action = action["exp_idx"]
        split_item_actions = [item_action[i:i+25] for i in range(0, len(action), 25)]
        split_expl_actions = [expl_action[i:i+25] for i in range(0, len(action), 25)]
        # use the most recent actions as input for each user.
        latest_item_action = split_item_actions[-1]
        latest_expl_action = split_expl_actions[-1]
        if len(latest_item_action) < 25:
            test_item_action4rec = [item_pad_idx] * (25 - len(latest_item_action)) + latest_item_action
            test_item_action4expl = [item_pad_idx] * (24 - len(latest_item_action)) + latest_item_action + [test_item_output[user]]
        
        else:
            test_item_action4expl = latest_item_action[1:] + [test_item_output[user]]

        if len(latest_expl_action) < 25:
            latest_expl_action = [expl_pad_idx] * (25 - len(latest_expl_action)) + latest_expl_action
        test_items_inputs.append([test_item_action4rec, latest_expl_action])
        test_expl_inputs.append([test_item_action4expl, latest_expl_action])
    test_item_input = torch.tensor(test_items_inputs, dtype=torch.long).permute(0, 2, 1)
    test_item_dataset = TensorDataset(test_item_input)
    test_item_dataloader = DataLoader(test_item_dataset, batch_size=batch_size, shuffle=False)
    test_expl_input = torch.tensor(test_expl_inputs, dtype=torch.long).permute(0, 2, 1)
    test_expl_dataset = TensorDataset(test_expl_input)
    test_expl_dataloader = DataLoader(test_expl_dataset, batch_size=batch_size, shuffle=False)

    train_dataloader = {"item": train_item_dataloader, "expl": train_expl_dataloader}
    valid_dataloader = {"item": valid_item_dataloader, "expl": valid_expl_dataloader}
    test_dataloader = {"item": test_item_dataloader, "expl": test_expl_dataloader}
    return train_dataloader, valid_dataloader, test_dataloader



def generate_square_mask(seq):
    seqlen = seq.shape[1]
    device = seq.device
    mask = torch.triu(torch.ones((seqlen, seqlen), device=device), diagonal=1) == 1
    return mask


def compute_ndcg_k(users_ground, users_ranked, k):
    """
    example:
        >>> users_ground = {0: [0, 10, 302, 365, 656, 665, 751], 2: [0,2,3,24,24,24]}
        >>> users_topk = {0:[0,1,2,124], 1:[2,3,235,25]}
    """
    ndcg_values = []

    for user in tqdm(users_ground):
        ground_truth = users_ground[user]
        predicted_rankings = users_ranked[user][:k]

        relevance_scores = [1 if item in ground_truth else 0 for item in predicted_rankings]
        dcg = np.sum(relevance_scores / np.log2(np.arange(2, len(relevance_scores) + 2)))

        ideal_rankings = sorted(ground_truth, reverse=True)[:k]
        ideal_scores = [1] * len(ideal_rankings)
        idcg = np.sum(ideal_scores / np.log2(np.arange(2, len(ideal_scores) + 2)))
        ndcg = dcg / idcg
        ndcg_values.append(ndcg)

    return np.mean(ndcg_values)


def compute_recall_k(users_ground, users_ranked, k):
    recall_values = []
    for user in tqdm(users_ground):
        ground_truth = users_ground[user]
        predicted_rankings = users_ranked[user][:k]
        relevant_items_count = len(set(ground_truth) & set(predicted_rankings))
        recall = relevant_items_count / len(ground_truth)
        recall_values.append(recall)
    return np.mean(recall_values)


# this one is more efficient for computing both recall and ndcg.
def compute_k(users_ground, users_ranked, k):
    recall_values = []
    ndcg_values = []
    for user in tqdm(users_ground):
        ground_truth = users_ground[user]
        predicted_rankings = users_ranked[user][:k]

        # recall
        relevant_items_count = len(set(ground_truth) & set(predicted_rankings))
        recall = relevant_items_count / len(ground_truth)
        recall_values.append(recall)

        # ndcg
        relevance_scores = [1 if item in ground_truth else 0 for item in predicted_rankings]
        dcg = np.sum(relevance_scores / np.log2(np.arange(2, len(relevance_scores) + 2)))
        ideal_rankings = sorted(ground_truth, reverse=True)[:k]
        ideal_scores = [1] * len(ideal_rankings)
        idcg = np.sum(ideal_scores / np.log2(np.arange(2, len(ideal_scores) + 2)))
        ndcg = dcg / idcg
        ndcg_values.append(ndcg)
    return round(np.mean(recall_values), 4), round(np.mean(ndcg_values),4)


