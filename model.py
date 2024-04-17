import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from fmlprec_utils import FMLP
from lrurec_utils import LRU
from sasrec_utils import SASRec
from utils import * 
import copy

class Mine(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.fc1 = nn.Linear(embed_size*2, embed_size)
        self.fc2 = nn.Linear(embed_size, embed_size)
        self.fc3 = nn.Linear(embed_size, 1)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.1)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
        
    def forward(self, input):
        output = torch.relu(self.fc1(input))
        output = torch.relu(self.fc2(output))
        output = self.fc3(output).squeeze()
        return output


class EncoderLayer(nn.Module):
    def __init__(self, config):
        super(EncoderLayer, self).__init__()
        self.model_name = config.get("model_name")
        embed_size = config.get("embed_size")
        nlayers = config.get("nlayers")
        dropout_rate = config.get("dropout_rate")
        if self.model_name == "GRU4Rec":
            self.encoder = nn.GRU(embed_size, embed_size, nlayers, bias=False, batch_first=True)
        elif self.model_name == "SASRec":
            layer = SASRec(hidden_size=embed_size, nhead=2, dropout=0.2, batch_first=True)
            self.encoder = TransformerEncoder(layer, num_layers=nlayers)
        elif self.model_name == "FMLPRec":
            self.encoder = FMLP(embed_size, dropout=0.2, nlayers=nlayers)
        else:
            self.encoder = LRU(embed_size, dropout=0.2, nlayers=nlayers)


    def forward(self, embeddings, input,  pad_idx):
        if self.model_name == "GRU4Rec":
            output, _ = self.encoder(embeddings)
            return output

        elif self.model_name == "SASRec":
            mask = generate_square_mask(input)
            output = self.encoder(src=embeddings, mask=mask)
            return output


        elif self.model_name == "FMLPRec":
            mask = generate_square_mask(input)
            mask = mask.float()* -10000.0
            outputs = self.encoder(embeddings, mask)
            output = outputs[-1]
            return output

        else:
            mask = (input[:,:,0] != pad_idx)
            output = self.encoder(embeddings, mask)
            return output



class SCEModel(nn.Module):
    def __init__(self, config):
        super(SCEModel, self).__init__()
        self.model_name = config.get("model_name")
        self.mi_lowerbound = config.get("mi_lowerbound")
        self.embed_size = config.get("embed_size")
        self.nitems = config.get("nitems")
        self.nexpls = config.get("nexpls")
        self.item_pad_idx = config.get("item_pad_idx")
        self.expl_pad_idx = config.get("expl_pad_idx")
        self.dropout_rate = config.get("dropout_rate")
        self.alpha = config.get("alpha")
        self.lambda_value = config.get("lambda_value")
        self.mu = config.get("mu")
        self.max_len = config.get("max_len")
        self.negatives = config.get("negatives")
        self.item_embed = nn.Embedding(self.nitems+1, self.embed_size, padding_idx=self.item_pad_idx)
        self.expl_embed = nn.Embedding(self.nexpls+1, self.embed_size, padding_idx=self.expl_pad_idx)
        self.item_bias = nn.Embedding(self.nitems+1, 1, padding_idx=self.item_pad_idx)
        self.expl_bias = nn.Embedding(self.nexpls+1, 1, padding_idx=self.expl_pad_idx)
        self.position_embedding = nn.Embedding(self.max_len, self.embed_size)
        self.item_encoder = EncoderLayer(config)
        self.expl_encoder = EncoderLayer(config)
        self.dropout = nn.Dropout(p=self.dropout_rate)
        self.layernorm = nn.LayerNorm(self.embed_size)
        self.item_loss_fn = nn.CrossEntropyLoss(ignore_index=self.item_pad_idx)
        self.expl_loss_fn = nn.CrossEntropyLoss(ignore_index=self.expl_pad_idx)
        if self.mi_lowerbound == "mine":
            self.mine_miv = Mine(self.embed_size)
            self.mine_mie = Mine(self.embed_size)
        else:
            self.W0 = nn.Parameter(torch.ones(self.embed_size, self.embed_size))
            self.W1 = nn.Parameter(torch.ones(self.embed_size, self.embed_size))
        self.apply(self.init_weights)



    def init_weights(self, module):
        if self.model_name == "LRURec":
            mean = 0
            std = 0.02
            lower = -0.04
            upper = 0.04
            with torch.no_grad():
                l = (1. + math.erf(((lower - mean) / std) / math.sqrt(2.))) / 2.
                u = (1. + math.erf(((upper - mean) / std) / math.sqrt(2.))) / 2.

                for n, p in self.named_parameters():
                    if not 'layer_norm' in n and 'params_log' not in n:
                        if torch.is_complex(p):
                            p.real.uniform_(2 * l - 1, 2 * u - 1)
                            p.imag.uniform_(2 * l - 1, 2 * u - 1)
                            p.real.erfinv_()
                            p.imag.erfinv_()
                            p.real.mul_(std * math.sqrt(2.))
                            p.imag.mul_(std * math.sqrt(2.))
                            p.real.add_(mean)
                            p.imag.add_(mean)
                        else:
                            p.uniform_(2 * l - 1, 2 * u - 1)
                            p.erfinv_()
                            p.mul_(std * math.sqrt(2.))
                            p.add_(mean)

        else:
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.1)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()


    def gather(self, batch, device, test=False):
        if test:
            input = batch[0].to(device)
            return input

        else:
            input, output = batch
            input = input.to(device)
            output = output.to(device)
            return input, output


    def rank_action(self, item_input, expl_input):
        item_outputs, expl_outputs = self.forward(item_input, None, expl_input, False)
        if self.model_name == "LRURec":
            item_outputs = item_outputs[:,-1,:]
            expl_outputs = expl_outputs[:,-1,:]
        else:
            # find the last action excluding padding. 
            index = (item_input[:,:,0] != self.item_pad_idx).sum(dim=1)
            item_outputs = torch.gather(item_outputs, dim=1, index=index.view(-1, 1, 1).expand(-1, -1, item_outputs.size(2))).squeeze()
            expl_outputs = torch.gather(expl_outputs, dim=1, index=index.view(-1, 1, 1).expand(-1, -1, expl_outputs.size(2))).squeeze()
        
        item_ranked = item_outputs.topk(self.nitems, dim=-1).indices
        expl_ranked = expl_outputs.topk(self.nexpls, dim=-1).indices
        return item_ranked, expl_ranked


    def forward(self, item_input, item_output, expl_input, training=True):
        batch_size = item_input.shape[0]
        device = item_input.device

        position_ids = torch.arange(item_input.size(1), dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_input[:,:,0])
        position_embedding = self.position_embedding(position_ids) # (N, len, emsize)

        item_embeddings4item = self.item_embed(item_input[:,:,0]) 
        expl_embeddings4item = self.expl_embed(item_input[:,:,1])  #(N, len, emsize)
        input_embeds4item = self.alpha*item_embeddings4item + (1-self.alpha)*expl_embeddings4item + position_embedding
        input_embeds4item = self.dropout(self.layernorm(input_embeds4item))
        item_digits = self.item_encoder(input_embeds4item, item_input, self.item_pad_idx) # item_input is used for determining mask
        item_outputs = torch.einsum('nsd, cd-> nsc', item_digits, self.item_embed.weight) + self.item_bias.weight.squeeze()

        item_embeddings4expl = self.item_embed(expl_input[:,:,0])
        expl_embeddings4expl = self.expl_embed(expl_input[:,:,1])
        input_embeds4expl = (1-self.alpha)*item_embeddings4expl + self.alpha*expl_embeddings4expl + position_embedding
        input_embeds4expl = self.dropout(self.layernorm(input_embeds4expl))
        expl_digits = self.expl_encoder(input_embeds4expl, expl_input, self.item_pad_idx)
        expl_outputs = torch.einsum('nsd, cd-> nsc', expl_digits, self.expl_embed.weight) + self.expl_bias.weight.squeeze()


        if training:
            if self.mi_lowerbound == "mine":
                mask = (expl_input[:,:,1] != self.expl_pad_idx).float()
                num_valid_tokens = torch.sum(mask, dim=1)
                miv_joint = torch.cat([expl_digits, item_embeddings4expl], dim=-1) # (N, 25, embed_size*2)
                indices = torch.randperm(item_embeddings4expl.size(0))
                miv_neg_items = item_embeddings4expl[indices]
                miv_marginal = torch.cat([expl_digits, miv_neg_items],dim=-1)
                miv_t = self.mine_miv(miv_joint)
                miv_t = torch.sum(miv_t * mask, dim=1)/num_valid_tokens
                miv_et = self.mine_miv(miv_marginal).exp()
                miv_et = torch.sum(miv_et * mask, dim=1)/num_valid_tokens
                loss_miv = -(miv_t.mean() - miv_et.mean().log())

                expl_embeddings4mie = self.expl_embed(item_output[:,:,1])
                mask = (item_input[:,:,0] != self.item_pad_idx).float()
                num_valid_tokens = torch.sum(mask, dim=1)
                mie_joint = torch.cat([item_digits, expl_embeddings4mie], dim=-1)
                indices = torch.randperm(expl_embeddings4mie.size(0))
                mie_neg_expls = expl_embeddings4mie[indices]
                mie_marginal = torch.cat([item_digits, mie_neg_expls], dim=-1)
                mie_t = self.mine_mie(mie_joint)
                mie_t = torch.sum(mie_t * mask, dim=1)/num_valid_tokens
                mie_et = self.mine_mie(mie_marginal).exp()
                mie_et = torch.sum(mie_et * mask, dim=1)/num_valid_tokens
                loss_mie = -(mie_t.mean() - mie_et.mean().log())
                loss_mi = loss_miv + loss_mie

            else:
                if self.negatives == "all":
                    preds_miv = expl_digits@self.W0
                    preds_miv = torch.einsum('nsd, cd-> nsc', preds_miv, self.item_embed.weight)
                    item_pos = item_input[:,:,0]
                    loss_miv = F.cross_entropy(preds_miv.view(-1, preds_miv.size(-1)), item_pos.view(-1), ignore_index=self.item_pad_idx)

                    expl_pos = item_output[:,:,1]
                    preds_mie = item_digits@self.W1
                    preds_mie = torch.einsum('nsd, cd-> nsc', preds_mie, self.expl_embed.weight)
                    loss_mie = F.cross_entropy(preds_mie.view(-1, preds_mie.size(-1)), expl_pos.view(-1), ignore_index=self.expl_pad_idx)
                    loss_mi = loss_miv + loss_mie

            return item_outputs, expl_outputs, loss_mi

        else:
            return item_outputs, expl_outputs


