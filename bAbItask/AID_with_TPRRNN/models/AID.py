from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils import MLP, LayerNorm, OptionalLayer

import math
import numpy as np

from itertools import combinations

AVAILABLE_ELEMENTS = ('e1', 'e2', 'r1', 'r2', 'r3')


class TprRnn(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super(TprRnn, self).__init__()

        config['slot_size'] = 64
        config['mlp_hidden_size'] = 128
        config['binding_num_slots'] = 5
        config['reasoning_num_slots'] = 4
        config['num_iter'] = 2

        self.input_module = InputModule(config)
        self.update_module = UpdateModule(config=config)
        self.inference_module = InferenceModule(config=config)

        self.decomposition_module = Decomposition(
            input_size=config["symbol_size"],
            slot_size=config["slot_size"],
            mlp_hidden_size=config["mlp_hidden_size"],
            binding_num_slots=config['binding_num_slots'] ,
            reasoning_num_slots=config['reasoning_num_slots'],
            num_iterations=config['num_iter'],
        )

        self.vocab_size = config['vocab_size']
        self.reconstruction_linear = nn.Linear(config["symbol_size"], self.vocab_size)
        self.recon_fn = nn.CrossEntropyLoss(reduction='none')
        self.recon_dropout = nn.Dropout(p=0.1)

        self.r_linear = nn.Linear(config["slot_size"] * self.decomposition_module.num_slots, self.vocab_size)

    def forward(self, story: torch.Tensor, query: torch.Tensor, reconstruction=False):
        # story_embed: [b, s, w, e]
        # query_embed: [b, w, e]
        story_embed, query_embed, story_mask, query_mask, sentence_sum, query_sum = self.input_module(story, query)

        binding_slots = self.decomposition_module.binding_slot_attention(story_embed, sentence_sum, story_mask)
        TPR = self.update_module(binding_slots)

        query_embed = query_embed.unsqueeze(dim=1)
        query_mask = query_mask.unsqueeze(dim=1)
        query_sum = query_sum.unsqueeze(dim=1)
        reasoning_slots = self.decomposition_module.reasoning_slot_attention(query_embed, query_sum, query_mask)
        reasoning_slots = reasoning_slots.squeeze(dim=1)
        logits = self.inference_module(reasoning_slots, TPR)

        if reconstruction:
            auxiliary = {}

            embed = torch.cat((story_embed, query_embed), dim=1)  # [b, s+1, w, e]
            recon_logits = self.reconstruction_linear(embed)
            recon_target = torch.cat((story, query.unsqueeze(dim=1)), dim=1)  # [b, s+1, w]

            recon_loss = self.recon_fn(recon_logits.permute(0, 3, 1, 2), recon_target)
            recon_loss = recon_loss

            auxiliary['recon_loss'] = recon_loss.mean()

            return logits, auxiliary

        return logits


class InputModule(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super(InputModule, self).__init__()
        self.word_embed = nn.Embedding(num_embeddings=config["vocab_size"],
                                       embedding_dim=config["symbol_size"])
        nn.init.uniform_(self.word_embed.weight, -config["init_limit"], config["init_limit"])
        # positional embeddings
        self.pos_embed = nn.Parameter(torch.ones(config["max_seq"], config["symbol_size"]))
        nn.init.ones_(self.pos_embed.data)
        self.pos_embed.data /= config["max_seq"]

    def forward(self, story: torch.Tensor, query: torch.Tensor):
        # Sentence embedding
        sentence_embed = self.word_embed(story)  # [b, s, w, e]
        sentence_embed = torch.einsum('bswe,we->bswe', sentence_embed, self.pos_embed[:sentence_embed.shape[2]])
        sentence_mask = (story != 0)
        sentence_sum = torch.einsum("bswe,bsw->bse", sentence_embed, sentence_mask.type(torch.float))

        # Query embedding
        query_embed = self.word_embed(query)  # [b, w, e]
        query_embed = torch.einsum('bwe,we->bwe', query_embed, self.pos_embed[:query_embed.shape[1]])
        query_mask = (query != 0)
        query_sum = torch.einsum("bwe,bw->be", query_embed, query_mask.type(torch.float))
        return sentence_embed, query_embed, sentence_mask, query_mask, sentence_sum, query_sum


class Decomposition(nn.Module):
    def __init__(self,  input_size, slot_size, mlp_hidden_size,
                        binding_num_slots,
                        reasoning_num_slots,
                        num_iterations=3,
                        num_head=4,
                        epsilon=1e-6,
        ):
        super(Decomposition, self).__init__()
    
        self.input_size = input_size
        self.binding_num_slots = binding_num_slots
        self.reasoning_num_slots = reasoning_num_slots
        self.num_slots = max(binding_num_slots, reasoning_num_slots)
        self.slot_size = slot_size
        self.mlp_hidden_size = mlp_hidden_size
        self.num_iterations = num_iterations
        self.epsilon = epsilon

        self.ln = LayerNorm(self.slot_size)

        assert slot_size % num_head == 0

        self.norm_input = nn.LayerNorm(self.input_size)
        self.norm_slots = nn.LayerNorm(self.slot_size)
        self.norm_pre_ff = nn.LayerNorm(self.slot_size)

        self.to_q = nn.Linear(self.slot_size, self.slot_size)
        self.to_k = nn.Linear(self.input_size, self.slot_size)
        self.to_v = nn.Linear(self.input_size, self.slot_size)
        self.to_linear = nn.Sequential(
            nn.Linear(self.slot_size, self.mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_size, self.slot_size)
        )

        self.binding_linear = nn.Linear(self.input_size, self.binding_num_slots * self.slot_size)
        self.reasoning_linear = nn.Linear(self.input_size, self.reasoning_num_slots * self.slot_size)

        self.slot_proj_linear = nn.Linear(self.slot_size * 2, self.slot_size)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, inputs):
        """
            inputs:         [seq_len, batch_size, word_len, input_size]
            inputs_mask:    [seq_len, batch_size, word_len]
        """
        
        return inputs
    
    def binding_slot_attention(self, sentence, sentence_sum, mask):

        batch_size, seq_len = sentence.shape[0], sentence.shape[1]

        initial_slots = self.binding_linear(sentence_sum)
        initial_slots = initial_slots.view(batch_size, seq_len, -1, self.slot_size)

        mask = mask.unsqueeze(dim=-1)     # [batch_size, seq_len, word_len, 1]
        mask = -1e6 * (1 - mask.type(torch.float))

        k, v = self.to_k(sentence), self.to_v(sentence)
        k = F.elu(k, 1., False) + 1.

        slots = initial_slots

        scale = 1. / self.slot_size
        for _ in range(self.num_iterations):

            q = self.to_q(slots) + initial_slots
            q = q * (self.slot_size ** -0.5)
            q = F.elu(q, 1., False) + 1.

            attn_logits = torch.einsum("nbis,nbjs->nbij", k, q)     # [seq_len, batch_size, word_len, num_slots]
            attn_logits = attn_logits + mask
            attn = F.softmax(attn_logits, dim=3)

            attn = attn + self.epsilon
            attn = attn / attn.sum(dim=2, keepdim=True)
            updates = torch.einsum("nbij,nbik->nbjk", attn, v)      # [seq_len, batch_size, num_slots, slot_size]
            
            slots = slots + self.to_linear(self.norm_pre_ff(updates)) * scale

        slots = torch.cat((self.dropout(initial_slots), slots), dim=-1)
        slots = self.slot_proj_linear(slots)

        return slots
    
    def reasoning_slot_attention(self, sentence, sentence_sum, mask):

        batch_size, seq_len = sentence.shape[0], sentence.shape[1]

        initial_slots = self.reasoning_linear(sentence_sum)
        initial_slots = initial_slots.view(batch_size, seq_len, -1, self.slot_size)

        mask = mask.unsqueeze(dim=-1)     # [batch_size, seq_len, word_len, 1]
        mask = -1e6 * (1 - mask.type(torch.float))

        k, v = self.to_k(sentence), self.to_v(sentence)
        k = F.elu(k, 1., False) + 1.

        slots = initial_slots

        scale = 1. / self.slot_size
        for _ in range(self.num_iterations):

            q = self.to_q(slots) + initial_slots
            q = q * (self.slot_size ** -0.5)
            q = F.elu(q, 1., False) + 1.

            attn_logits = torch.einsum("nbis,nbjs->nbij", k, q)     # [seq_len, batch_size, word_len, num_slots]
            attn_logits = attn_logits + mask
            attn = F.softmax(attn_logits, dim=3)

            attn = attn + self.epsilon
            attn = attn / attn.sum(dim=2, keepdim=True)
            updates = torch.einsum("nbij,nbik->nbjk", attn, v)      # [seq_len, batch_size, num_slots, slot_size]
            
            slots = slots + self.to_linear(self.norm_pre_ff(updates)) * scale

        slots = torch.cat((self.dropout(initial_slots), slots), dim=-1)
        slots = self.slot_proj_linear(slots)

        return slots


class UpdateModule(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super(UpdateModule, self).__init__()
        self.role_size = config["role_size"]
        self.ent_size = config["entity_size"]
        self.hidden_size = config["hidden_size"]
        self.symbol_size = config["symbol_size"]

        self.slot_size = config["slot_size"]
        epsilon = 1e-6

        self.e1_linear = nn.Linear(self.slot_size, self.ent_size)
        self.e2_linear = nn.Linear(self.slot_size, self.ent_size)

        self.r1_linear = nn.Linear(self.slot_size, self.role_size)
        self.r2_linear = nn.Linear(self.slot_size, self.role_size)
        self.r3_linear = nn.Linear(self.slot_size, self.role_size)

        self.attention_mask = torch.tensor([1.-4*epsilon, epsilon, epsilon, epsilon, epsilon])

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        # sentence: [b, s, w, e]
        batch_size = slots.size(0)

        attention_mask = self.attention_mask.to(slots.device)
        
        e1 = torch.tanh(self.e1_linear(torch.einsum("bsne,n->bse", slots, torch.roll(attention_mask, 0))))
        e2 = torch.tanh(self.e2_linear(torch.einsum("bsne,n->bse", slots, torch.roll(attention_mask, 1))))

        r1 = torch.tanh(self.r1_linear(torch.einsum("bsne,n->bse", slots, torch.roll(attention_mask, 2))))
        r2 = torch.tanh(self.r2_linear(torch.einsum("bsne,n->bse", slots, torch.roll(attention_mask, 3))))
        r3 = torch.tanh(self.r3_linear(torch.einsum("bsne,n->bse", slots, torch.roll(attention_mask, 4))))

        partial_add_W = torch.einsum('bsr,bsf->bsrf', r1, e2)
        partial_add_B = torch.einsum('bsr,bsf->bsrf', r3, e1)

        inputs = (e1, r1, partial_add_W, e2, r2, partial_add_B, r3)

        # TPR-RNN steps
        TPR = torch.zeros(batch_size, self.ent_size, self.role_size, self.ent_size).to(slots.device)
        for x in zip(*[torch.unbind(t, dim=1) for t in inputs]):
            e1_i, r1_i, partial_add_W_i, e2_i, r2_i, partial_add_B_i, r3_i = x
            w_hat = torch.einsum('be,br,berf->bf', e1_i, r1_i, TPR)
            partial_remove_W = torch.einsum('br,bf->brf', r1_i, w_hat)

            m_hat = torch.einsum('be,br,berf->bf', e1_i, r2_i, TPR)
            partial_remove_M = torch.einsum('br,bf->brf', r2_i, m_hat)
            partial_add_M = torch.einsum('br,bf->brf', r2_i, w_hat)

            b_hat = torch.einsum('be,br,berf->bf', e2_i, r3_i, TPR)
            partial_remove_B = torch.einsum('br,bf->brf', r3_i, b_hat)

            # operations
            write_op = partial_add_W_i - partial_remove_W
            move_op = partial_add_M - partial_remove_M
            backlink_op = partial_add_B_i - partial_remove_B
            delta_F = torch.einsum('be,brf->berf', e1_i, write_op + move_op) + \
                      torch.einsum('be,brf->berf', e2_i, backlink_op)
            delta_F = torch.clamp(delta_F, -1., 1.)
            TPR = TPR + delta_F
        return TPR


class InferenceModule(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super(InferenceModule, self).__init__()
        self.hidden_size = config["hidden_size"]
        self.ent_size = config["entity_size"]
        self.role_size = config["role_size"]
        self.symbol_size = config["symbol_size"]
        # output embeddings
        self.Z = nn.Linear(config["entity_size"], config["vocab_size"])

        self.slot_size = config["slot_size"]
        epsilon = 1e-6

        self.attention_mask = torch.tensor([1.-3*epsilon, epsilon, epsilon, epsilon])

        self.e1_linear = nn.Linear(self.slot_size, self.ent_size)

        self.r1_linear = nn.Linear(self.slot_size, self.role_size)
        self.r2_linear = nn.Linear(self.slot_size, self.role_size)
        self.r3_linear = nn.Linear(self.slot_size, self.role_size)

        self.l1, self.l2, self.l3 = [OptionalLayer(LayerNorm(hidden_size=self.ent_size), active=config["LN"])
                                     for _ in range(3)]

    def forward(self, slots: torch.Tensor, TPR: torch.Tensor):
        
        attention_mask = self.attention_mask.to(slots.device)

        e1 = self.e1_linear(torch.einsum("bne,n->be", slots, torch.roll(attention_mask, 0)))

        r1 = self.r1_linear(torch.einsum("bne,n->be", slots, torch.roll(attention_mask, 1)))
        r2 = self.r2_linear(torch.einsum("bne,n->be", slots, torch.roll(attention_mask, 2)))
        r3 = self.r3_linear(torch.einsum("bne,n->be", slots, torch.roll(attention_mask, 3)))

        i1 = self.l1(torch.einsum('be,br,berf->bf', e1, r1, TPR))
        i2 = self.l2(torch.einsum('be,br,berf->bf', i1, r2, TPR))
        i3 = self.l3(torch.einsum('be,br,berf->bf', i2, r3, TPR))

        step_sum = i1 + i2 + i3
        logits = self.Z(step_sum)
        return logits
