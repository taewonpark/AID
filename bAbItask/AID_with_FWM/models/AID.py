# (1) Sequential Context:   (a) Simple RNN (LSTM)
# (2) Decomposition:        (a) RMC | Slot-based form (Role/Filer)
# (3) Association Binding   (a) Role/Filer form
# (4) Association Reasoning (a) Role/Filer form

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SequentialContext(nn.Module):
    def __init__(self, input_size, hidden_size, num_hidden=2):
        super(SequentialContext, self).__init__()
    
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_hidden = num_hidden

        self.rnn = nn.LSTM(self.input_size, self.hidden_size, self.num_hidden, dropout=0)

    def init_hidden(self, batch_size, device):
        rnn_init_hidden = (
            torch.zeros(self.num_hidden, batch_size, self.hidden_size).to(device),
            torch.zeros(self.num_hidden, batch_size, self.hidden_size).to(device),
        )
        return rnn_init_hidden

    def forward(self, inputs):
        
        max_length, batch_size = inputs.size(0), inputs.size(1)

        prev_states = self.init_hidden(batch_size, inputs.device)
        output, _ = self.rnn(inputs, prev_states)

        return output


class Decomposition(nn.Module):
    def __init__(self,  input_size, num_input, slot_size, mlp_hidden_size,
                        binding_num_slots,
                        reasoning_num_slots,
                        num_iterations=3,
                        epsilon=1e-8,
        ):
        super(Decomposition, self).__init__()
    
        self.input_size = input_size
        self.num_input = num_input
        self.binding_num_slots = binding_num_slots
        self.reasoning_num_slots = reasoning_num_slots
        self.slot_size = slot_size
        self.mlp_hidden_size = mlp_hidden_size
        self.num_iterations = num_iterations
        self.epsilon = epsilon

        self.norm_pre_ff = nn.LayerNorm(self.slot_size)

        self.to_q = nn.Linear(self.slot_size, self.slot_size)
        self.to_k = nn.Linear(self.input_size, self.slot_size)
        self.to_v = nn.Linear(self.input_size, self.slot_size)
        self.to_linear = nn.Sequential(
            nn.Linear(self.slot_size, self.mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_size, self.slot_size)
        )

        self.binding_linear = nn.Linear(self.num_input * self.input_size, self.binding_num_slots * self.slot_size)
        self.reasoning_linear = nn.Linear(self.num_input * self.input_size, self.reasoning_num_slots * self.slot_size)

        self.slot_proj_linear = nn.Linear(self.slot_size * 2, self.slot_size)

        self.dropout = nn.Dropout(p=0.5)

    def forward(self, inputs):
        """
            inputs:   [batch_size, input_size]
        """
        seq_len, batch_size = inputs.shape[0], inputs.shape[1]

        return inputs, inputs.view(seq_len, batch_size, -1)
    
    def binding_slot_attention(self, inputs, inputs_cat):

        seq_len, batch_size = inputs.shape[0], inputs.shape[1]

        initial_slots = self.binding_linear(inputs_cat)
        initial_slots = initial_slots.view(seq_len, batch_size, -1, self.slot_size)
        
        k, v = self.to_k(inputs), self.to_v(inputs)
        k = F.elu(k, 1., False) + 1.

        slots = initial_slots

        scale = 1. / self.slot_size
        for _ in range(self.num_iterations):

            q = self.to_q(slots) + initial_slots
            q = q * (self.slot_size ** -0.5)
            q = F.elu(q, 1., False) + 1.

            attn_logits = torch.einsum("nbis,nbjs->nbij", k, q)     # [seq_len, batch_size, word_len, num_slots]
            attn_logits = attn_logits
            attn = F.softmax(attn_logits, dim=3)

            attn = attn + self.epsilon
            attn = attn / attn.sum(dim=2, keepdim=True)
            updates = torch.einsum("nbij,nbik->nbjk", attn, v)      # [seq_len, batch_size, num_slots, slot_size]
            
            slots = slots + self.to_linear(self.norm_pre_ff(updates)) * scale

        slots = torch.cat((self.dropout(initial_slots), slots), dim=-1)
        slots = self.slot_proj_linear(slots)

        return slots

    def reasoning_slot_attention(self, inputs, inputs_cat):

        seq_len, batch_size = inputs.shape[0], inputs.shape[1]

        initial_slots = self.reasoning_linear(inputs_cat)
        initial_slots = initial_slots.view(seq_len, batch_size, -1, self.slot_size)

        k, v = self.to_k(inputs), self.to_v(inputs)
        k = F.elu(k, 1., False) + 1.

        slots = initial_slots

        scale = 1. / self.slot_size
        for _ in range(self.num_iterations):

            q = self.to_q(slots) + initial_slots
            q = q * (self.slot_size ** -0.5)
            q = F.elu(q, 1., False) + 1.

            attn_logits = torch.einsum("nbis,nbjs->nbij", k, q)     # [seq_len, batch_size, word_len, num_slots]
            attn_logits = attn_logits
            attn = F.softmax(attn_logits, dim=3)

            attn = attn + self.epsilon
            attn = attn / attn.sum(dim=2, keepdim=True)
            updates = torch.einsum("nbij,nbik->nbjk", attn, v)      # [seq_len, batch_size, num_slots, slot_size]
            
            slots = slots + self.to_linear(self.norm_pre_ff(updates)) * scale

        slots = torch.cat((self.dropout(initial_slots), slots), dim=-1)
        slots = self.slot_proj_linear(slots)

        return slots


class AssociativeBinding(nn.Module):
    def __init__(self, input_size, hidden_size, mem_size):
        super(AssociativeBinding, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.mem_size = mem_size

        self.linear_memory_r1 = nn.Linear(input_size, mem_size)
        nn.init.xavier_normal_(self.linear_memory_r1.weight)
        self.linear_memory_r2 = nn.Linear(input_size, mem_size)
        nn.init.xavier_normal_(self.linear_memory_r2.weight)
        self.linear_memory_f = nn.Linear(input_size, mem_size)
        nn.init.xavier_normal_(self.linear_memory_f.weight)

        self.linear_write_gate = nn.Linear(hidden_size, 1)
        nn.init.xavier_normal_(self.linear_write_gate.weight)

        epsilon = 1e-6
        self.attention_mask = torch.tensor([1.-2*epsilon, epsilon, epsilon])
    
    def prepare(self, slots):

        attention_mask = self.attention_mask.to(slots.device)

        role1 = torch.einsum("sbne,n->sbe", slots, torch.roll(attention_mask, 0))
        role2 = torch.einsum("sbne,n->sbe", slots, torch.roll(attention_mask, 1))
        filer = torch.einsum("sbne,n->sbe", slots, torch.roll(attention_mask, 2))

        role1 = torch.tanh(role1)
        role2 = torch.tanh(role2)
        filer = torch.tanh(filer)
        # role1 = torch.tanh(self.linear_memory_r1(role1))
        # role2 = torch.tanh(self.linear_memory_r2(role2))
        # filer = torch.tanh(self.linear_memory_f(filer))

        return role1, role2, filer

    def forward(self, memory_state, hidden_state, role1, role2, filer):
        """
            memory_state:   [batch_size, mem_size, mem_size, mem_size]
            role1:           [batch_size, input_size]            (decomposition)
            role2:           [batch_size, input_size]            (decomposition)
            filer:          [batch_size, input_size]            (decomposition)
            hidden_state:   [batch_size, hidden_size]           (sequential context) -> gate information
        """
        
        write_gate = self.linear_write_gate(hidden_state)
        write_gate = torch.sigmoid(write_gate + 1)

        role = torch.einsum("br,bt->brt", role1, role2)
        prev_info = torch.einsum("brt,brtf->bf", role, memory_state)
        cur_info = write_gate * (filer - prev_info)

        scale = 1. / self.mem_size
        new_memory_state = memory_state + torch.einsum("brt,bf->brtf", role, cur_info * scale)

        memory_norm = new_memory_state.view(new_memory_state.shape[0], -1).norm(dim=-1)
        memory_norm = torch.relu(memory_norm - 1) + 1
        new_memory_state = new_memory_state / memory_norm.unsqueeze(1).unsqueeze(1).unsqueeze(1)

        return new_memory_state


class AssociativeReasoning(nn.Module):
    def __init__(self, input_size, mem_size, n_read):
        super(AssociativeReasoning, self).__init__()
    
        self.input_size = input_size
        self.mem_size = mem_size
        self.n_read = n_read

        self.linear_memory_u1 = nn.Linear(input_size, mem_size)
        nn.init.xavier_normal_(self.linear_memory_u1.weight)
        self.linear_memory_u2 = nn.ModuleList([nn.Linear(input_size, mem_size) for _ in range(n_read)])
        for i in range(n_read):
            nn.init.xavier_normal_(self.linear_memory_u2[i].weight)

        self.ln = nn.LayerNorm(self.mem_size, elementwise_affine=False)

        epsilon = 1e-6
        self.attention_mask = torch.tensor([1.-(1+n_read)*epsilon] + [epsilon] * n_read)
    
    def prepare(self, slots):

        attention_mask = self.attention_mask.to(slots.device)

        unbinding1 = torch.einsum("sbne,n->sbe", slots, torch.roll(attention_mask, 0))
        unbinding1 = torch.tanh(unbinding1)
        # unbinding1 = self.linear_memory_u1(unbinding1)
        
        unbinding2 = []
        for i in range(self.n_read):
            u = torch.einsum("sbne,n->sbe", slots, torch.roll(attention_mask, i+1))
            u = torch.tanh(u)
            # u = self.linear_memory_u2[i](u)
            unbinding2.append(u)
        unbinding2 = torch.stack(unbinding2, dim=1)

        return unbinding1, unbinding2

    def forward(self, memory_state, unbinding1, unbinding2):
        """
            memory_state:   [batch_size, mem_size, mem_size, mem_size]
            unbinding1:      [batch_size, input_size]
            unbinding2:      [batch_size, n_read, input_size]
        """

        unbinding = unbinding1

        for i in range(self.n_read):
            unbinding = torch.einsum("bsrv,bs,br->bv", memory_state, unbinding, unbinding2[i])
            unbinding = self.ln(unbinding)

        return unbinding


class Network(nn.Module):
    def __init__(
        self, input_size, hidden_size, output_size, vocab_size,
        input_proj_size=64,
        num_input=3,
        num_hidden=2,
        slot_size=32,
        mlp_hidden_size=64,
        num_iterations=3,
        mem_size=32,
        n_read=1,
        batch_first=True,
    ):
        super(Network, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.mem_size = mem_size
        self.batch_first = batch_first
        self.vocab_size = vocab_size
        self.n_read = n_read
        self.slot_size = slot_size
        self.num_input = num_input

        self.input_proj_size = input_proj_size

        self.sequential_context = SequentialContext(
            input_size=self.input_proj_size,
            hidden_size=self.hidden_size,
            num_hidden=num_hidden,
        )
        self.decomposition = Decomposition(
            input_size=self.hidden_size,
            num_input=num_input,
            binding_num_slots=3,
            reasoning_num_slots=1 + n_read,
            slot_size=self.slot_size,
            mlp_hidden_size=mlp_hidden_size,
            num_iterations=num_iterations,
        )
        self.binding = AssociativeBinding(
            input_size=self.decomposition.slot_size,
            hidden_size=self.num_input * self.hidden_size,
            mem_size=self.mem_size,
        )
        self.reasoning = AssociativeReasoning(
            input_size=self.decomposition.slot_size,
            mem_size=self.mem_size,
            n_read=n_read,
        )
        self.hidden_proj_linear = nn.Linear(self.num_input * self.hidden_size, self.hidden_size)
        self.output_proj_linear = nn.Linear(self.mem_size, self.hidden_size)
        self.output_layer = nn.Linear(self.hidden_size, self.output_size)
        self.dropout = nn.Dropout(p=0.5)

        self.embedding = Embedding(vocab_size=vocab_size, embedding_size=input_size)
        self.reconstruction_linear = nn.Linear(input_size, vocab_size)

        self.input_proj_linear = nn.Linear(self.input_size, self.num_input * self.input_proj_size)

        self.recon_fn = nn.CrossEntropyLoss(reduction='mean', ignore_index=0)

    def init_memory_state(self, batch_size, device):
        
        memory_init_hidden = torch.zeros(batch_size, self.mem_size, self.mem_size, self.mem_size).to(device)
        
        return memory_init_hidden

    def forward(self, inputs, reconstruction=False):
        """
            inputs:   [batch_size, seq_len, input_size]
        """
        """
            self.sequential_context:        [inputs, prev_states, sequence_length]
            self.decomposition:             [inputs]
            self.binding:                   [memory_state, role, filer, hidden_state]
            self.reasoning:                 [memory_state, unbinding]
        """

        if self.batch_first:
            inputs = inputs.transpose(0, 1)

        seq_len, batch_size = inputs.size(0), inputs.size(1)
        inputs_embed = self.embedding(inputs)
        inputs_proj = self.input_proj_linear(inputs_embed)  # [seq_len, batch_size, num_input * slot_size]
        inputs_proj = inputs_proj.view(seq_len, batch_size, -1, self.input_proj_size)

        # Sequential Context
        sequential_context_inputs = inputs_proj.view(seq_len, -1, self.input_proj_size)
        sequential_context = self.sequential_context(sequential_context_inputs)
        sequential_context = sequential_context.view(seq_len, batch_size, -1, self.hidden_size)

        # Decomposition
        sequential_context, sequential_context_cat = self.decomposition(sequential_context)
        bindings = self.decomposition.binding_slot_attention(sequential_context, sequential_context_cat)
        reasonings = self.decomposition.reasoning_slot_attention(sequential_context, sequential_context_cat)

        memory_state = self.init_memory_state(batch_size, inputs.device)

        role1, role2, filer = self.binding.prepare(bindings)
        unbinding1, unbinding2 = self.reasoning.prepare(reasonings)

        outputs = []
        for t, context in enumerate(sequential_context_cat):

            # Association Binding
            memory_state = self.binding(memory_state, context, role1[t], role2[t], filer[t])

            # Association Reasoning
            output_t = self.reasoning(memory_state, unbinding1[t], unbinding2[t])

            outputs.append(output_t)

        outputs = torch.stack(outputs, dim=0)
        outputs = self.dropout(self.hidden_proj_linear(sequential_context_cat)) + self.output_proj_linear(outputs)
        final_outputs = self.output_layer(outputs)

        if self.batch_first:
            final_outputs = final_outputs.transpose(0, 1)

        if reconstruction:
            loss = 0.

            if self.batch_first:
                inputs = inputs.transpose(0, 1)
                inputs_embed = inputs_embed.transpose(0, 1)
            
            reconstruction = self.reconstruction_linear(inputs_embed)
            reconstruction_loss = self.recon_fn(reconstruction.transpose(1, 2), inputs)
            loss += reconstruction_loss

            return final_outputs, loss

        return final_outputs


class Embedding(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super(Embedding, self).__init__()
        self.word_embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size)
        nn.init.xavier_uniform_(self.word_embed.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, input_sequence):
        return self.word_embed(input_sequence)
