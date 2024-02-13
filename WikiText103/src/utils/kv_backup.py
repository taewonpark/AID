# Fast weight layers using custom kernels.
# Many code duplications to be refactored!
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.fast_fast_weight import fast_weight_delta



@torch.jit.script
def elu_p1(x):
    return F.elu(x, 1., False) + 1.


@torch.jit.script
def sum_norm(x):
    return x / x.sum(-1, keepdim=True)


@torch.jit.script
def sum_norm_eps(x):
    return x / (x.sum(-1, keepdim=True) + 1e-5)


@torch.jit.script
def elu_p1_sum_norm(x):
    y = F.elu(x, 1., False) + 1.
    return y / y.sum(-1, keepdim=True)


@torch.jit.script
def elu_p1_sum_norm_eps(x):
    y = F.elu(x, 1., False) + 1.
    return y / (y.sum(-1, keepdim=True) + 1e-5)


# Linear Transformer version
# our update rule + Katharopoulos et al's ELU based attention
class DecompositionTransformerLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 pre_lnorm=False, eps=1e-5, layer_id=None, num_layer=None,
                 skip_attn_normalization=True):
        # skip_attn_normalization is now set to True by default, thus it can
        # be removed.
        # Originally, with skip_attn_normalization set to False,
        # we had a version of the model which applies attention normalization
        # to the output (but not when we retrieve with the key for removal).
        super(DecompositionTransformerLayer, self).__init__()
        print(f"Using DecompositionTransformerLayer {layer_id} -")

        assert layer_id is not None
        assert num_layer is not None
        self.layer_id = layer_id
        self.num_layer = num_layer

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.qkvb_net = nn.Linear(
            d_model, n_head * (3 * d_head + 1), bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm
        self.normalize_attn_scores = (not skip_attn_normalization)
        self.eps = eps

        # Decomposition parameters.
        self.num_iter = 3
        self.slot_size = self.d_head

        self.norm_pre_ff = nn.LayerNorm(self.d_head)

        self.input_proj_linear = nn.Linear(self.d_model, 2 * self.n_head * self.d_head)
        self.to_q = nn.Linear(self.d_head, self.d_head)
        self.to_k = nn.Linear(self.d_head, self.d_head)
        self.to_v = nn.Linear(self.d_head, self.d_head)
        self.to_linear = nn.Sequential(
            nn.Linear(self.d_head, self.d_head * 2),
            nn.ReLU(),
            nn.Linear(self.d_head * 2, self.d_head),
        )
        self.slot_proj_linear = nn.Linear(self.d_head * 2, self.d_head)
        self.slot_dropout = nn.Dropout(p=0.5)

        self.hidden_dropout = nn.Dropout(p=0.5)

    def slot_attention(self, inputs, initial_slots):
        # inputs: [seq_len, batch_size, n_head * d_head]

        seq_len, batch_size = inputs.size(0), inputs.size(1)

        inputs = self.input_proj_linear(inputs)
        inputs = inputs.view(seq_len, batch_size, -1, self.d_head)

        k, v = self.to_k(inputs), self.to_v(inputs)
        k = F.elu(k, 1., False) + 1.

        slots = initial_slots

        scale = 1. / self.slot_size
        for idx in range(self.num_iter):

            q = self.to_q(slots) + initial_slots
            q = q * (self.slot_size ** -0.5)
            q = F.elu(q, 1., False) + 1.

            attn_logits = torch.einsum("sbwe,sbne->sbwn", k, q)
            attn = F.softmax(attn_logits, dim=3)
            
            attn = attn + self.eps
            attn = attn / attn.sum(dim=2, keepdim=True)
            updates = torch.einsum("sbwn,sbwe->sbne", attn, v)

            slots = slots + self.to_linear(self.norm_pre_ff(updates)) * scale

        slots = torch.cat((self.slot_dropout(initial_slots), slots), dim=-1)
        slots = self.slot_proj_linear(slots)

        return slots  # [s, b, n, e]

    def forward(self, h, attn_mask=None, mems=None,
                carry_over_fast_weight=False):
        # multihead attention
        # shape h: (len, B, n_head * d_head)

        if self.pre_lnorm:
            # layer normalization
            h = self.layer_norm(h)

        slen, bsz, _ = h.size()

        # h = self.input_proj_linear(h)

        qkvb = self.qkvb_net(h)
        qkvb = qkvb.view(slen, bsz, self.n_head, 3 * self.d_head + 1)
        head_q, head_k, head_v, head_beta = torch.split(
            qkvb, (self.d_head,) * 3 + (1,), -1)
        head_beta = torch.sigmoid(head_beta)

        heads = torch.cat((head_k, head_v), dim=2)
        heads = self.slot_attention(h, heads)
        head_k, head_v = torch.split(heads, (self.n_head,) * 2, 2)

        # reshape to (B, heads, len, dim)
        head_q = head_q.permute(1, 2, 0, 3)
        head_k = head_k.permute(1, 2, 0, 3)
        head_v = head_v.permute(1, 2, 0, 3)
        head_beta = head_beta.permute(1, 2, 0, 3)

        # TODO add dropout here?
        # transform q and k
        head_q = F.elu(head_q, 1., False) + 1.
        head_k = F.elu(head_k, 1., False) + 1.

        # normalize k and q, crucial for stable training.
        head_k = head_k / head_k.sum(-1, keepdim=True)
        head_q = head_q / head_q.sum(-1, keepdim=True)

        if self.normalize_attn_scores:
            # another version would be:
            # head_k_beta = head_k * head_beta
            # denominator_acc = torch.cumsum(head_k_beta, dim=2)
            denominator_acc = torch.cumsum(head_k, dim=2)

        if mems is None:
            mem_fast_weights = torch.zeros(
                bsz, self.n_head, self.d_head, self.d_head,
                device=head_k.device)
        else:
            assert carry_over_fast_weight
            mem_fast_weights, fast_denom = mems
            # bsz can be smaller for the last batch
            mem_fast_weights = mem_fast_weights[:bsz]
            if self.normalize_attn_scores:
                denominator_acc = denominator_acc + fast_denom[:bsz]

        if self.normalize_attn_scores:
            denominator = torch.einsum(
                'lbij,lbij->lbi', denominator_acc, head_q).unsqueeze(-1)

        layer_out = fast_weight_delta(
            head_q, head_k, head_v, head_beta, mem_fast_weights)

        # shape (B, n_head, len, d_head)
        if self.normalize_attn_scores:
            layer_out = self.scale * layer_out / (denominator + self.eps)
        else:
            layer_out = self.scale * layer_out

        layer_out = layer_out.transpose(1, 2)

        layer_out = layer_out.reshape(
            bsz, slen, self.n_head * self.d_head)

        layer_out = layer_out.transpose(0, 1)

        # expect [qlen, B, n_head * d_head]

        # linear projection
        attn_out = self.o_net(layer_out)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = h + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(h + attn_out)

        if carry_over_fast_weight:
            # last values of accumulator should be carried over.
            # clone is needed as backward modifies the data of fast weight
            if self.normalize_attn_scores:
                new_k_acc = denominator_acc[:, :, -1, :].unsqueeze(2).detach()
            else:
                new_k_acc = None
            new_mem = (mem_fast_weights.clone().detach(), new_k_acc)
            return output, new_mem

        return output
