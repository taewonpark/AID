from sklearn.model_selection import train_test_split
from torchtext.vocab import GloVe
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup

import os
import pickle
import random
import math
import logging
import torch
import torch.nn as nn
import numpy as np


class SARtask(object):
    def __init__(self, batch_size,
                 num_sample=30,
                 ratio=None):

        self.batch_size = batch_size
        self.num_sample = num_sample

        self.ratio = ratio if ratio else 0.0
        assert self.ratio <= 1.0

        self._build_var()

    def _build_var(self):

        dataset = load_dataset()

        self.vocab = dataset['dictionary']

        X1_list = dataset['list']['X1']
        X2_list = dataset['list']['X2']

        x1_split_idx = int(len(X1_list) * self.ratio)

        self.train_data, self.test_data = dict(), dict()
        
        data = dataset['data']

        for x in X1_list[:x1_split_idx]:
            self.train_data[x] = data[x]['y1'] + data[x]['y2']
        for x in X1_list[x1_split_idx:]:
            self.train_data[x] = data[x]['y2']
        for x in X2_list:
            self.train_data[x] = data[x]['y1']
            self.test_data[x] = data[x]['y2']

    def __iter__(self):
        return self

    def __next__(self):

        x_list = list(self.train_data.keys())

        length = np.ones([self.batch_size], np.int32) * self.num_sample
        max_length = max(length) + 1

        enc_seq_len = np.ones([self.batch_size], np.int32) * self.num_sample + 1
        dec_seq_len = length + 1
        s_l = max(enc_seq_len + dec_seq_len)

        roles = []
        filers = []

        # input flag: 0, output flag: 1
        flags = torch.zeros(self.batch_size, s_l, 2)
        flags[:, 0, 0] = 1
        flags[:, self.num_sample + 1, 1] = 1

        for l in length:

            input_role = np.zeros([s_l], dtype=np.int32)
            input_filer = np.zeros([s_l], dtype=np.int32)

            # select X
            sampled_x = list(np.random.choice(x_list, size=self.num_sample, replace=False))
            sampled_x_idx = [self.vocab[x] for x in sampled_x]
            input_role[1:self.num_sample + 1] = sampled_x_idx
 
            sampled_y = []
            for x in sampled_x:
                _, y = random.choice(self.train_data[x])
                sampled_y.append(y)
            sampled_y_idx = [self.vocab[y] for y in sampled_y]
            input_filer[1:self.num_sample + 1] = sampled_y_idx

            # select object from picked objects.
            query_index = list(np.random.choice(range(self.num_sample), size=l, replace=False))
            input_role[self.num_sample + 2:self.num_sample + 2 + l] = input_role[1:self.num_sample + 1][query_index]
            input_filer[self.num_sample + 2:self.num_sample + 2 + l] = input_filer[1:self.num_sample + 1][query_index]

            roles.append(input_role)
            filers.append(input_filer)

        roles = torch.LongTensor(np.array(roles))
        filers = torch.LongTensor(np.array(filers))

        enc_inputs = (
            flags[:, :-max_length],
            roles[:, :-max_length],
            filers[:, :-max_length])
        dec_inputs = (
            flags[:, -max_length:],
            roles[:, -max_length:],
            torch.zeros_like(filers[:, -max_length:]))
        outputs = filers[:, -max_length:]

        return (
            enc_inputs,       # Encoder input.
            dec_inputs,       # Decoder input.
            outputs,
        )

    @property
    def output_size(self):
        return len(self.vocab)

    def get_test_samples(self):

        x_list = list(self.test_data.keys())
        num_sample = len(x_list)

        batch_size = len(self.test_data[x_list[0]])
        length = np.ones([batch_size], np.int32) * num_sample
        max_length = max(length) + 1

        enc_seq_len = np.ones([batch_size], np.int32) * num_sample + 1
        dec_seq_len = length + 1
        s_l = max(enc_seq_len + dec_seq_len)

        roles = []
        filers = []

        # input flag: 0, output flag: 1
        flags = torch.zeros(batch_size, s_l, 2)
        flags[:, 0, 0] = 1
        flags[:, num_sample + 1, 1] = 1

        for b_idx, l in enumerate(length):

            input_role = np.zeros([s_l], dtype=np.int32)
            input_filer = np.zeros([s_l], dtype=np.int32)

            # select X
            sampled_x = x_list
            sampled_x_idx = [self.vocab[x] for x in sampled_x]
            input_role[1:num_sample + 1] = sampled_x_idx

            sampled_y = []
            for x in sampled_x:
                _, y = self.test_data[x][b_idx]
                sampled_y.append(y)
            sampled_y_idx = [self.vocab[y] for y in sampled_y]
            input_filer[1:num_sample + 1] = sampled_y_idx

            # select object from picked objects.
            query_index = list(range(num_sample))
            input_role[num_sample + 2:num_sample + 2 + l] = input_role[1:num_sample + 1][query_index]
            input_filer[num_sample + 2:num_sample + 2 + l] = input_filer[1:num_sample + 1][query_index]

            roles.append(input_role)
            filers.append(input_filer)

        roles = torch.LongTensor(np.array(roles))
        filers = torch.LongTensor(np.array(filers))

        enc_inputs = (
            flags[:, :-max_length],
            roles[:, :-max_length],
            filers[:, :-max_length])
        dec_inputs = (
            flags[:, -max_length:],
            roles[:, -max_length:],
            torch.zeros_like(filers[:, -max_length:]))
        outputs = filers[:, -max_length:]

        return (
            enc_inputs,       # Encoder input.
            dec_inputs,       # Decoder input.
            outputs,
        )
    
    def get_fixed_x(self):

        x_list = list(self.test_data.keys())
        num_sample = len(x_list)

        batch_size = 1
        length = np.ones([batch_size], np.int32) * num_sample
        max_length = max(length) + 1

        enc_seq_len = np.ones([batch_size], np.int32) * num_sample + 1
        dec_seq_len = length + 1
        s_l = max(enc_seq_len + dec_seq_len)

        roles = []
        filers = []

        # input flag: 0, output flag: 1
        flags = torch.zeros(batch_size, s_l, 2)
        flags[:, 0, 0] = 1
        flags[:, num_sample + 1, 1] = 1

        for b_idx, l in enumerate(length):

            input_role = np.zeros([s_l], dtype=np.int32)
            input_filer = np.zeros([s_l], dtype=np.int32)

            # select X
            sampled_x = x_list
            sampled_x_idx = [self.vocab[sampled_x[0]]] * len(sampled_x)
            input_role[1:num_sample + 1] = sampled_x_idx

            sampled_y = []
            for x in sampled_x:
                _, y = self.test_data[x][b_idx]
                sampled_y.append(y)
            sampled_y_idx = [self.vocab[y] for y in sampled_y]
            input_filer[1:num_sample + 1] = sampled_y_idx

            # select object from picked objects.
            query_index = list(range(num_sample))
            input_role[num_sample + 2:num_sample + 2 + l] = input_role[1:num_sample + 1][query_index]
            input_filer[num_sample + 2:num_sample + 2 + l] = input_filer[1:num_sample + 1][query_index]

            roles.append(input_role)
            filers.append(input_filer)

        roles = torch.LongTensor(np.array(roles))
        filers = torch.LongTensor(np.array(filers))

        enc_inputs = (
            flags[:, :-max_length],
            roles[:, :-max_length],
            filers[:, :-max_length])
        dec_inputs = (
            flags[:, -max_length:],
            roles[:, -max_length:],
            torch.zeros_like(filers[:, -max_length:]))
        outputs = filers[:, -max_length:]

        return (
            enc_inputs,       # Encoder input.
            dec_inputs,       # Decoder input.
            outputs,
        )
    
    def get_fixed_y(self):

        x_list = list(self.test_data.keys())
        num_sample = len(x_list)

        batch_size = 1
        length = np.ones([batch_size], np.int32) * num_sample
        max_length = max(length) + 1

        enc_seq_len = np.ones([batch_size], np.int32) * num_sample + 1
        dec_seq_len = length + 1
        s_l = max(enc_seq_len + dec_seq_len)

        roles = []
        filers = []

        # input flag: 0, output flag: 1
        flags = torch.zeros(batch_size, s_l, 2)
        flags[:, 0, 0] = 1
        flags[:, num_sample + 1, 1] = 1

        for b_idx, l in enumerate(length):

            input_role = np.zeros([s_l], dtype=np.int32)
            input_filer = np.zeros([s_l], dtype=np.int32)

            # select X
            sampled_x = x_list
            sampled_x_idx = [self.vocab[x] for x in sampled_x]
            input_role[1:num_sample + 1] = sampled_x_idx

            _, y = self.test_data[sampled_x[0]][b_idx]
            sampled_y = [y] * len(sampled_x)
            sampled_y_idx = [self.vocab[y] for y in sampled_y]
            input_filer[1:num_sample + 1] = sampled_y_idx

            # select object from picked objects.
            query_index = list(range(num_sample))
            input_role[num_sample + 2:num_sample + 2 + l] = input_role[1:num_sample + 1][query_index]
            input_filer[num_sample + 2:num_sample + 2 + l] = input_filer[1:num_sample + 1][query_index]

            roles.append(input_role)
            filers.append(input_filer)

        roles = torch.LongTensor(np.array(roles))
        filers = torch.LongTensor(np.array(filers))

        enc_inputs = (
            flags[:, :-max_length],
            roles[:, :-max_length],
            filers[:, :-max_length])
        dec_inputs = (
            flags[:, -max_length:],
            roles[:, -max_length:],
            torch.zeros_like(filers[:, -max_length:]))
        outputs = filers[:, -max_length:]

        return (
            enc_inputs,       # Encoder input.
            dec_inputs,       # Decoder input.
            outputs,
        )
    
    def get_dci_samples(self, item_range):

        x_list = list(self.test_data.keys())[:item_range]
        num_sample = len(x_list)

        batch_size = item_range
        length = np.ones([batch_size], np.int32) * num_sample
        max_length = max(length) + 1

        enc_seq_len = np.ones([batch_size], np.int32) * num_sample + 1
        dec_seq_len = length + 1
        s_l = max(enc_seq_len + dec_seq_len)

        roles = []
        filers = []

        # input flag: 0, output flag: 1
        flags = torch.zeros(batch_size, s_l, 2)
        flags[:, 0, 0] = 1
        flags[:, num_sample + 1, 1] = 1

        sampled_y = []
        for i, j in self.test_data[x_list[0]][:item_range]:
            sampled_y.append(j)

        for b_idx, l in enumerate(length):

            input_role = np.zeros([s_l], dtype=np.int32)
            input_filer = np.zeros([s_l], dtype=np.int32)

            # select X
            sampled_x = x_list
            sampled_x_idx = [self.vocab[x] for x in sampled_x]
            input_role[1:num_sample + 1] = sampled_x_idx

            sampled_y_idx = [self.vocab[y] for y in np.roll(sampled_y, b_idx)]
            input_filer[1:num_sample + 1] = sampled_y_idx

            # select object from picked objects.
            query_index = list(range(num_sample))
            input_role[num_sample + 2:num_sample + 2 + l] = input_role[1:num_sample + 1][query_index]
            input_filer[num_sample + 2:num_sample + 2 + l] = input_filer[1:num_sample + 1][query_index]

            roles.append(input_role)
            filers.append(input_filer)

        roles = torch.LongTensor(np.array(roles))
        filers = torch.LongTensor(np.array(filers))

        enc_inputs = (
            flags[:, :-max_length],
            roles[:, :-max_length],
            filers[:, :-max_length])
        dec_inputs = (
            flags[:, -max_length:],
            roles[:, -max_length:],
            torch.zeros_like(filers[:, -max_length:]))
        outputs = filers[:, -max_length:]

        return (
            enc_inputs,       # Encoder input.
            dec_inputs,       # Decoder input.
            outputs,
        )
    
    def get_dci_samples1(self, item_range):

        x_list = list(self.test_data.keys())[:item_range]
        num_sample = len(x_list)

        batch_size = len(self.test_data[x_list[0]])
        length = np.ones([batch_size], np.int32) * num_sample
        max_length = max(length) + 1

        enc_seq_len = np.ones([batch_size], np.int32) * num_sample + 1
        dec_seq_len = length + 1
        s_l = max(enc_seq_len + dec_seq_len)

        roles = []
        filers = []

        # input flag: 0, output flag: 1
        flags = torch.zeros(batch_size, s_l, 2)
        flags[:, 0, 0] = 1
        flags[:, num_sample + 1, 1] = 1

        sampled_y = []
        for i, j in self.test_data[x_list[0]]:
            sampled_y.append(j)

        for b_idx, l in enumerate(length):

            input_role = np.zeros([s_l], dtype=np.int32)
            input_filer = np.zeros([s_l], dtype=np.int32)

            # select X
            sampled_x = x_list
            sampled_x_idx = [self.vocab[x] for x in sampled_x]
            input_role[1:num_sample + 1] = sampled_x_idx

            sampled_y_idx = [self.vocab[sampled_y[b_idx]] for _ in range(num_sample)]
            input_filer[1:num_sample + 1] = sampled_y_idx

            # select object from picked objects.
            query_index = list(range(num_sample))
            input_role[num_sample + 2:num_sample + 2 + l] = input_role[1:num_sample + 1][query_index]
            input_filer[num_sample + 2:num_sample + 2 + l] = input_filer[1:num_sample + 1][query_index]

            roles.append(input_role)
            filers.append(input_filer)

        roles = torch.LongTensor(np.array(roles))
        filers = torch.LongTensor(np.array(filers))

        enc_inputs = (
            flags[:, :-max_length],
            roles[:, :-max_length],
            filers[:, :-max_length])
        dec_inputs = (
            flags[:, -max_length:],
            roles[:, -max_length:],
            torch.zeros_like(filers[:, -max_length:]))
        outputs = filers[:, -max_length:]

        return (
            enc_inputs,       # Encoder input.
            dec_inputs,       # Decoder input.
            outputs,
        )
    
    def get_dci_samples2(self):

        item_range = 10
        x_list = list(self.test_data.keys())
        num_sample = len(x_list)

        batch_size = item_range
        length = np.ones([batch_size], np.int32) * num_sample
        max_length = max(length) + 1

        enc_seq_len = np.ones([batch_size], np.int32) * num_sample + 1
        dec_seq_len = length + 1
        s_l = max(enc_seq_len + dec_seq_len)

        roles = []
        filers = []

        # input flag: 0, output flag: 1
        flags = torch.zeros(batch_size, s_l, 2)
        flags[:, 0, 0] = 1
        flags[:, num_sample + 1, 1] = 1

        sampled_y = []
        for i, j in self.test_data[x_list[0]][:item_range]:
            sampled_y.append(j)

        for b_idx, l in enumerate(length):

            input_role = np.zeros([s_l], dtype=np.int32)
            input_filer = np.zeros([s_l], dtype=np.int32)

            # select X
            sampled_x = x_list
            sampled_x_idx = [self.vocab[x] for x in sampled_x]
            input_role[1:num_sample + 1] = sampled_x_idx

            sampled_y_idx = [self.vocab[sampled_y[b_idx]] for _ in range(num_sample)]
            input_filer[1:num_sample + 1] = sampled_y_idx

            # select object from picked objects.
            query_index = list(range(num_sample))
            input_role[num_sample + 2:num_sample + 2 + l] = input_role[1:num_sample + 1][query_index]
            input_filer[num_sample + 2:num_sample + 2 + l] = input_filer[1:num_sample + 1][query_index]

            roles.append(input_role)
            filers.append(input_filer)

        roles = torch.LongTensor(np.array(roles))
        filers = torch.LongTensor(np.array(filers))

        enc_inputs = (
            flags[:, :-max_length],
            roles[:, :-max_length],
            filers[:, :-max_length])
        dec_inputs = (
            flags[:, -max_length:],
            roles[:, -max_length:],
            torch.zeros_like(filers[:, -max_length:]))
        outputs = filers[:, -max_length:]

        return (
            enc_inputs,       # Encoder input.
            dec_inputs,       # Decoder input.
            outputs,
        )


def load_dataset():
    """
        1000 most common words in English (https://www.ef.com/wwen/english-resources/english-vocabulary/top-1000-words/)
        - 50 words at each line:

        'a', 'ability', 'able', 'about', 'above', 'accept', 'according', 'account', 'across', 'act', 'action', 'activity', 'actually', 'add', 'address', 'administration', 'admit', 'adult', 'affect', 'after', 'again', 'against', 'age', 'agency', 'agent', 'ago', 'agree', 'agreement', 'ahead', 'air', 'all', 'allow', 'almost', 'alone', 'along', 'already', 'also', 'although', 'always', 'American', 'among', 'amount', 'analysis', 'and', 'animal', 'another', 'answer', 'any', 'anyone', 'anything', 
        'appear', 'apply', 'approach', 'area', 'argue', 'arm', 'around', 'arrive', 'art', 'article', 'artist', 'as', 'ask', 'assume', 'at', 'attack', 'attention', 'attorney', 'audience', 'author', 'authority', 'available', 'avoid', 'away', 'baby', 'back', 'bad', 'bag', 'ball', 'bank', 'bar', 'base', 'be', 'beat', 'beautiful', 'because', 'become', 'bed', 'before', 'begin', 'behavior', 'behind', 'believe', 'benefit', 'best', 'better', 'between', 'beyond', 'big', 'bill', 
        'billion', 'bit', 'black', 'blood', 'blue', 'board', 'body', 'book', 'born', 'both', 'box', 'boy', 'break', 'bring', 'brother', 'budget', 'build', 'building', 'business', 'but', 'buy', 'by', 'call', 'camera', 'campaign', 'can', 'cancer', 'candidate', 'capital', 'car', 'card', 'care', 'career', 'carry', 'case', 'catch', 'cause', 'cell', 'center', 'central', 'century', 'certain', 'certainly', 'chair', 'challenge', 'chance', 'change', 'character', 'charge', 'check', 
        'child', 'choice', 'choose', 'church', 'citizen', 'city', 'civil', 'claim', 'class', 'clear', 'clearly', 'close', 'coach', 'cold', 'collection', 'college', 'color', 'come', 'commercial', 'common', 'community', 'company', 'compare', 'computer', 'concern', 'condition', 'conference', 'Congress', 'consider', 'consumer', 'contain', 'continue', 'control', 'cost', 'could', 'country', 'couple', 'course', 'court', 'cover', 'create', 'crime', 'cultural', 'culture', 'cup', 'current', 'customer', 'cut', 'dark', 'data', 
        'daughter', 'day', 'dead', 'deal', 'death', 'debate', 'decade', 'decide', 'decision', 'deep', 'defense', 'degree', 'Democrat', 'democratic', 'describe', 'design', 'despite', 'detail', 'determine', 'develop', 'development', 'die', 'difference', 'different', 'difficult', 'dinner', 'direction', 'director', 'discover', 'discuss', 'discussion', 'disease', 'do', 'doctor', 'dog', 'door', 'down', 'draw', 'dream', 'drive', 'drop', 'drug', 'during', 'each', 'early', 'east', 'easy', 'eat', 'economic', 'economy', 
        'edge', 'education', 'effect', 'effort', 'eight', 'either', 'election', 'else', 'employee', 'end', 'energy', 'enjoy', 'enough', 'enter', 'entire', 'environment', 'environmental', 'especially', 'establish', 'even', 'evening', 'event', 'ever', 'every', 'everybody', 'everyone', 'everything', 'evidence', 'exactly', 'example', 'executive', 'exist', 'expect', 'experience', 'expert', 'explain', 'eye', 'face', 'fact', 'factor', 'fail', 'fall', 'family', 'far', 'fast', 'father', 'fear', 'federal', 'feel', 'feeling', 
        'few', 'field', 'fight', 'figure', 'fill', 'film', 'final', 'finally', 'financial', 'find', 'fine', 'finger', 'finish', 'fire', 'firm', 'first', 'fish', 'five', 'floor', 'fly', 'focus', 'follow', 'food', 'foot', 'for', 'force', 'foreign', 'forget', 'form', 'former', 'forward', 'four', 'free', 'friend', 'from', 'front', 'full', 'fund', 'future', 'game', 'garden', 'gas', 'general', 'generation', 'get', 'girl', 'give', 'glass', 'go', 'goal', 
        'good', 'government', 'great', 'green', 'ground', 'group', 'grow', 'growth', 'guess', 'gun', 'guy', 'hair', 'half', 'hand', 'hang', 'happen', 'happy', 'hard', 'have', 'he', 'head', 'health', 'hear', 'heart', 'heat', 'heavy', 'help', 'her', 'here', 'herself', 'high', 'him', 'himself', 'his', 'history', 'hit', 'hold', 'home', 'hope', 'hospital', 'hot', 'hotel', 'hour', 'house', 'how', 'however', 'huge', 'human', 'hundred', 'husband', 
        'I', 'idea', 'identify', 'if', 'image', 'imagine', 'impact', 'important', 'improve', 'in', 'include', 'including', 'increase', 'indeed', 'indicate', 'individual', 'industry', 'information', 'inside', 'instead', 'institution', 'interest', 'interesting', 'international', 'interview', 'into', 'investment', 'involve', 'issue', 'it', 'item', 'its', 'itself', 'job', 'join', 'just', 'keep', 'key', 'kid', 'kill', 'kind', 'kitchen', 'know', 'knowledge', 'land', 'language', 'large', 'last', 'late', 'later', 
        'laugh', 'law', 'lawyer', 'lay', 'lead', 'leader', 'learn', 'least', 'leave', 'left', 'leg', 'legal', 'less', 'let', 'letter', 'level', 'lie', 'life', 'light', 'like', 'likely', 'line', 'list', 'listen', 'little', 'live', 'local', 'long', 'look', 'lose', 'loss', 'lot', 'love', 'low', 'machine', 'magazine', 'main', 'maintain', 'major', 'majority', 'make', 'man', 'manage', 'management', 'manager', 'many', 'market', 'marriage', 'material', 'matter', 
        'may', 'maybe', 'me', 'mean', 'measure', 'media', 'medical', 'meet', 'meeting', 'member', 'memory', 'mention', 'message', 'method', 'middle', 'might', 'military', 'million', 'mind', 'minute', 'miss', 'mission', 'model', 'modern', 'moment', 'money', 'month', 'more', 'morning', 'most', 'mother', 'mouth', 'move', 'movement', 'movie', 'Mr', 'Mrs', 'much', 'music', 'must', 'my', 'myself', 'name', 'nation', 'national', 'natural', 'nature', 'near', 'nearly', 'necessary', 
        'need', 'network', 'never', 'new', 'news', 'newspaper', 'next', 'nice', 'night', 'no', 'none', 'nor', 'north', 'not', 'note', 'nothing', 'notice', 'now', 'n't', 'number', 'occur', 'of', 'off', 'offer', 'office', 'officer', 'official', 'often', 'oh', 'oil', 'ok', 'old', 'on', 'once', 'one', 'only', 'onto', 'open', 'operation', 'opportunity', 'option', 'or', 'order', 'organization', 'other', 'others', 'our', 'out', 'outside', 'over', 
        'own', 'owner', 'page', 'pain', 'painting', 'paper', 'parent', 'part', 'participant', 'particular', 'particularly', 'partner', 'party', 'pass', 'past', 'patient', 'pattern', 'pay', 'peace', 'people', 'per', 'perform', 'performance', 'perhaps', 'period', 'person', 'personal', 'phone', 'physical', 'pick', 'picture', 'piece', 'place', 'plan', 'plant', 'play', 'player', 'PM', 'point', 'police', 'policy', 'political', 'politics', 'poor', 'popular', 'population', 'position', 'positive', 'possible', 'power', 
        'practice', 'prepare', 'present', 'president', 'pressure', 'pretty', 'prevent', 'price', 'private', 'probably', 'problem', 'process', 'produce', 'product', 'production', 'professional', 'professor', 'program', 'project', 'property', 'protect', 'prove', 'provide', 'public', 'pull', 'purpose', 'push', 'put', 'quality', 'question', 'quickly', 'quite', 'race', 'radio', 'raise', 'range', 'rate', 'rather', 'reach', 'read', 'ready', 'real', 'reality', 'realize', 'really', 'reason', 'receive', 'recent', 'recently', 'recognize', 
        'record', 'red', 'reduce', 'reflect', 'region', 'relate', 'relationship', 'religious', 'remain', 'remember', 'remove', 'report', 'represent', 'Republican', 'require', 'research', 'resource', 'respond', 'response', 'responsibility', 'rest', 'result', 'return', 'reveal', 'rich', 'right', 'rise', 'risk', 'road', 'rock', 'role', 'room', 'rule', 'run', 'safe', 'same', 'save', 'say', 'scene', 'school', 'science', 'scientist', 'score', 'sea', 'season', 'seat', 'second', 'section', 'security', 'see', 
        'seek', 'seem', 'sell', 'send', 'senior', 'sense', 'series', 'serious', 'serve', 'service', 'set', 'seven', 'several', 'sex', 'sexual', 'shake', 'share', 'she', 'shoot', 'short', 'shot', 'should', 'shoulder', 'show', 'side', 'sign', 'significant', 'similar', 'simple', 'simply', 'since', 'sing', 'single', 'sister', 'sit', 'site', 'situation', 'six', 'size', 'skill', 'skin', 'small', 'smile', 'so', 'social', 'society', 'soldier', 'some', 'somebody', 'someone', 
        'something', 'sometimes', 'son', 'song', 'soon', 'sort', 'sound', 'source', 'south', 'southern', 'space', 'speak', 'special', 'specific', 'speech', 'spend', 'sport', 'spring', 'staff', 'stage', 'stand', 'standard', 'star', 'start', 'state', 'statement', 'station', 'stay', 'step', 'still', 'stock', 'stop', 'store', 'story', 'strategy', 'street', 'strong', 'structure', 'student', 'study', 'stuff', 'style', 'subject', 'success', 'successful', 'such', 'suddenly', 'suffer', 'suggest', 'summer', 
        'support', 'sure', 'surface', 'system', 'table', 'take', 'talk', 'task', 'tax', 'teach', 'teacher', 'team', 'technology', 'television', 'tell', 'ten', 'tend', 'term', 'test', 'than', 'thank', 'that', 'the', 'their', 'them', 'themselves', 'then', 'theory', 'there', 'these', 'they', 'thing', 'think', 'third', 'this', 'those', 'though', 'thought', 'thousand', 'threat', 'three', 'through', 'throughout', 'throw', 'thus', 'time', 'to', 'today', 'together', 'tonight', 
        'too', 'top', 'total', 'tough', 'toward', 'town', 'trade', 'traditional', 'training', 'travel', 'treat', 'treatment', 'tree', 'trial', 'trip', 'trouble', 'true', 'truth', 'try', 'turn', 'TV', 'two', 'type', 'under', 'understand', 'unit', 'until', 'up', 'upon', 'us', 'use', 'usually', 'value', 'various', 'very', 'victim', 'view', 'violence', 'visit', 'voice', 'vote', 'wait', 'walk', 'wall', 'want', 'war', 'watch', 'water', 'way', 'we', 
        'weapon', 'wear', 'week', 'weight', 'well', 'west', 'western', 'what', 'whatever', 'when', 'where', 'whether', 'which', 'while', 'white', 'who', 'whole', 'whom', 'whose', 'why', 'wide', 'wife', 'will', 'win', 'wind', 'window', 'wish', 'with', 'within', 'without', 'woman', 'wonder', 'word', 'work', 'worker', 'world', 'worry', 'would', 'write', 'writer', 'wrong', 'yard', 'yeah', 'year', 'yes', 'yet', 'you', 'young', 'your', 'yourself', 
    """

    filename = 'SAR_dataset.pkl'

    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            dataset = pickle.load(f)
    else:
        X1 = [
            'a', 'ability', 'able', 'about', 'above', 'accept', 'according', 'account', 'across', 'act', 'action', 'activity', 'actually', 'add', 'address', 'administration', 'admit', 'adult', 'affect', 'after', 'again', 'against', 'age', 'agency', 'agent', 'ago', 'agree', 'agreement', 'ahead', 'air', 'all', 'allow', 'almost', 'alone', 'along', 'already', 'also', 'although', 'always', 'American', 'among', 'amount', 'analysis', 'and', 'animal', 'another', 'answer', 'any', 'anyone', 'anything', 
            'appear', 'apply', 'approach', 'area', 'argue', 'arm', 'around', 'arrive', 'art', 'article', 'artist', 'as', 'ask', 'assume', 'at', 'attack', 'attention', 'attorney', 'audience', 'author', 'authority', 'available', 'avoid', 'away', 'baby', 'back', 'bad', 'bag', 'ball', 'bank', 'bar', 'base', 'be', 'beat', 'beautiful', 'because', 'become', 'bed', 'before', 'begin', 'behavior', 'behind', 'believe', 'benefit', 'best', 'better', 'between', 'beyond', 'big', 'bill', 
            'billion', 'bit', 'black', 'blood', 'blue', 'board', 'body', 'book', 'born', 'both', 'box', 'boy', 'break', 'bring', 'brother', 'budget', 'build', 'building', 'business', 'but', 'buy', 'by', 'call', 'camera', 'campaign', 'can', 'cancer', 'candidate', 'capital', 'car', 'card', 'care', 'career', 'carry', 'case', 'catch', 'cause', 'cell', 'center', 'central', 'century', 'certain', 'certainly', 'chair', 'challenge', 'chance', 'change', 'character', 'charge', 'check', 
            'child', 'choice', 'choose', 'church', 'citizen', 'city', 'civil', 'claim', 'class', 'clear', 'clearly', 'close', 'coach', 'cold', 'collection', 'college', 'color', 'come', 'commercial', 'common', 'community', 'company', 'compare', 'computer', 'concern', 'condition', 'conference', 'Congress', 'consider', 'consumer', 'contain', 'continue', 'control', 'cost', 'could', 'country', 'couple', 'course', 'court', 'cover', 'create', 'crime', 'cultural', 'culture', 'cup', 'current', 'customer', 'cut', 'dark', 'data', 
            'daughter', 'day', 'dead', 'deal', 'death', 'debate', 'decade', 'decide', 'decision', 'deep', 'defense', 'degree', 'Democrat', 'democratic', 'describe', 'design', 'despite', 'detail', 'determine', 'develop', 'development', 'die', 'difference', 'different', 'difficult', 'dinner', 'direction', 'director', 'discover', 'discuss', 'discussion', 'disease', 'do', 'doctor', 'dog', 'door', 'down', 'draw', 'dream', 'drive', 'drop', 'drug', 'during', 'each', 'early', 'east', 'easy', 'eat', 'economic', 'economy', 
        ]
        X2 = [
            'edge', 'education', 'effect', 'effort', 'eight', 'either', 'election', 'else', 'employee', 'end', 'energy', 'enjoy', 'enough', 'enter', 'entire', 'environment', 'environmental', 'especially', 'establish', 'even', 'evening', 'event', 'ever', 'every', 'everybody', 'everyone', 'everything', 'evidence', 'exactly', 'example', 'executive', 'exist', 'expect', 'experience', 'expert', 'explain', 'eye', 'face', 'fact', 'factor', 'fail', 'fall', 'family', 'far', 'fast', 'father', 'fear', 'federal', 'feel', 'feeling', 
            'few', 'field', 'fight', 'figure', 'fill', 'film', 'final', 'finally', 'financial', 'find', 'fine', 'finger', 'finish', 'fire', 'firm', 'first', 'fish', 'five', 'floor', 'fly', 'focus', 'follow', 'food', 'foot', 'for', 'force', 'foreign', 'forget', 'form', 'former', 'forward', 'four', 'free', 'friend', 'from', 'front', 'full', 'fund', 'future', 'game', 'garden', 'gas', 'general', 'generation', 'get', 'girl', 'give', 'glass', 'go', 'goal', 
            'good', 'government', 'great', 'green', 'ground', 'group', 'grow', 'growth', 'guess', 'gun', 'guy', 'hair', 'half', 'hand', 'hang', 'happen', 'happy', 'hard', 'have', 'he', 'head', 'health', 'hear', 'heart', 'heat', 'heavy', 'help', 'her', 'here', 'herself', 'high', 'him', 'himself', 'his', 'history', 'hit', 'hold', 'home', 'hope', 'hospital', 'hot', 'hotel', 'hour', 'house', 'how', 'however', 'huge', 'human', 'hundred', 'husband', 
            'I', 'idea', 'identify', 'if', 'image', 'imagine', 'impact', 'important', 'improve', 'in', 'include', 'including', 'increase', 'indeed', 'indicate', 'individual', 'industry', 'information', 'inside', 'instead', 'institution', 'interest', 'interesting', 'international', 'interview', 'into', 'investment', 'involve', 'issue', 'it', 'item', 'its', 'itself', 'job', 'join', 'just', 'keep', 'key', 'kid', 'kill', 'kind', 'kitchen', 'know', 'knowledge', 'land', 'language', 'large', 'last', 'late', 'later', 
            'laugh', 'law', 'lawyer', 'lay', 'lead', 'leader', 'learn', 'least', 'leave', 'left', 'leg', 'legal', 'less', 'let', 'letter', 'level', 'lie', 'life', 'light', 'like', 'likely', 'line', 'list', 'listen', 'little', 'live', 'local', 'long', 'look', 'lose', 'loss', 'lot', 'love', 'low', 'machine', 'magazine', 'main', 'maintain', 'major', 'majority', 'make', 'man', 'manage', 'management', 'manager', 'many', 'market', 'marriage', 'material', 'matter', 
        ]
        Y1 = [
            'may', 'maybe', 'me', 'mean', 'measure', 'media', 'medical', 'meet', 'meeting', 'member', 'memory', 'mention', 'message', 'method', 'middle', 'might', 'military', 'million', 'mind', 'minute', 'miss', 'mission', 'model', 'modern', 'moment', 'money', 'month', 'more', 'morning', 'most', 'mother', 'mouth', 'move', 'movement', 'movie', 'Mr', 'Mrs', 'much', 'music', 'must', 'my', 'myself', 'name', 'nation', 'national', 'natural', 'nature', 'near', 'nearly', 'necessary', 
            'need', 'network', 'never', 'new', 'news', 'newspaper', 'next', 'nice', 'night', 'no', 'none', 'nor', 'north', 'not', 'note', 'nothing', 'notice', 'now', 'nt', 'number', 'occur', 'of', 'off', 'offer', 'office', 'officer', 'official', 'often', 'oh', 'oil', 'ok', 'old', 'on', 'once', 'one', 'only', 'onto', 'open', 'operation', 'opportunity', 'option', 'or', 'order', 'organization', 'other', 'others', 'our', 'out', 'outside', 'over', 
            'own', 'owner', 'page', 'pain', 'painting', 'paper', 'parent', 'part', 'participant', 'particular', 'particularly', 'partner', 'party', 'pass', 'past', 'patient', 'pattern', 'pay', 'peace', 'people', 'per', 'perform', 'performance', 'perhaps', 'period', 'person', 'personal', 'phone', 'physical', 'pick', 'picture', 'piece', 'place', 'plan', 'plant', 'play', 'player', 'PM', 'point', 'police', 'policy', 'political', 'politics', 'poor', 'popular', 'population', 'position', 'positive', 'possible', 'power', 
            'practice', 'prepare', 'present', 'president', 'pressure', 'pretty', 'prevent', 'price', 'private', 'probably', 'problem', 'process', 'produce', 'product', 'production', 'professional', 'professor', 'program', 'project', 'property', 'protect', 'prove', 'provide', 'public', 'pull', 'purpose', 'push', 'put', 'quality', 'question', 'quickly', 'quite', 'race', 'radio', 'raise', 'range', 'rate', 'rather', 'reach', 'read', 'ready', 'real', 'reality', 'realize', 'really', 'reason', 'receive', 'recent', 'recently', 'recognize', 
            'record', 'red', 'reduce', 'reflect', 'region', 'relate', 'relationship', 'religious', 'remain', 'remember', 'remove', 'report', 'represent', 'Republican', 'require', 'research', 'resource', 'respond', 'response', 'responsibility', 'rest', 'result', 'return', 'reveal', 'rich', 'right', 'rise', 'risk', 'road', 'rock', 'role', 'room', 'rule', 'run', 'safe', 'same', 'save', 'say', 'scene', 'school', 'science', 'scientist', 'score', 'sea', 'season', 'seat', 'second', 'section', 'security', 'see', 
        ]
        Y2 = [
            'seek', 'seem', 'sell', 'send', 'senior', 'sense', 'series', 'serious', 'serve', 'service', 'set', 'seven', 'several', 'sex', 'sexual', 'shake', 'share', 'she', 'shoot', 'short', 'shot', 'should', 'shoulder', 'show', 'side', 'sign', 'significant', 'similar', 'simple', 'simply', 'since', 'sing', 'single', 'sister', 'sit', 'site', 'situation', 'six', 'size', 'skill', 'skin', 'small', 'smile', 'so', 'social', 'society', 'soldier', 'some', 'somebody', 'someone', 
            'something', 'sometimes', 'son', 'song', 'soon', 'sort', 'sound', 'source', 'south', 'southern', 'space', 'speak', 'special', 'specific', 'speech', 'spend', 'sport', 'spring', 'staff', 'stage', 'stand', 'standard', 'star', 'start', 'state', 'statement', 'station', 'stay', 'step', 'still', 'stock', 'stop', 'store', 'story', 'strategy', 'street', 'strong', 'structure', 'student', 'study', 'stuff', 'style', 'subject', 'success', 'successful', 'such', 'suddenly', 'suffer', 'suggest', 'summer', 
            'support', 'sure', 'surface', 'system', 'table', 'take', 'talk', 'task', 'tax', 'teach', 'teacher', 'team', 'technology', 'television', 'tell', 'ten', 'tend', 'term', 'test', 'than', 'thank', 'that', 'the', 'their', 'them', 'themselves', 'then', 'theory', 'there', 'these', 'they', 'thing', 'think', 'third', 'this', 'those', 'though', 'thought', 'thousand', 'threat', 'three', 'through', 'throughout', 'throw', 'thus', 'time', 'to', 'today', 'together', 'tonight', 
            'too', 'top', 'total', 'tough', 'toward', 'town', 'trade', 'traditional', 'training', 'travel', 'treat', 'treatment', 'tree', 'trial', 'trip', 'trouble', 'true', 'truth', 'try', 'turn', 'TV', 'two', 'type', 'under', 'understand', 'unit', 'until', 'up', 'upon', 'us', 'use', 'usually', 'value', 'various', 'very', 'victim', 'view', 'violence', 'visit', 'voice', 'vote', 'wait', 'walk', 'wall', 'want', 'war', 'watch', 'water', 'way', 'we', 
            'weapon', 'wear', 'week', 'weight', 'well', 'west', 'western', 'what', 'whatever', 'when', 'where', 'whether', 'which', 'while', 'white', 'who', 'whole', 'whom', 'whose', 'why', 'wide', 'wife', 'will', 'win', 'wind', 'window', 'wish', 'with', 'within', 'without', 'woman', 'wonder', 'word', 'work', 'worker', 'world', 'worry', 'would', 'write', 'writer', 'wrong', 'yard', 'yeah', 'year', 'yes', 'yet', 'you', 'young', 'your', 'yourself', 
        ]
        """
        X1 = [
            'a', 'ability', 'able', 'about', 'above', 'accept', 'according', 'account', 'across', 'act', 'action', 'activity', 'actually',
            'add', 'address', 'administration', 'admit', 'adult', 'affect', 'after', 'again', 'against', 'age', 'agency', 'agent', 'ago',
            'agree', 'agreement', 'ahead', 'air', 'all', 'allow', 'almost', 'alone', 'along', 'already', 'also', 'although', 'always',
            'American', 'among', 'amount', 'analysis', 'and', 'animal', 'another', 'answer', 'any', 'anyone', 'anything',
        ]
        X2 = [
            'appear', 'apply', 'approach', 'area', 'argue', 'arm', 'around', 'arrive', 'art', 'article', 'artist', 'as', 'ask', 'assume',
            'at', 'attack', 'attention', 'attorney', 'audience', 'author', 'authority', 'available', 'avoid', 'away', 'baby', 'back',
            'bad', 'bag', 'ball', 'bank', 'bar', 'base', 'be', 'beat', 'beautiful', 'because', 'become', 'bed', 'before', 'begin',
            'behavior', 'behind', 'believe', 'benefit', 'best', 'better', 'between', 'beyond', 'big', 'bill',
        ]
        Y1 = [
            'billion', 'bit', 'black', 'blood', 'blue', 'board', 'body', 'book', 'born', 'both', 'box', 'boy', 'break', 'bring', 'brother',
            'budget', 'build', 'building', 'business', 'but', 'buy', 'by', 'call', 'camera', 'campaign', 'can', 'cancer', 'candidate',
            'capital', 'car', 'card', 'care', 'career', 'carry', 'case', 'catch', 'cause', 'cell', 'center', 'central', 'century',
            'certain', 'certainly', 'chair', 'challenge', 'chance', 'change', 'character', 'charge', 'check',
        ]
        Y2 = [
            'child', 'choice', 'choose', 'church', 'citizen', 'city', 'civil', 'claim', 'class', 'clear', 'clearly', 'close', 'coach', 'cold',
            'collection', 'college', 'color', 'come', 'commercial', 'common', 'community', 'company', 'compare', 'computer', 'concern',
            'condition', 'conference', 'Congress', 'consider', 'consumer', 'contain', 'continue', 'control', 'cost', 'could', 'country',
            'couple', 'course', 'court', 'cover', 'create', 'crime', 'cultural', 'culture', 'cup', 'current', 'customer', 'cut', 'dark', 'data',
        ]
        """

        # Put list info.
        dataset = dict()
        dataset['list'] = {
            'X1': X1, 'X2': X2, 'Y1': Y1, 'Y2': Y2,
        }

        data = dict()

        for x in X1 + X2:
            y1_combination = []
            for y1 in Y1:
                y1_combination.append([x, y1])
                random.shuffle(y1_combination)
            y2_combination = []
            for y2 in Y2:
                y2_combination.append([x, y2])
                random.shuffle(y2_combination)
            data[x] = {'y1': y1_combination, 'y2': y2_combination}

        dataset['data'] = data

        dictionary = {'<PAD>': 0}
        for word in X1 + X2 + Y1 + Y2:
            dictionary[word] = len(dictionary)
        dataset['dictionary'] = dictionary

        with open(filename, 'wb') as f:
            pickle.dump(dataset, f)

    return dataset
