import pickle
from shutil import rmtree
from os import listdir, mkdir
from os.path import join, isfile, basename
import requests, io, tarfile

import os
import sys
import re
import random
import numpy as np
from itertools import chain, zip_longest
from distutils.dir_util import copy_tree

import torch


def create_dictionary(files_list):

    lexicons_dict = {'<PAD>': 0, '.': 1, '?': 2, '-': 3}

    for indx, filename in enumerate(files_list):
        with open(filename, 'r') as fobj:
            for line in fobj:

                line = line.replace('.', ' .')
                line = line.replace('?', ' ?')

                for word in line.split():
                    if not word.lower() in lexicons_dict and (word.isalpha() or (',' in word)):
                        lexicons_dict[word.lower()] = len(lexicons_dict)
    return lexicons_dict


def encode_data(files_list, lexicons_dictionary, length_limit=200):

    files = {}

    for indx, filename in enumerate(files_list):

        test_flag = ('test.txt' in filename)
        files[filename] = []
        story_inputs = []
        story_outputs = []
        questions = {}
        stack_pointer = 0

        with open(filename, 'r') as fobj:
            for line in fobj:

                line = line.replace('.', ' .')
                line = line.replace('?', ' ?')

                line_number, line = line.strip().split(' ', 1)
                if line_number == '1':

                    if test_flag and not story_inputs == []:
                        files[filename].append({
                            'input': story_inputs,
                            'output': story_outputs,
                        })

                    story_inputs = []
                    story_outputs = []
                    questions = {}
                    stack_pointer = 0

                if '?' in line:
                    line, answer, supporting_fact = line.split('\t')
                    supporting_fact = supporting_fact.split()
                    questions[int(line_number)] = int(min(supporting_fact))

                    question = []
                    for i, word in enumerate(line.split()):
                        if word.isalpha() or word == '?' or (',' in word):
                            question.append(lexicons_dictionary[word.lower()])

                    question.append(lexicons_dictionary['-'])
                    answer = lexicons_dictionary[answer.lower()]

                    story_inputs.append(question)
                    story_outputs.append([lexicons_dictionary['<PAD>']] * (len(question) - 1) + [answer])

                    if not test_flag:
                        remove_candidate = []
                        for tid, sup in questions.items():
                            if tid < (stack_pointer + 1):
                                story_outputs[tid - 1][-1] = lexicons_dictionary['<PAD>']
                                remove_candidate.append(tid)
                        for tid in remove_candidate:
                            del questions[tid]

                        files[filename].append({
                            'input': story_inputs[stack_pointer:],
                            'output': story_outputs[stack_pointer:],
                        })

                else:
                    sentence = []
                    for i, word in enumerate(line.split()):
                        if word.isalpha() or word == '?' or word == '.':
                            sentence.append(lexicons_dictionary[word.lower()])
                    story_inputs.append(sentence)
                    story_outputs.append([lexicons_dictionary['<PAD>']] * len(sentence))

                    story_length = sum([len(s) for s in story_inputs[stack_pointer:]])
                    if (story_length > length_limit) and not test_flag:
                        stack_pointer += 1

            if test_flag and not story_inputs == []:
                files[filename].append({
                    'input': story_inputs,
                    'output': story_outputs,
                })

    return files


def bAbI_preprocessing(data_dir):

    download_url = 'http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz'
    download_dir = 'tasks_1-20_v1-2'

    r = requests.get(download_url)
    t = tarfile.open(fileobj=io.BytesIO(r.content), mode='r|gz')
    t.extractall('.')

    files_list = []
    task_dir = join(download_dir, 'en-valid-10k')
    for entryname in listdir(task_dir):
        entry_path = join(task_dir, entryname)
        if isfile(entry_path):
            files_list.append(entry_path)

    lexicon_dictionary = create_dictionary(files_list)
    encoded_files = encode_data(files_list, lexicon_dictionary)

    rmtree(download_dir)

    train_data_dir = join(data_dir, 'train')
    valid_data_dir = join(data_dir, 'valid')
    test_data_dir = join(data_dir, 'test')

    mkdir(data_dir)
    mkdir(train_data_dir)
    mkdir(valid_data_dir)
    mkdir(test_data_dir)

    pickle.dump(lexicon_dictionary, open(join(data_dir, 'lexicon-dict.pkl'), 'wb'))

    train_data = []
    valid_data = []

    for filename in encoded_files:
        if filename.endswith("test.txt"):
            pickle.dump(encoded_files[filename], open(join(test_data_dir, basename(filename) + '.pkl'), 'wb'))
        elif filename.endswith("valid.txt"):
            valid_data.extend(encoded_files[filename])
        elif filename.endswith("train.txt"):
            train_data.extend(encoded_files[filename])

    pickle.dump(train_data, open(join(train_data_dir, 'train.pkl'), 'wb'))
    pickle.dump(valid_data, open(join(valid_data_dir, 'valid.pkl'), 'wb'))

    return None


def load(path):
    return pickle.load(open(path, 'rb'))


class BAbIBatchGenerator(object):
    def __init__(self, batch_size,
                 shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle

        data_dir = os.path.join('task', 'bAbIQA_data')
        if not os.path.exists(data_dir):
            bAbI_preprocessing(data_dir)

        self.dataset = load(os.path.join(data_dir, 'train', 'train.pkl'))
        self.lexicon_dict = load(os.path.join(data_dir, 'lexicon-dict.pkl'))

        random.shuffle(self.dataset)

        self.count = 0

        self.limit = int(len(self.dataset)/self.batch_size)

    def increase_count(self):
        self.count += 1
        if self.count >= self.limit:
            if self.shuffle:
                random.shuffle(self.dataset)
            self.count = 0

    def __iter__(self):
        return self

    def __next__(self):
        samples = self.dataset[self.count * self.batch_size: (self.count + 1) * self.batch_size]

        input_sequence = []
        output_sequence = []
        for sample in samples:
            input_sequence.append(sum(sample['input'], []))
            output_sequence.append(sum(sample['output'], []))

        seq_len = list(map(len, input_sequence))

        input_sequence = np.array(list(zip_longest(*input_sequence, fillvalue=0)), dtype=np.int32)
        input_sequence = np.transpose(input_sequence, (1, 0))

        output_sequence = np.array(list(zip_longest(*output_sequence, fillvalue=0)), dtype=np.int32)
        output_sequence = np.transpose(output_sequence, (1, 0))

        target_mask = (output_sequence != self.lexicon_dict['<PAD>'])

        self.increase_count()

        return (
            torch.LongTensor(input_sequence),
            torch.BoolTensor(target_mask),
            torch.LongTensor(output_sequence),
            torch.LongTensor(seq_len),
        )

    @property
    def input_size(self):
        return len(self.lexicon_dict)

    @property
    def output_size(self):
        return len(self.lexicon_dict)

    @property
    def data_size(self):
        return len(self.dataset)


class BAbITestBatchGenerator(object):
    def __init__(self):

        data_dir = os.path.join('task', 'bAbIQA_data')
        if not os.path.exists(data_dir):
            bAbI_preprocessing(data_dir)

        self.batch_size = 0
        self.test_data_dir = os.path.join(data_dir, 'test')
        self.dataset = None
        self.lexicon_dict = load(os.path.join(data_dir, 'lexicon-dict.pkl'))
        self.target_code = self.lexicon_dict['-']
        self.question_code = self.lexicon_dict['?']

        self.count = None

    def feed_data(self, task_dir):
        self.count = 0

        cur_task_dir = os.path.join(self.test_data_dir, task_dir)
        task_regexp = r'qa([0-9]{1,2})_test.txt.pkl'
        task_filename = os.path.basename(task_dir)
        task_match_obj = re.match(task_regexp, task_filename)
        task_number = task_match_obj.group(1)

        self.dataset = load(cur_task_dir)

        return task_number, len(self.dataset)

    def feed_batch_size(self, batch_size):
        self.batch_size = batch_size

    def __iter__(self):
        return self

    def __next__(self):
        samples = self.dataset[self.count: self.count + self.batch_size]
        self.count += self.batch_size

        input_sequence = []
        output_sequence = []
        for sample in samples:
            input_sequence.append(sum(sample['input'], []))
            output_sequence.append(sum(sample['output'], []))

        seq_len = list(map(len, input_sequence))

        input_sequence = np.array(list(zip_longest(*input_sequence, fillvalue=0)), dtype=np.int32)
        input_sequence = np.transpose(input_sequence, (1, 0))

        output_sequence = np.array(list(zip_longest(*output_sequence, fillvalue=0)), dtype=np.int32)
        output_sequence = np.transpose(output_sequence, (1, 0))

        target_mask = (output_sequence != self.lexicon_dict['<PAD>'])

        return (
            torch.LongTensor(input_sequence),
            torch.BoolTensor(target_mask),
            torch.LongTensor(output_sequence),
            torch.LongTensor(seq_len),
        )

    @property
    def input_size(self):
        return len(self.lexicon_dict)

    @property
    def output_size(self):
        return len(self.lexicon_dict)

    @property
    def data_size(self):
        return len(self.dataset)


class BAbIValidBatchGenerator(object):
    def __init__(self, batch_size=100):
        self.batch_size = batch_size

        data_dir = os.path.join('task', 'bAbIQA_data')
        if not os.path.exists(data_dir):
            bAbI_preprocessing(data_dir)

        self.dataset = load(os.path.join(data_dir, 'valid', 'valid.pkl'))
        self.lexicon_dict = load(os.path.join(data_dir, 'lexicon-dict.pkl'))
        self.target_code = self.lexicon_dict['-']

        self.count = 0

        self.limit = int(len(self.dataset)/self.batch_size) + 1

    def reset(self):
        self.count = 0

    def __iter__(self):
        return self

    def __next__(self):
        samples = self.dataset[self.count * self.batch_size: (self.count + 1) * self.batch_size]

        input_sequence = []
        output_sequence = []
        for sample in samples:
            input_sequence.append(sum(sample['input'], []))
            output_sequence.append(sum(sample['output'], []))

        seq_len = list(map(len, input_sequence))

        input_sequence = np.array(list(zip_longest(*input_sequence, fillvalue=0)), dtype=np.int32)
        input_sequence = np.transpose(input_sequence, (1, 0))

        output_sequence = np.array(list(zip_longest(*output_sequence, fillvalue=0)), dtype=np.int32)
        output_sequence = np.transpose(output_sequence, (1, 0))

        target_mask = (output_sequence != self.lexicon_dict['<PAD>'])

        self.count += 1

        return (
            torch.LongTensor(input_sequence),
            torch.BoolTensor(target_mask),
            torch.LongTensor(output_sequence),
            torch.LongTensor(seq_len),
        )

    @property
    def input_size(self):
        return len(self.lexicon_dict)

    @property
    def output_size(self):
        return len(self.lexicon_dict)

    @property
    def data_size(self):
        return len(self.dataset)


class BAbIDiffTestBatchGenerator(object):
    def __init__(self):

        data_dir = os.path.join('task', 'bAbIQA_data')
        if not os.path.exists(data_dir):
            bAbI_preprocessing(data_dir)

        self.batch_size = 0
        self.test_data_dir = os.path.join(data_dir, 'test_diff')
        self.dataset = None
        self.lexicon_dict = load(os.path.join(data_dir, 'lexicon-dict.pkl'))
        self.target_code = self.lexicon_dict['-']
        self.question_code = self.lexicon_dict['?']

        self.generate()

        self.count = None

    def generate(self):

        if not os.path.exists(self.test_data_dir):
            original_test_data_dir = os.path.join('task', 'bAbIQA_data', 'test')
            copy_tree(original_test_data_dir, self.test_data_dir)

            name_mapping = {
                self.lexicon_dict['daniel']: self.lexicon_dict['bill'],
                self.lexicon_dict['john']: self.lexicon_dict['fred'],
                self.lexicon_dict['sandra']: self.lexicon_dict['julie'],
            }
            replacer = name_mapping.get
            for task_id in [1, 2, 3, 6, 7, 8, 9, 11, 12, 13]:
                file_name = os.path.join(self.test_data_dir, 'qa{0}_test.txt.pkl'.format(task_id))
                with open(file_name, 'rb') as f:
                    d = pickle.load(f)
                for i in range(len(d)):
                    d[i]['input'] = [[replacer(n, n) for n in a] for a in d[i]['input']]
                    d[i]['output'] = [[replacer(n, n) for n in a] for a in d[i]['output']]
                with open(file_name, 'wb') as f:
                    pickle.dump(d, f)

            reverse_name_mapping = dict([[v, k] for k, v in name_mapping.items()])
            replacer = reverse_name_mapping.get
            for task_id in [5, 10, 14]:
                file_name = os.path.join(self.test_data_dir, 'qa{0}_test.txt.pkl'.format(task_id))
                with open(file_name, 'rb') as f:
                    d = pickle.load(f)
                for i in range(len(d)):
                    d[i]['input'] = [[replacer(n, n) for n in a] for a in d[i]['input']]
                    d[i]['output'] = [[replacer(n, n) for n in a] for a in d[i]['output']]
                with open(file_name, 'wb') as f:
                    pickle.dump(d, f)

    def feed_data(self, task_dir):
        self.count = 0

        cur_task_dir = os.path.join(self.test_data_dir, task_dir)
        task_regexp = r'qa([0-9]{1,2})_test.txt.pkl'
        task_filename = os.path.basename(task_dir)
        task_match_obj = re.match(task_regexp, task_filename)
        task_number = task_match_obj.group(1)

        self.dataset = load(cur_task_dir)

        return task_number, len(self.dataset)

    def feed_batch_size(self, batch_size):
        self.batch_size = batch_size

    def __iter__(self):
        return self

    def __next__(self):
        samples = self.dataset[self.count: self.count + self.batch_size]
        self.count += self.batch_size

        input_sequence = []
        output_sequence = []
        for sample in samples:
            input_sequence.append(sum(sample['input'], []))
            output_sequence.append(sum(sample['output'], []))

        seq_len = list(map(len, input_sequence))

        input_sequence = np.array(list(zip_longest(*input_sequence, fillvalue=0)), dtype=np.int32)
        input_sequence = np.transpose(input_sequence, (1, 0))

        output_sequence = np.array(list(zip_longest(*output_sequence, fillvalue=0)), dtype=np.int32)
        output_sequence = np.transpose(output_sequence, (1, 0))

        target_mask = (output_sequence != self.lexicon_dict['<PAD>'])

        return (
            torch.LongTensor(input_sequence),
            torch.BoolTensor(target_mask),
            torch.LongTensor(output_sequence),
            torch.LongTensor(seq_len),
        )

    @property
    def input_size(self):
        return len(self.lexicon_dict)

    @property
    def output_size(self):
        return len(self.lexicon_dict)

    @property
    def data_size(self):
        return len(self.dataset)


if __name__ == '__main__':
    data_dir = os.path.join('bAbIQA_data')
    if not os.path.exists(data_dir):
        bAbI_preprocessing(data_dir)
