import pickle
import os
import shutil


from_path = 'babi/data/en-valid-10k-pik'
to_path = 'babi/data/revised-pik'

if not os.path.exists(to_path):
    shutil.copytree(from_path, to_path)

    with open(os.path.join(to_path, 'qa20_test.pik'), 'rb') as f:
        d = pickle.load(f)

    lexi = d[-1]

    mapping = {
            lexi['daniel']: lexi['bill'],
            lexi['john']: lexi['fred'],
            lexi['sandra']: lexi['julie']}

    reverse = dict([[v, k] for k, v in mapping.items()])

    for i in [1, 2, 3, 6, 7, 8, 9, 11, 12, 13]:
        with open(os.path.join(to_path, 'qa{0}_test.pik'.format(i)), 'rb') as f:
            d = pickle.load(f)

        story, story_length, query, answer, word2id = d

        for key in mapping.keys():
            target = key
            replace = mapping[key]

            story[story == target] = replace
            query[query == target] = replace
            answer[answer == target] = replace

        data = (story, story_length, query, answer, word2id)

        with open(os.path.join(to_path, 'qa{0}_test.pik'.format(i)), 'wb') as f:
            pickle.dump(data, f)

    for i in [5, 10, 14]:
        with open(os.path.join(to_path, 'qa{0}_test.pik'.format(i)), 'rb') as f:
            d = pickle.load(f)

        story, story_length, query, answer, word2id = d

        for key in reverse.keys():
            target = key
            replace = reverse[key]

            story[story == target] = replace
            query[query == target] = replace
            answer[answer == target] = replace

        data = (story, story_length, query, answer, word2id)

        with open(os.path.join(to_path, 'qa{0}_test.pik'.format(i)), 'wb') as f:
            pickle.dump(data, f)
