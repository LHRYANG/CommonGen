import spacy
import argparse
import codecs

nlp = spacy.load("en_core_web_sm")
#nlp.pipeline = [('tagger', nlp.tagger)]

import json


parser = argparse.ArgumentParser()
parser.add_argument('--src', default="", type=str)
parser.add_argument('--trg', default="", type=str)
parser.add_argument('--pred', default="", type=str)
args = parser.parse_args()

def tokenize(dict):
    for key in dict:
        new_sentence_list = []
        for sentence in dict[key]:
            a = ''
            for token in nlp(str(sentence)):
                a += token.text
                a += ' '
            new_sentence_list.append(a.rstrip())
        dict[key] = new_sentence_list
        #print(key)
    return dict

def modify(args):
    gts = {}
    res = {}

    with codecs.open(args.src, encoding='utf-8') as f:
        key_lines = f.readlines()
        # key_lines = [line.decode('utf-8') for line in f.readlines()]
    with codecs.open(args.trg, encoding='utf-8') as f:
        gts_lines = f.readlines()
        # gts_lines = [line.decode('utf-8') for line in f.readlines()]
    with codecs.open(args.pred, encoding='utf-8') as f:
        res_lines = f.readlines()
        # res_lines = [line.decode('utf-8') for line in f.readlines()]

    for key_line, gts_line, res_line in zip(key_lines, gts_lines, res_lines):
        key = '#'.join(key_line.rstrip('\n').split(' ')).strip()
        #print(key)
        if key not in gts:
            gts[key] = []
            gts[key].append(gts_line.rstrip('\n'))
            res[key] = []
            res[key].append(res_line.rstrip('\n'))
        else:
            gts[key].append(gts_line.rstrip('\n'))

    return gts, res

if __name__ == "__main__":
    #args = parse_args()
    gts,res = modify(args)
    print('tokenization...')
    gts = tokenize(gts)
    res = tokenize(res)
    with open('temp_trg.json', 'w') as fp:
        json.dump(gts, fp)
    with open('temp_pred.json', 'w') as fp:
        json.dump(res, fp)

    # print(gts)
    # print(res)

