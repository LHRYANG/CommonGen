from torch.utils.data import Dataset
import random
from itertools import groupby
from nltk.tokenize import word_tokenize

class UnorderPTData(Dataset):
    # used for domain adaptation
    def __init__(self,args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        self.p = args.p
        with open("data/train_trg.txt") as f:
            self.trgs = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.trgs)

    def __getitem__(self, index):
        targets = self.trgs[index]
        inputs = self.trgs[index]
        inputs = word_tokenize(inputs)
        mask_index = random.sample(range(0,len(inputs)),int(self.p*len(inputs)))
        mask_index = sorted(mask_index)
        for i in mask_index:
            inputs[i] = "<mask>"

        new_inputs = [key for key, _group in groupby(inputs)]
        temp_src = []
        for a in new_inputs:
            if a != "<mask>":
                temp_src.append(a)
        random.shuffle(temp_src)
        count = 0
        for iii, a in enumerate(new_inputs):
            if a!="<mask>":
                new_inputs[iii] = temp_src[count]
                count+=1
        model_inputs = self.tokenizer(' '.join(new_inputs), max_length=64, padding="max_length",truncation=True)
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(targets, max_length=96, padding="max_length",truncation=True)

        labels["input_ids"] =[(l if l != self.tokenizer.pad_token_id else -100) for l in labels["input_ids"]]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs




class UnOrder_Mask(Dataset):
    def __init__(self, args, tokenizer, part="train", mask =False):
        self.args = args
        self.tokenizer = tokenizer
        self.mask = mask
        with open("data/"+part+"_src.txt") as f1, open("data/"+part+"_trg.txt") as f2:
            if self.mask:
                self.srcs = ['<mask> '+' <mask> '.join(line.strip().split(' ')) for line in f1.readlines()]
            else:
                self.srcs = [line.strip() for line in f1.readlines()]
            self.trgs = [line.strip() for line in f2.readlines()]

    def __len__(self):
        return len(self.srcs)

    def __getitem__(self, index):
        inputs = self.srcs[index]
        targets = self.trgs[index]

        model_inputs = self.tokenizer(inputs, max_length=self.args.max_src_len, padding="max_length",
                  truncation=True)
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(targets, max_length=self.args.max_trg_len, padding="max_length",
                               truncation=True)
            #print(labels)
        labels["input_ids"] =[(l if l != self.tokenizer.pad_token_id else -100) for l in labels["input_ids"]]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
