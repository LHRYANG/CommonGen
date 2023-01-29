from datasets import load_dataset,load_metric
from torch.utils.data import DataLoader
from cg_dataset import UnOrder_Mask, UnorderPTData
from transformers import BartForConditionalGeneration, BartTokenizer
import warnings
from torch import cuda
import argparse
from rouge_score import rouge_scorer, scoring
import numpy as np
import os
import torch
from os import path
from transformers import (
    DataCollatorForSeq2Seq,
    AdamW,
)
from transformers.optimization import  get_linear_schedule_with_warmup

device = 'cuda' if cuda.is_available() else 'cpu'
warnings.simplefilter(action='ignore', category=FutureWarning)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tr_batch_size', type=int, default=32)
    parser.add_argument('--te_batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument("--num_warmup_steps", type=int, default=400)
    parser.add_argument("--p", type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--pt_epochs', type=int, default=8)
    parser.add_argument('--max_src_len', type=int, default=48)
    parser.add_argument('--max_trg_len', type=int, default=128)
    parser.add_argument('--n_beam', type=int, default=5)
    parser.add_argument('--model_name_or_path', type=str, default="facebook/bart-large")
    parser.add_argument('--pt_model_path', type=str, default="models/ptmodel.bin")
    parser.add_argument('--output_dir',type=str,default='models')
    parser.add_argument('--output_name', type=str, default='model.bin')
    parser.add_argument("--pad_to_max_length",action="store_true")
    parser.add_argument('--desc', type=str, default='nothing')
    parser.add_argument('--mask', type=str, default="True")
    args = parser.parse_args()
    return args


def train(args):

    model = BartForConditionalGeneration.from_pretrained(args.model_name_or_path).to(device)
    tok = BartTokenizer.from_pretrained(args.model_name_or_path)
    if args.pt_epochs ==0 and args.p == 0.5:
        #load pretrained bart
        model.load_state_dict(torch.load(args.pt_model_path))

    ptdata = UnorderPTData(args,tok)
    if args.mask == "True":
        train_dataset = UnOrder_Mask(args, tok, 'train', mask=True)
        valid_dataset = UnOrder_Mask(args, tok, 'valid', mask=True)
    else:
        train_dataset = UnOrder_Mask(args, tok, 'train')
        valid_dataset = UnOrder_Mask(args, tok, 'valid')

    data_collator = DataCollatorForSeq2Seq(
        tok,
        model=model,
        label_pad_token_id=-100,
    )

    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.tr_batch_size)
    valid_dataloader = DataLoader(valid_dataset, collate_fn=data_collator, batch_size=args.te_batch_size,shuffle=False)
    pt_dataloader = DataLoader(ptdata, collate_fn=data_collator, batch_size=args.tr_batch_size, shuffle=True)

    if not path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    pt_optimizer = AdamW(model.parameters(),lr=1e-7,eps=1e-8)
    for epoch in range(args.pt_epochs):
        tr_loss = 0
        nb_tr_steps = 0
        model.train()
        for step, batch in enumerate(pt_dataloader):
            # prepare data
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            tr_loss += loss.item()
            nb_tr_steps += 1
            loss.backward()
            pt_optimizer.step()
            pt_optimizer.zero_grad()

        epoch_loss = tr_loss / nb_tr_steps
        print(f"Pretraining Loss Epoch: {epoch_loss}")
    if args.pt_epochs!=0:
        torch.save(model.state_dict(), args.output_dir + "/ptmodel.bin")

    total_steps = int(args.epochs*len(train_dataset)/args.tr_batch_size)
    optimizer = AdamW(model.parameters(),lr=args.lr,eps=1e-8)
    if args.num_warmup_steps!=0:
        lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.num_warmup_steps, num_training_steps=total_steps)
        print("scheduler is used")

    rouge = load_metric('rouge')
    best_rouge = 0


    for epoch in range(args.epochs):
        tr_loss = 0
        nb_tr_steps = 0
        model.train()
        for step, batch in enumerate(train_dataloader):
            #prepare data
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            tr_loss += loss.item()
            nb_tr_steps += 1
            loss.backward()
            optimizer.step()

            if args.num_warmup_steps != 0:
                lr_scheduler.step()
            optimizer.zero_grad()

        epoch_loss = tr_loss/nb_tr_steps

        print(f"Training Loss Epoch: {epoch_loss}")
        model.eval()
        gen_kwargs = {
            "max_length": args.max_trg_len,
            "num_beams": args.n_beam,
        }

        with torch.no_grad():
            with open(args.output_dir+"/valid_pred"+str(epoch)+".txt",'a') as f:
                first_line = True
                val_loss = 0
                nb_val_steps = 0
                for step, batch in enumerate(valid_dataloader):
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(**batch)
                    loss = outputs.loss
                    val_loss += loss.item()
                    nb_val_steps +=1
                    generated_tokens = model.generate(batch["input_ids"],attention_mask=batch["attention_mask"],**gen_kwargs)

                    labels = batch["labels"]
                    generated_tokens = generated_tokens.cpu().numpy()
                    labels = labels.cpu().numpy()
                    labels = np.where(labels != -100, labels, tok.pad_token_id)

                    decoded_preds = tok.batch_decode(generated_tokens, skip_special_tokens=True,clean_up_tokenization_spaces=True)
                    decoded_labels = tok.batch_decode(labels, skip_special_tokens=True,clean_up_tokenization_spaces=True)
                    rouge.add_batch(predictions=decoded_preds, references=decoded_labels)
                    for line in decoded_preds:
                        if first_line:
                            f.write(line.strip())
                            first_line = False
                        else:
                            f.write('\n')
                            f.write(line)

        result = rouge.compute(use_stemmer=True)
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        result = {k: round(v, 4) for k, v in result.items()}

        if result['rougeL']>best_rouge:
            best_rouge = result['rougeL']
            torch.save(model.state_dict(), args.output_dir + "/model.bin")
            print("epoch" + str(epoch))
            print(result)
            print("val_loss", val_loss / nb_val_steps)
if __name__ == "__main__":
    args = parse_args()
    print(args)
    train(args)

