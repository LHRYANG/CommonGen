from datasets import load_dataset,load_metric
from torch.utils.data import DataLoader
from cg_dataset import UnOrder_Mask
from transformers import BartForConditionalGeneration, BartTokenizer
import warnings
from torch import cuda
import argparse
from rouge_score import rouge_scorer, scoring
import numpy as np
import torch
from transformers import (
    DataCollatorForSeq2Seq,
)

device = 'cuda' if cuda.is_available() else 'cpu'
warnings.simplefilter(action='ignore', category=FutureWarning)



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--tr_batch_size', type=int, default=2)
    parser.add_argument('--te_batch_size', type=int, default=64)

    parser.add_argument('--max_src_len', type=int, default=48)
    parser.add_argument('--max_trg_len', type=int, default=128)
    parser.add_argument('--n_beam', type=int, default=5)
    parser.add_argument('--n_model', type=int, default=5)
    parser.add_argument('--output_dir',type=str,default='save_model/normal_train')
    parser.add_argument('--output_name', type=str, default='model.bin')
    parser.add_argument("--pad_to_max_length",action="store_true")
    parser.add_argument('--mask', type=str, default="True")

    args = parser.parse_args()
    return args


def train(args):

    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large").to(device)
    model.load_state_dict(torch.load(args.output_dir+"/model.bin"))
    tok = BartTokenizer.from_pretrained("facebook/bart-large")

    if args.mask == "True":
        test_dataset = UnOrder_Mask(args, tok, 'test', mask=True)
    else:
        test_dataset = UnOrder_Mask(args, tok, 'test')

    data_collator = DataCollatorForSeq2Seq(
        tok,
        model=model,
        label_pad_token_id=-100,
    )
    gen_kwargs = {
        "max_length": args.max_trg_len,
        "num_beams": args.n_beam,
    }

    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=args.te_batch_size, shuffle=False)
    rouge = load_metric('rouge')
    with torch.no_grad():
        with open(args.output_dir+"/test_pred.txt",'a') as f:
            first_line = True
            for step, batch in enumerate(test_dataloader):
                batch = {k: v.to(device) for k, v in batch.items()}
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
        print(result)


if __name__ == "__main__":
    args = parse_args()
    print(args)
    train(args)

