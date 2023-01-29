# CommenGen
Pytorch Implementation of EACL paper(findings) **Bridging the Gap between Pre-Training and Fine-Tuning  for Commonsense Generation**

## How to train
 `./train.sh`
 
 We also provide our trained model [here](https://drive.google.com/drive/folders/1SHHNpg4bsFYOoPeQ-sB46XOjfV-_Siba?usp=sharing)
 
## How to test

`python test.py --output_dir models/`

models/ is the directory that contains the trained model **model.bin**.

## How to evaluate 
We provide our predicted test file **test_pred.txt** under data/

Download and unzip the evaluation files [here](https://drive.google.com/drive/folders/1SHHNpg4bsFYOoPeQ-sB46XOjfV-_Siba?usp=sharing)

**1. obtain coverage score**

`cd evaluation/PivotScore`

`python evaluate.py --pred your_pred_file --ref ../../data/test_trg.txt --cs ../../data/test_src.txt`

**2. obtain other score**

`python convert_to_json.py --src data/test_src.txt --trg data/test_trg.txt --pred your_pred_file`

`cd evaluation/CaptionMetrics/` 

`python main.py --trg ../../temp_trg.json --pred ../../temp_pred.json`
