CUDA_VISIBLE_DEVICES=2 python train.py \
      --tr_batch_size 32 \
      --te_batch_size 32 \
      --epochs 10 \
      --max_src_len 48 \
      --max_trg_len 128 \
      --n_beam 5 \
      --output_dir models \
      --lr 2e-5 \
      --num_warmup_steps 10000 \
      --mask True \
      --p 0.5 \
      --pt_epochs 10

