#!/bin/sh

training(){
     python run_expt.py -s confounder -d CelebA -t Blond_Hair -c Male --lr 0.0001 --batch_size 32 --weight_decay 0.00005 --model resnet50 --n_epochs 50 --gamma 0.1 --generalization_adjustment 2 --train_type trades --log_dir_text log_.txt --test_type pgd --random_init 0.0001 --epsilon 0.00196 --attack_iters 10 --lamda 1 --tau 0.01 --alpha_ 0.01 --gpu 0  --name_index robust  --log_dir 'logs'  --l2_norm 0.00 --lr_tau 0.0  --beta 6.0   --clmax 1.0  --clmin 0.0    --robust --trades_new     --train_grad  --limit_nat --limit_adv --reweight_groups    
}
(
    
   training
);
