#!/usr/bin/env bash

#GPUs
gpus=0

#Set paths
checkpoint_root=/home/skymap/data/Newmodel_cd/change_multi/ChangeFormer/checkpoints/test_pretrain_V5_aug
vis_root=/home/skymap/data/Newmodel_cd/change_multi/ChangeFormer/vis
data_name=LEVIR


img_size=256    
batch_size=18   
lr=0.01       
max_epochs=700
embed_dim=64

net_G=ChangeFormerV6        #ChangeFormerV6 is the finalized verion

lr_policy=linear
optimizer=sgd                 #Choices: sgd (set lr to 0.01), adam, adamw
loss=ce                         #Choices: ce, fl (Focal Loss), miou
multi_scale_train=True
multi_scale_infer=False
shuffle_AB=False

#Initializing from pretrained weights
pretrain=/home/skymap/data/Newmodel_cd/change_multi/ChangeFormer/checkpoints/change_multi_tet2024_v2/best_ckpt.pt

#Train and Validation splits
split=train         #trainval
split_val=test      #test
project_name=CD_AUG_${net_G}_${data_name}_b${batch_size}_lr${lr}_${optimizer}_${split}_${split_val}_${max_epochs}_${lr_policy}_${loss}_multi_train_${multi_scale_train}_multi_infer_${multi_scale_infer}_shuffle_AB_${shuffle_AB}_embed_dim_${embed_dim}

CUDA_VISIBLE_DEVICES=0 python main_cd.py --img_size ${img_size} --loss ${loss} --checkpoint_root ${checkpoint_root} --vis_root ${vis_root} --lr_policy ${lr_policy} --optimizer ${optimizer} --pretrain ${pretrain} --split ${split} --split_val ${split_val} --net_G ${net_G} --multi_scale_train ${multi_scale_train} --multi_scale_infer ${multi_scale_infer} --gpu_ids ${gpus} --max_epochs ${max_epochs} --project_name ${project_name} --batch_size ${batch_size} --shuffle_AB ${shuffle_AB} --data_name ${data_name}  --lr ${lr} --embed_dim ${embed_dim}