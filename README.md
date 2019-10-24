# code for Asymmetric Valleys: Beyond Sharp and Flat Local Minima
This repo contains the code used for NeurIPS 2019 paper "Asymmetric Valleys: Beyond Sharp and Flat Local Minima". [Paper Link](https://arxiv.org/abs/1902.00744v2)  
Main experiment results can be directly reimplemented.  

## Environment:

```
pytorch == 1.2.0
```

## Find asymmetric valleys in 2D case
Run
```
python3 logistic_regression_2params.py
```

## Find asymmetric valleys in deep neural networks


The experimental results can be directly reproduced by the following code. Take ResNet164 on CIFAR10 as an example.    
First you need to train a standard model by SGD and a correspending swa model.   
```
python3 train2.py --dir=[save dir name] --dataset=CIFAR10 --data_path=[data path] --model=PreResNet164 --epochs=225 --lr_init=0.1 --wd=3e-4 --swa --swa_start=126 --swa_lr=0.05 --eval_freq=1 --save_freq=2 --cuda_visible_devices=0
```

Now you have several checkpoints. To find an asymmetric valley, you need to continuely run sgd after the obtained swa model. This helps you find another solution locates in an asymmetric valley.  
```
python3 sgd_after_swa.py --dir=savdirname --dataset=CIFAR10 --data_path=[data path] --model=PreResNet164  --wd=3e-4  --cuda_visible_devices=0 --resume=./train164/checkpoint-225.pt --lr_set=0.001
```

Next you can calculate the loss shape between the two resumed solutions (model1_resume loads a sgd model and model2_resume loads a swa model by default).
```
python3 interpulation.py --dir=[save dir name] --dataset=CIFAR10 --data_path=[data path] --model=PreResNet164 --wd=3e-4 --model1_resume=[sgd after swa dir]/[checkpoint] --model2_resume=./[train2 dir]/[checkpoint] --cuda_visible_devices=0 --division_part=40 --distances=80
```
The result (several txt and png files) will be saved.  

## Reference
Part of the code is adapted from [Averaging Weights Leads to Wider Optima and Better Generalization](https://github.com/timgaripov/swa). 
