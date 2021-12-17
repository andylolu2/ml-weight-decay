# Exploring Effects of Weight Decay on Model Generalization

Repository for seeing how weight decay is able to help 
models generalize.

Go to the [blog folder](blog/) to read the associated blog post.

## Run

Main script example

```console
python -m src.train --train_size 5000 --val_size 500 --batch_size 64 \
    --num_layers 2 --hidden_dim 256 --max_epochs 5000 --lr 0.001 \
    --dataset binary_add --gpus 1 --l2_norm 0.1 --model resnet
```

Help

```console
python -m src.train -h
```