# Exploring Effects of Weight Decay on Model Generalization

Repository for seeing how weight decay is able to help 
models generalize.

You can read the associated blog on [Medium](https://medium.com/@andylolu24/weight-decay-and-its-peculiar-effects-66e0aee3e7b8?source=friends_link&sk=d4296f54a91775679d1521b77c763050). 
Alternatively, you can go to the [blog folder](blog/) to read it.

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