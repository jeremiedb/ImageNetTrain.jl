# Train Image Classification on Imagenet

Data downloaded from: https://www.kaggle.com/c/imagenet-object-localization-challenge/data using [Kaggle API](https://github.com/Kaggle/kaggle-api):

```
kaggle competitions download imagenet-object-localization-challenge
```

Training is performed by the `resnet-optim.jl` script: 

```
julia --project=@. --threads=8 resnet-optim.jl
```
