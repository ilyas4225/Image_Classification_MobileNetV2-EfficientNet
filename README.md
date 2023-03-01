# Image_Classification_MobileNetV2-EfficientNet
In this project, we have used two models  for image classification ,like EfficientNetB0 and MobileNETV2 on different dataset 

Firstly, we will pass the chosen dataset in the argument  of dataset and set the batch size  and number of epochs to train the model and set the name of saving log file.
The Command is 
```
python train.py --dataset DTD --batch_size 16 --epochs 50 --save EfficientNetB0_DTD
```

The datset we are using:
* [CIFAR10](https://pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html#torchvision.datasets.CIFAR10)
* [CIFAR100](https://pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR100.html#torchvision.datasets.CIFAR100)
* [DTD](https://pytorch.org/vision/stable/generated/torchvision.datasets.DTD.html#torchvision.datasets.DTD)
* [Food101](https://pytorch.org/vision/stable/generated/torchvision.datasets.Food101.html#torchvision.datasets.Food101)
* [OxfordIIITPet](https://pytorch.org/vision/stable/generated/torchvision.datasets.OxfordIIITPet.html#torchvision.datasets.OxfordIIITPet)
* [FLower102](https://pytorch.org/vision/stable/generated/torchvision.datasets.Flowers102.html#torchvision.datasets.Flowers102)
* [StanfordCars](https://pytorch.org/vision/stable/generated/torchvision.datasets.StanfordCars.html#torchvision.datasets.StanfordCars)
* [STL10](https://pytorch.org/vision/stable/generated/torchvision.datasets.STL10.html#torchvision.datasets.STL10)


## Benchmarks on GPU
Include the benchmark results of running multiple model precisions. 
 The Performance tests were run on GPU.

### when model is EfficeintNETB0

| Dataset  | Classes|Resolution| Epochs|Accuracy|
|----------|--------|----------|-------|--------|
| Cifar10  | 10     | 32x32    |20     |71.1999%|
| Cifar100 | 100    | 32x32    |50     |27.21 % |
| CINIC10  | 10     | 224x224  |100    |79.5578%|
| DTD      | 47     | 224x224  |50     |30.2127 |
| Flower102| 102    | 224x224  |50     |37.25490|
| Pets     | 37     | 224X224  |50     |65.16304|
| STL10    |10      | 224x224  |20     |63.00004|
| Food101  |101     | 224x224  |15     |50.82508|
| Cars     |196     | 224x224  |       |        |

### When model is MobileNET_V2

| Dataset  | Classes|Resolution| Epochs|Accuracy|
|----------|--------|----------|-------|--------|
| Cifar10  | 10     | 32x32    |15     |64.7699%|
| Cifar100 | 100    | 32x32    |50     |29.05 % |
| CINIC10  | 10     | 224x224  |15     |73      |
| DTD      | 47     | 224x224  |50     |26.64   |
| Flower102| 102    | 224x224  |50     |34.1176 |
| Pets     | 37     | 224X224  |50     |76.60356|
| Food101  |101     | 224x224  |20     |51.04505|



