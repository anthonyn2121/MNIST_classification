# MNIST_classification
The 'Hello World' of machine learning, this is a project to classify hand drawn digits from 0 to 9. The method I used here was to create a Convolutional Neural Network (CNN) written in Torch, although SVMS can work just as well. 

Final results are > 99% accuracy

-The code can be checked in 'mnist.ipynb' 
-The trained model is saved as 'cnn.th'

## Network Architecture: 
```python 
CNN(
  (conv_net): Sequential(
    (0): Conv2d(1, 20, kernel_size=(5, 5), stride=(1, 1))
    (1): ReLU(inplace=True)
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (3): Conv2d(20, 50, kernel_size=(5, 5), stride=(1, 1))
    (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (5): ReLU(inplace=True)
  )
  (fc): Sequential(
    (0): Linear(in_features=800, out_features=500, bias=True)
    (1): ReLU(inplace=True)
    (2): Linear(in_features=500, out_features=10, bias=True)
  )
)
```

## Requirements 
- torch 1.5.0
- torchvision 0.6.0 
- numpy 
- pandas 
