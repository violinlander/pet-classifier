import torchvision.models as models
resnet18 = models.resnet18(pretrained=True)

# if we call resnet18
resnet18
# by itself, it will display the model. We see at the very bottom a (fc) fully
# connected layer with 1000 outputs. We will change that layer shortly.
num_ftrs = 2
resnet18.fc = nn.Linear(num_ftrs, 102)

