import torchvision.models as models
from torch.utils.data.dataset import Dataset
from PIL import Image
from os import path
from torchvision import transforms
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier as KNN
from torch.autograd import Variable
from sklearn.metrics import accuracy_score
from torch.optim import SGD
from torch.autograd import Variable
from torchnet.logger import VisdomPlotLogger, VisdomSaver
from torchnet.meter import AverageValueMeter
from Functions import *

train = True

np.random.seed(1234)
torch.random.manual_seed(1234)

#Variabilil globali
width = 256
height = 256
mean_pre_trained =[0.485, 0.456, 0.406]
std_pre_trained =[0.229, 0.224, 0.225]

transformss = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean_pre_trained,std_pre_trained)])
barilla_train = ScenesDataset('Dataset/Train','train.txt',transform=transformss)
barilla_test = ScenesDataset('Dataset/Test','test.txt',transform=transformss)
barilla_train_loader = torch.utils.data.DataLoader(barilla_train, batch_size=10, num_workers=0, shuffle=True)
barilla_test_loader = torch.utils.data.DataLoader(barilla_test, batch_size=10, num_workers=0)

net = models.alexnet(pretrained=True)

for param in net.parameters():
    param.requires_grad = False
net.classifier[6] = nn.Linear(4096, 4) #Numero esatto di classi nel nostro dataset.
sum([p.numel() for p in net.parameters()])

if train:
    lenet_mnist, lenet_mnist_logs = train_classification(net, epochs=20, train_loader = barilla_train_loader,
                                                         test_loader = barilla_test_loader, lr=0.001)
else:
    net.load_state_dict(torch.load("./net.pth"))


lenet_mnist_predictions, lenet_mnist_gt = test_model_classification(net, barilla_train_loader)
print ("Accuracy aLexNet di train su barillatestloader: %0.2f" % accuracy_score(lenet_mnist_gt,lenet_mnist_predictions.argmax(1)))

lenet_mnist_predictions, lenet_mnist_gt = test_model_classification(net, barilla_test_loader)
print ("Accuracy aLexNet di test su barillatestloader: %0.2f" % accuracy_score(lenet_mnist_gt,lenet_mnist_predictions.argmax(1)))

knn_1 = KNN(n_neighbors=3)

if torch.cuda.is_available():
    net = net.cuda()
    torch.cuda.empty_cache()
net.eval()
    #
barilla_train_loader_OB = torch.utils.data.DataLoader(barilla_train, batch_size=1, num_workers=0, shuffle=True)
barilla_test_loader_OB = torch.utils.data.DataLoader(barilla_test, batch_size=1, num_workers=0)

input_for_datafram_train, label_array_train = get_dataframe(barilla_train_loader_OB, net)
df = pd.DataFrame(input_for_datafram_train)
knn_1.fit(df, label_array_train)
feature_test = extract_features(barilla_test_loader_OB, net)
#print("Accuracy con rete preallenata e dataset base.")
#print(accuracy(knn_1, feature_test))

img_test = Image.open("Stefano.jpg").convert("RGB")
img_test = img_test.resize((256, 256))
img_test = transformss(img_test)

#params = net(img_test)
#print(params)
#df = pd.DataFrame(params)
print(feature_test[8]["label"])
print(knn_1.predict(feature_test[8]["feature"].cpu().detach().numpy().reshape(1, -1)))

