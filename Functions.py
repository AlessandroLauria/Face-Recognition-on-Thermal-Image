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


#Variabilil globali
width = 256
height = 256
mean_pre_trained =[0.485, 0.456, 0.406]
std_pre_trained =[0.229, 0.224, 0.225]

class ScenesDataset(Dataset):
    def __init__(self, base_path, txt_list, transform=None):
        # conserviamo il path alla cartella contenente le immagini
        self.base_path = base_path
        # carichiamo la lista dei file
        # sarà una matrice con n righe (numero di immagini) e 2 colonne (path, etichetta)
        self.images = np.loadtxt(txt_list, dtype=str, delimiter=',')
        # print("self.images ha i seguenti elementi:", len(self.images))
        # conserviamo il riferimento alla trasformazione da applicare
        self.transform = transform

    def __getitem__(self, index):
        # print("Get item numero -->", index)
        # recuperiamo il path dell'immagine di indice index e la relativa etichetta
        f, c = self.images[index]
        # carichiamo l'immagine utilizzando PIL e facciamo il resize a 3 canali.
        im = Image.open(path.join(self.base_path, f)).convert("RGB")

        # Resize:
        im = im.resize((width, height))
        # se la trasfromazione è definita, applichiamola all'immagine
        if self.transform is not None:
            im = self.transform(im)

            # convertiamo l'etichetta in un intero
        label = int(c)
        # restituiamo un dizionario contenente immagine etichetta
        # print("Mentre creo il tutto, label vale-->", label, ", name vale -->", f)
        return {'image': im, 'label': label, 'name': f}

    # restituisce il numero di campioni: la lunghezza della lista "images"
    def __len__(self):
        # print("Ho invocato len, vale-->", len(self.images))
        return len(self.images)


def get_mean_devst(dataset):
    m = np.zeros(3)
    for sample in dataset:
        m+= np.array(sample['image'].sum(1).sum(1)) #accumuliamo la somma dei pixel canale per canale
    #dividiamo per il numero di immagini moltiplicato per il numero di pixel
    m=m/(len(dataset)*width*height)
    #procedura simile per calcolare la deviazione standard
    s = np.zeros(3)
    for sample in dataset:
        s+= np.array(((sample['image']-torch.Tensor(m).view(3,1,1))**2).sum(1).sum(1))
    s=np.sqrt(s/(len(dataset)*width*height))
    print("Medie",m)
    print("Dev.Std.",s)
    return m, s


# Prende in input l'array di feature e il classificatore(knn.)
def accuracy(classifier, samples):
    right_pred = 0
    for i in range(len(samples)):
        pred_label = classifier.predict(samples[i]["feature"].cpu().detach().numpy().reshape(1, -1))
        if pred_label[0] == samples[i]["label"]:
            right_pred += 1

    return float(right_pred) / len(samples)



def extract_features(dataset, net):
    #Presa ogni riga del dataloader li passa alla net senza attivare il layer di classificazione
    feature_dataset = []
    print("Avviato extract_feature.")
    for i, dataset_train in enumerate(dataset):
        x=Variable(dataset_train['image'], requires_grad=False)
        y=Variable(dataset_train['label'])
        x, y = x.cpu(), y.cpu()
        if torch.cuda.is_available():
            x, y = x.cuda(), y.cuda()
            #print("Con cuda")
        output = net(x)
        feature_dataset.append({"label": dataset_train['label'], "feature":output, "name": dataset_train['name']})
    return feature_dataset


def get_dataframe(dataset, net):
    print("Avviato get_dataframe.")
    feature_dataset = extract_features(dataset, net)
    feature_dataset_matrix = np.zeros((len(feature_dataset), len(feature_dataset[0]["feature"][0])))
    #Qui abbiamo nelle righe tutte le immagini, nella lable feature tutte le 9000 colonne, ossia le feature.
    label_array = np.zeros(len(feature_dataset))
    for i in range(0, len(feature_dataset)):#302
        for j in range(0, len(feature_dataset[0]["feature"][0])):#9206
            if j == 0:#salviamo la y finale nell'array label_array
                label_array[i] = feature_dataset[i]['label'][0]
                #print(i, end= " ")
            feature_dataset_matrix[i][j] =feature_dataset[i]["feature"][0][j]

    return feature_dataset_matrix, label_array



def train_classification(model, train_loader, test_loader, lr=0.01, epochs=20, momentum=0.9):
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr, momentum=momentum)
    loaders = {'train': train_loader, 'test': test_loader}
    losses = {'train': [], 'test': []}
    accuracies = {'train': [], 'test': []}

    exp_name = 'experiment_50_epochs'
    loss_meter = AverageValueMeter()
    acc_meter = AverageValueMeter()
    loss_logger = VisdomPlotLogger('line', env=exp_name, opts={'title': 'Loss', 'legend': ['train', 'test']})
    acc_logger = VisdomPlotLogger('line', env=exp_name, opts={'title': 'Accuracy', 'legend': ['train', 'test']})
    visdom_saver = VisdomSaver(envs=[exp_name])

    if torch.cuda.is_available():
        model = model.cuda()
    for e in range(epochs):
        # print("Primo ciclo for.")
        for mode in ['train', 'test']:
            # print("Secondo ciclo for.")

            loss_meter.reset()
            acc_meter.reset()

            if mode == 'train':
                model.train()
            else:
                model.eval()
            epoch_loss = 0
            epoch_acc = 0
            samples = 0
            # print("Mode-->",mode)
            # print("Enumerate-->", loaders[mode])
            for i, batch in enumerate(loaders[mode]):
                # trasformiamo i tensori in variabili
                x = Variable(batch['image'], requires_grad=(mode == 'train'))
                y = Variable(batch['label'])
                if torch.cuda.is_available():
                    x, y = x.cuda(), y.cuda()
                    print("Con cuda")
                # else:
                # print("Senza cuda")
                output = model(x)
                # print(type(output))
                # print(output)
                l = criterion(output, y)
                if mode == 'train':
                    l.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                # print("L-->",l.item())
                acc = accuracy_score(y.cpu().data, output.cpu().max(1)[1].data)
                epoch_loss += l.data.item() * x.shape[0]
                epoch_acc += acc * x.shape[0]
                samples += x.shape[0]
                print ("\r[%s] Epoch %d/%d. Iteration %d/%d. Loss: %0.2f. Accuracy: %0.2f\t\t\t\t\t" % \
                       (mode, e + 1, epochs, i, len(loaders[mode]), epoch_loss / samples, epoch_acc / samples),
                       epoch_loss / samples,
                       epoch_acc / samples,
                       losses[mode].append(epoch_loss))
                accuracies[mode].append(epoch_acc)
                n = batch['image'].shape[0]
                loss_meter.add(l.item() * n, n)
                acc_meter.add(acc * n, n)
                loss_logger.log(e + (i + 1) / len(loaders[mode]), loss_meter.value()[0], name=mode)
                acc_logger.log(e + (i + 1) / len(loaders[mode]), acc_meter.value()[0], name=mode)

            loss_logger.log(e + 1, loss_meter.value()[0], name=mode)
            acc_logger.log(e + 1, acc_meter.value()[0], name=mode)

            # print("Fine secondo ciclo for")
        print("\r[%s] Epoch %d/%d. Iteration %d/%d. Loss: %0.2f. Accuracy: %0.2f\t\t\t\t\t" % \
              (mode, e + 1, epochs, i, len(loaders[mode]), epoch_loss, epoch_acc))

    torch.save(model.state_dict(), "./net.pth")
    print("Ho finito.")
    # restituiamo il modello e i vari log
    return model, (losses, accuracies)



def test_model_classification(model, test_loader):
    softmax = nn.Softmax(dim=1)
    model.eval()
    preds = []
    gts = []
    for batch in test_loader:
        x=Variable(batch["image"])
        #applichiamo la funzione softmax per avere delle probabilità
        if torch.cuda.is_available():
            x = x.cuda()
        pred = softmax(model(x)).data.cpu().numpy().copy()
        gt = batch["label"].cpu().numpy().copy()
        #print("Pred-->", pred, ", gt-->", gt)
        preds.append(pred)
        gts.append(gt)
        #print(len(preds), len(gts))
    return np.concatenate(preds),np.concatenate(gts)