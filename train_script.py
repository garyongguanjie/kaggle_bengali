from naive_loader import BengaliDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import copy
import matplotlib.pyplot as plt

def train_model(model, criterion, optimizer, device, dataloaders, scheduler=None, num_epochs=25):
    since = time.time()

    best_acc = 0.0

    dataset_sizes = {'train': len(dataloaders['train'].dataset),'val': len(dataloaders['val'].dataset)}

    train_acc_list = []; train_loss_list= []; val_acc_list = []; val_loss_list = []; test_acc_list = []; test_loss_list = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataloaders[phase]:

                inputs = data['image']
                grapheme_root_label = data['grapheme_root']
                vowel_diacritic_label = data['vowel_diacritic']
                consonant_diacritic_label = data['consonant_diacritic']

                inputs = inputs.to(device)
                grapheme_root_label =  grapheme_root_label.to(device)
                vowel_diacritic_label = vowel_diacritic_label.to(device)
                consonant_diacritic_label =  consonant_diacritic_label.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    # preds = outputs.argmax(dim=1)

                    loss = criterion(outputs, grapheme_root_label.squeeze(1),vowel_diacritic_label.squeeze(1),consonant_diacritic_label.squeeze(1))

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)

                # running_corrects += torch.sum(preds == labels.data.squeeze(1))
            if phase == 'train' and scheduler != None:
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            # epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_acc = 0

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
            if phase == "train_eval":
                train_acc_list.append(epoch_acc)
                train_loss_list.append(epoch_loss)
            elif phase == "val":
                val_acc_list.append(epoch_acc)
                val_loss_list.append(epoch_loss)
            elif phase == "test":
                test_acc_list.append(epoch_acc)
                test_loss_list.append(epoch_loss)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    plots = (train_acc_list,train_loss_list,val_acc_list,val_loss_list,test_acc_list,test_loss_list)

    return model, plots

def plot_model_metrics(plots,name):
    train_acc_list,train_loss_list,val_acc_list,val_loss_list,test_acc_list,test_loss_list = plots
    plot(train_acc_list,val_acc_list,test_acc_list,"accuracy",name)
    plot(train_loss_list,val_loss_list,test_loss_list,"loss",name)


def plot(train,val,test,metric,name):
    plt.title(name)
    plt.plot(train,label="train {}".format(metric))
    plt.plot(val,label="val {}".format(metric))
    plt.plot(test,label="test {}".format(metric))
    plt.legend(loc="best")
    plt.savefig("{}-{}".format(name,metric))
    plt.close()
    
def loss(outputs,grapheme_root_label,vowel_diacritic_label,consonant_diacritic_label):

    grapheme_root_output = outputs[:,:168]
    vowel_diacritic_output = outputs[:,168:179]
    consonant_diacritic_output = outputs[:,179:186]
    gloss = nn.CrossEntropyLoss()(grapheme_root_output,grapheme_root_label)
    vloss = nn.CrossEntropyLoss()(vowel_diacritic_output,vowel_diacritic_label)
    closs = nn.CrossEntropyLoss()(consonant_diacritic_output,consonant_diacritic_label)

    return gloss + vloss + closs

if __name__ == "__main__":

    mean = 0.0692
    std = 0.205

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transformation = transforms.Compose([
                                         transforms.ToTensor(),
                                         transforms.Normalize([mean,mean,mean], [std,std,std])])


    train_data = BengaliDataset("split_1/validation_set_1.npy","split_1/validation_1.csv",transform=transformation)
    val_data = BengaliDataset("split_0/validation_set_0.npy","split_0/validation_0.csv",transform=transformation)
 


    train_loader = DataLoader(train_data, batch_size=64, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=64, num_workers=4)


    dataloaders = {'train': train_loader,'val': val_loader}

    # train all layers with pretrained model
    model = models.mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(model.last_channel, 186)
    print(model.classifier)
    model = model.to(device)
    criterion = loss
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model,plots = train_model(model, criterion, optimizer,
                device, dataloaders, num_epochs=2)
    # plot_model_metrics(plots,"graph")


    print("done")
