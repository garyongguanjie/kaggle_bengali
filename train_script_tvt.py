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

    dataset_sizes = {'train': len(dataloaders['train'].dataset),'val': len(dataloaders['val'].dataset),'test':len(dataloaders['test'].dataset)}

    train_acc_list = []; train_loss_list= []; val_acc_list = []; val_loss_list = []; test_acc_list = []; test_loss_list = []

    for epoch in range(num_epochs):
        start = time.time()
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val','test']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            grapheme_corrects = 0
            vowel_corrects = 0
            consonant_corrects = 0


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
                    
                    grapheme_preds = outputs[:,:168].argmax(dim=1)
                    vowel_preds = outputs[:,168:179].argmax(dim=1) 
                    consonant_preds = outputs[:,179:186].argmax(dim=1)

                    loss = criterion(outputs, grapheme_root_label.squeeze(1),vowel_diacritic_label.squeeze(1),consonant_diacritic_label.squeeze(1))

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                

                grapheme_corrects += torch.sum(grapheme_preds == grapheme_root_label.data.squeeze(1))
                vowel_corrects += torch.sum(vowel_preds == vowel_diacritic_label.data.squeeze(1))
                consonant_corrects += torch.sum(consonant_preds== consonant_diacritic_label.data.squeeze(1))
                
                # running_corrects += torch.sum(preds == labels.data.squeeze(1))
            if phase == 'train' and scheduler != None:
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]

            running_corrects = 0.5*grapheme_corrects.double() + 0.25*vowel_corrects.double() + 0.25*consonant_corrects.double()

            # epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            
            if phase == "train":
                # Note this are running values (calculated per batch) rather than actual values at the end of each epoch
                # Decreases training time
                # Not accurate especially at first few epochs
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
        end = time.time()
        print(f"time per epoch:{end-start}s")

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

    return 0.5*gloss + 0.25*vloss + 0.25*closs

def evaluate_test(model,criterion,dataloader,device):
    running_loss = 0.0
    grapheme_corrects = 0.0
    vowel_corrects = 0.0
    consonant_corrects = 0.0
    for data in dataloader:

        inputs = data['image']

        grapheme_root_label = data['grapheme_root']
        vowel_diacritic_label = data['vowel_diacritic']
        consonant_diacritic_label = data['consonant_diacritic']

        inputs = inputs.to(device)

        grapheme_root_label =  grapheme_root_label.to(device)
        vowel_diacritic_label = vowel_diacritic_label.to(device)
        consonant_diacritic_label =  consonant_diacritic_label.to(device)

        with torch.no_grad():
            outputs = model(inputs)
            
            grapheme_preds = outputs[:,:168].argmax(dim=1)
            vowel_preds = outputs[:,168:179].argmax(dim=1)
            consonant_preds = outputs[:,179:186].argmax(dim=1)

            loss = criterion(outputs, grapheme_root_label.squeeze(1),vowel_diacritic_label.squeeze(1),consonant_diacritic_label.squeeze(1))


        # statistics
        running_loss += loss.item() * inputs.size(0)
        

        grapheme_corrects += torch.sum(grapheme_preds == grapheme_root_label.data.squeeze(1))
        vowel_corrects += torch.sum(vowel_preds == vowel_diacritic_label.data.squeeze(1))
        consonant_corrects += torch.sum(consonant_preds== consonant_diacritic_label.data.squeeze(1))
        
    
    loss = running_loss / len(dataloader.dataset)

    running_corrects = 0.5*grapheme_corrects.double() + 0.25*vowel_corrects.double() + 0.25*consonant_corrects.double()

    # epoch_acc = running_corrects.double() / dataset_sizes[phase]
    acc = running_corrects / len(dataloader.dataset)

    print('{} Loss: {:.4f} Acc: {:.4f}'.format(
        "Final Test Accuracy", loss, acc))
            


if __name__ == "__main__":


    mean = 0.0692
    std = 0.205

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transformation = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize([mean,mean,mean], [std,std,std])])


    train_data = BengaliDataset("tvt_train2.npy","tvt_train_label2.csv",transform=transformation)
    val_data = BengaliDataset("tvt_val2.npy","tvt_val_label2.csv",transform=transformation)
    test_data = BengaliDataset("tvt_test2.npy","tvt_test_label2.csv",transform=transformation)


    train_loader = DataLoader(train_data, batch_size=64, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=64, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=64, num_workers=4)

    dataloaders = {'train': train_loader,'val': val_loader,'test':test_loader}

    # train all layers with pretrained model
    model = models.mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(model.last_channel, 186)
    print(model.classifier)
    model = model.to(device)
    criterion = loss
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model,plots = train_model(model, criterion, optimizer,
                device, dataloaders, num_epochs=60)

    evaluate_test(model,criterion,test_loader,device)
    # plot_model_metrics(plots,"graph")


    print("done")