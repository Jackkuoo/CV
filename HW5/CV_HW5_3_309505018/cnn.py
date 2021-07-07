import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device=torch.device('cpu')

class ResNet34(nn.Module):
    def __init__(self,num_class):
        super(ResNet34,self).__init__()
        self.model = models.resnet34(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad=False
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 15)

    def forward(self,X):
        out=self.model(X)
        return out

class VGG16(nn.Module):
    def __init__(self,num_class):
        super(VGG16,self).__init__()
        self.model = models.vgg16(pretrained=True)

        for param in self.model.features.parameters():
            param.require_grad = False

        num_features = self.model.classifier[6].in_features
        features = list(self.model.classifier.children())[:-1] # Remove last layer
        features.extend([nn.Linear(num_features, 15)]) # Add our layer with 4 outputs
        self.model.classifier = nn.Sequential(*features) # Replace the model classifier

    def forward(self,X):
        out=self.model(X)
        return out


def train(model,dataloader,criterion,optimizer,epochs):
    model.train()
    for epoch in range(1,epochs+1):
        total_loss = 0
        correct = 0
        for images,targets in dataloader:
            images,targets = images.to(device),targets.to(device)
            predicts = model(images)
            loss = criterion(predicts, targets)

            # bp
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # statistic
            total_loss += loss.item() * len(images)
            correct += predicts.max(dim=1)[1].eq(targets).sum().item()
        print(f'epoch{epoch}  Loss: {total_loss/len(dataloader.dataset):.4f} Acc: {correct/len(dataloader.dataset):.2f}')


def test(model,dataloader):
    model.eval()
    correct=0
    for images,targets in dataloader:
        images, targets = images.to(device), targets.to(device)
        predicts = model(images)

        # statistic
        correct += predicts.max(dim=1)[1].eq(targets).sum().item()
    print(f'Acc: {correct/len(dataloader.dataset):.2f}')


epochs_feature_extraction=5
epochs_fine_tuning=15
batch_size = 64
lr = 1e-3


if __name__=='__main__':
    # load data
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_data = ImageFolder(os.path.join('hw5_data', 'train'), transform=train_transforms)
    test_data = ImageFolder(os.path.join('hw5_data', 'test'), transform=test_transforms)
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=4)

    # create model

    model=VGG16(num_class=15)
    model.to(device)
    # loss function
    criterion = nn.CrossEntropyLoss()
    print("1111")
    # feature extraction
    params_to_update = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)
    optimizer = optim.SGD(params_to_update, lr=lr, momentum=0.9)
    print("2222")
    train(model, train_loader, criterion, optimizer,epochs_feature_extraction)
    print("3333")
    # finetuning
    for param in model.parameters():
        param.requires_grad = True
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    train(model, train_loader, criterion, optimizer, epochs_fine_tuning)

    # save model wts
    model_wts = copy.deepcopy(model.state_dict())
    torch.save(model_wts, 'model.pt')

    # test
    test(model, test_loader)