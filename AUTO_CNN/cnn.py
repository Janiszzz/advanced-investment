import torch
from torch.utils.data import Dataset
import cv2 as cv
from torch.utils.data import DataLoader

import torchvision.transforms as transforms

transform = transforms.Compose([
    #transforms.Resize([96, 180]),
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor()    
])

class myDataset(Dataset):
    def __init__(self,info):
        self.all_image_paths = info['file_name']
        self.all_image_labels = info['label']

    def __getitem__(self, index):
        img = cv.imread(self.all_image_paths[index])
        img = transform(img)

        label = int(self.all_image_labels[index])
        label = torch.tensor(label).to(torch.float32)
        return img, label
    def __len__(self):
        return len(self.all_image_paths)

#%%
import torch.nn as nn
class myNN(nn.Module):
    def __init__(self,size):
        super(myNN, self).__init__()
        self.cnnStack = nn.Sequential(
            nn.Conv2d(1, 64, (5, 5), padding = (2,2)),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d((2, 2), 1),

            nn.Conv2d(64, 128, (5, 5), padding = (2,2)),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d((2, 2), 1),
            
            nn.Conv2d(128, 256, (5, 5), padding = (2,2)),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d((2, 2), 1),
            
            nn.Conv2d(256, 512, (5, 5), padding = (2,2)),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d((2, 2), 1),
            
            nn.Flatten(),
            
            nn.Linear(512*size,2),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        return self.cnnStack(x)
    
    def printInfo(self):
        x = torch.rand([1,1,250,80])
        for name, module in self.cnnStack.named_children():
            x = module(x)
            if isinstance(module,nn.Upsample):
                print("Upsample({}) : {}".format(name,x.shape))
            elif isinstance(module,nn.Conv2d):
                print("Conv2d({}) : {}".format(name,x.shape))
            elif isinstance(module,nn.Flatten):
                print("Flatten({}) : {}".format(name,x.shape))
            elif isinstance(module,nn.MaxPool2d):
                print("MaxPool2d({}) : {}".format(name,x.shape))
            elif isinstance(module,nn.Linear):
                print("Linear({}) : {}".format(name,x.shape))  
            elif isinstance(module,nn.Softmax):
                print("Softmax({}) : {}".format(name,x.shape))  
            elif isinstance(module,nn.Softmax):
                print("Softmax({}) : {}".format(name,x.shape))       



#model.printInfo()
#%%

def train(dataloader, model, loss_fn, optimizer):
    
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        #print(X,y)
        pred = model(X).to(device)
        y = torch.eye(2).to(device)[y.long(), :]
        
        #print(pred,y)
        loss = loss_fn(pred, y)
        print(loss)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return loss.item()

def show(dataloader):
    for batch, (img,label) in enumerate(dataloader):
        img = transforms.ToPILImage()(img[0])
        img.show()
        
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            y = two_hot(y)
            test_loss += loss_fn(pred, y)
            pred = pred[:,0].ge(0.5)
            y = y[:,0].ge(0.5)
            #print(pred,y)
            correct += (pred == y).sum().item()
            
    test_loss /= len(dataloader.dataset)
    #print(f"Average Loss: {test_loss:>7f} Accuracy: {int(correct / len(dataloader.dataset) * 100)}%")
    return test_loss.item(), correct / len(dataloader.dataset)

#%%
if __name__ == "__main__":
    
    path = "./pic/"
    device = 'cpu'
    import preinput
    
    info = preinput.classification(path)
    #rolling_info
    ds = myDataset(info)
    print(len(ds))
    dl = DataLoader(ds, batch_size = 16, shuffle = True)
    model = myNN(246*76).to(device)
    loss_fn = nn.CrossEntropyLoss()

    epochs = 1000
    result = []
    lr = 0.0002
    optimizer = torch.optim.ASGD(model.parameters(), lr = lr)

    for t in range(epochs):
        #show(dl)
        train_loss = train(dl, model, loss_fn, optimizer)
        print([t, train_loss])
        