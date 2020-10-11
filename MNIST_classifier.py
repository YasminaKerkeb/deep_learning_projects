
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms
from torchvision.datasets import MNIST
import os

#Global values

COLOR_LIST = ['tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink',
              'tab:gray', 'tab:olive', 'tab:cyan', 'tab:red', 'tab:blue']


class MNISTClassifier(nn.Module):
    
    """
    Base class for MNIST Classifier model.
        
    """
    
    def __init__(self,num_epochs,batch_size,criterion=nn.CrossEntropyLoss(),num_classes=10):
        """
        Initialize a PyTorch Convolution network model given a loss function, optimizer and other parameters for training
        
        """
        super(MNISTClassifier, self).__init__()
        self.criterion=criterion
        self.num_epochs=num_epochs
        self.batch_size=batch_size
        # mnist images are (1, 28, 28) (channels, width, height) 
        self.layer_1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer_2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc_1 = nn.Linear(7 * 7 * 32, 1000)
        self.fc_2 = nn.Linear(1000, num_classes)

    def forward(self, x):

        #Layer 1
        x=self.layer_1(x)
        #Layer 2
        x=self.layer_2(x)
        #Reshaping the input
        x = x.view(-1, 7 * 7 * 32)
        #Implement Drop out
        x = self.drop_out(x)
        #Fully connected layer 1
        x=self.fc_1(x)
        #Fully connected layer 2
        x=self.fc_2(x)
        # probability distribution over labels
        x = torch.log_softmax(x, dim=1)

        return x

    def train_dataloader(self,data,**kwargs):
        return DataLoader(data, batch_size=self.batch_size,**kwargs)


    def test_dataloader(self,data,**kwargs):
        return DataLoader(data, batch_size=self.batch_size,**kwargs)

    def prepare_data(self):
        # transforms for images
        #The mean and variance are calculated beforehand (0.1305,0.3081)
        transform=transforms.Compose([transforms.ToTensor(), 
                                    transforms.Normalize((0.1305,), (0.3081,))])
        
        # prepare transforms standard to MNIST
        self.mnist_train = MNIST(os.getcwd(), train=True, download=True, transform=transform)
        self.mnist_test = MNIST(os.getcwd(), train=False, download=True, transform=transform)
        

        return self.mnist_train, self.mnist_test

    def training_step(self, train_batch,batch_idx):
        x, y = train_batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)

        logs = {'train_loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_step(self, val_batch,batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        return {'val_loss': loss}

    
    def compute_validation(self,val_data):
        with torch.no_grad():
                val_loss = []
                for (i,val_batch) in enumerate(val_data):
                    val_step=self.validation_step(val_batch,i)
                    val_loss.append(val_step["val_loss"])

                val_loss = torch.mean(torch.tensor(val_loss))

        return val_loss.item()



    def set_optimizer(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    @staticmethod
    def plot_learning_curve(learning_curve,ax,label,color):
        h, = ax.plot(learning_curve, color=color)
        h.set_label(label)
        ax.set_xlabel('Iterations')
        ax.set_xlim((0, len(learning_curve)))
        ax.set_ylabel('Loss')
        ax.set_title('Learning curves')

    @staticmethod
    def init_weights(m):
        #Use Xavier_uniform weight initialization
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def weight_initializer(self):
        #Use Xavier_uniform weight initialization
        self.apply(self.init_weights)
    

    def compute_training(self,data,tolerance=1e-5):
        best_final_loss=1e10
        total_step=len(data)
        old_loss=1e6
        for epoch in range(self.num_epochs):
            #Activate train mode
            self.train()
            #Initialize weights
            self.weight_initializer()
            #Store every loss to get the learning curve
            learning_curve = [] 
            for (i,train_batch) in enumerate(data):
                # Compute forward propagation
                train_step=self.training_step(train_batch,i)
                loss_value = train_step["log"]["train_loss"]
                learning_curve.append(loss_value)

                # Display Loss
                if (i + 1) % 100 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                    .format(epoch + 1, self.num_epochs, i + 1, total_step,loss_value))

                # Convergence check, see if the percentual loss decrease is within
                # tolerance:
                p_delta_loss = np.abs(loss_value.detach().numpy() - old_loss)/old_loss
                #Early Stopping
                if p_delta_loss < tolerance: break
                old_loss = loss_value.detach().numpy()

                # Compute backward propagation
                train_step["loss"].backward()
                self.set_optimizer().step()
                self.set_optimizer().zero_grad()

            # Validation at the end of an epoch
            #self.eval()
            #val_loss=self.compute_validation(self.mnist_val,i)
            #print('val_loss: ', val_loss)

            
            if loss_value < best_final_loss: 
                best_net = self
                best_final_loss = loss_value
                best_learning_curve = learning_curve
                

        
        return best_net, best_final_loss, best_learning_curve

                
    
                
    


    