
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms
from torchvision.datasets import MNIST
import torchvision.models as models
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns

#Global values

COLOR_LIST = ['tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink',
              'tab:gray', 'tab:olive', 'tab:cyan', 'tab:red', 'tab:blue']
GPU_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class StanfordMRNet(nn.Module):
    
    """
    Base class for MNIST Classifier model.
        
    """
    
    def __init__(self,num_epochs,batch_size,criterion=nn.BCEWithLogitsLoss()):
        """
        Initialize a PyTorch Convolution network model given a loss function, and other parameters for training

        Params:
        ------

        num_epochs:       An integer specifying number of replicates to train,
                          the neural network with the lowest loss is returned.
        batche_size:      An integer specifying the number of batches
                          to do (default 1000)
        criterion:        Loss function (default nn.BCEWithLogitsLoss())
        
        
        """
        super(StanfordMRNet, self).__init__()
        self.criterion=criterion
        self.num_epochs=num_epochs
        self.batch_size=batch_size 
        self.feature_extractor = models.alexnet(pretrained=True)
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, 1)

 
    def forward(self, x):

        #We use stochastic gradient descent, the training is then computed on each
        #training example
        x=torch.squeeze(x,dim=0)
        #Each image is passed through a AlexNet feature extractor
        extracted_features=self.feature_extractor(x).features
        #Average Pooling
        pooled_features=self.avg_pooling(extracted_features)
        #Flattenning the vectir
        flattened_output=torch.flatten(pooled_features,1,3)
        #Max pooling over the flattenned vector
        pooled_output=torch.max(flattened_output,0, keepdim=True)[0]
        # We don't need to add sigmoid activation since it's built in nn.BCEWithLogits()
        final_output=self.fc(pooled_output)

        return final_output



    def training_step(self, train_batch,batch_idx):
        x, y = train_batch
        x = x.to(GPU_DEVICE)
        y = y.to(GPU_DEVICE)
        logits = self.forward(x)
        loss = self.criterion(logits, y)

        logs = {'train_loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_step(self, val_batch,batch_idx):
        x, y = val_batch
        x = x.to(GPU_DEVICE)
        y = y.to(GPU_DEVICE)
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        return {'val_loss': loss,
                'y_pred':logits,
                'y_true':y}

    def set_optimizer(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    @staticmethod
    def plot_learning_curve(learning_curve,ax,color):
        h, = ax.plot(learning_curve, color=color)
        ax.set_xlabel('Iterations')
        ax.set_xlim((0, len(learning_curve)))
        ax.set_ylabel('Loss')
        ax.set_title('Learning curves')
        ax.legend()

    @staticmethod
    def plot_train_val_loss(losses,ax):
        train_loss, val_loss=losses
        epochs=range(1,len(train_loss)+1)
        ax.plot(epochs,train_loss, color='tab:red', label='Train loss')
        ax.plot(epochs,val_loss, color='tab:blue',label='Validation loss')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.set_title('Train and Validation Loss')
        ax.legend()

    @staticmethod
    def init_weights(m):
        #Use Xavier_uniform weight initialization
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def weight_initializer(self):
        #Use Xavier_uniform weight initialization
        self.apply(self.init_weights)
    

    def compute_training(self,train_data,val_data,tolerance=1e-4):
        """
        Params:
        ------

        data:             Train data
        tolerance:        A float describing the tolerance/convergence criterion
                          for minimum relative change in loss (default 1e-6)
        """
        best_final_loss=1e10
        total_step=len(train_data)
        old_loss=1e6
        epoch_training_loss=[]
        epoch_val_loss=[]
        for epoch in range(self.num_epochs):
            #Activate train mode
            self.train()
            #Initialize weights
            self.weight_initializer()
            #Store every loss to get the learning curve
            learning_curve = [] 
            for (i,train_batch) in enumerate(train_data):
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

            #Print final training loss
            mean_epoch_training_loss=torch.stack(learning_curve).mean()
            epoch_training_loss.append(mean_epoch_training_loss.item())
            # Validation at the end of an epoch
            self.eval()
            val_loss,_,_=self.compute_validation(val_data)
            epoch_val_loss.append(val_loss)
            print('\nEpoch validation loss: {}\n'.format(val_loss))
            if val_loss < best_final_loss: 
                best_model = self.state_dict()
                best_final_loss = val_loss
                best_learning_curve = learning_curve
            
        #Store training and validation losses across epochs
        epochs_losses=[epoch_training_loss,epoch_val_loss]
        
        return best_model, best_final_loss, best_learning_curve, epochs_losses

    def compute_validation(self,val_data):
        self.eval()
        with torch.no_grad():
                val_loss = []
                y_true=[]
                y_pred=[]
                for (i,val_batch) in enumerate(val_data):
                    val_step=self.validation_step(val_batch,i)
                    val_loss.append(val_step["val_loss"])
                    y_true.append(val_step["y_true"])
                    y_pred.append(val_step["y_pred"])
                val_loss = torch.stack(val_loss).mean()
                y_pred=torch.cat(y_pred)
                y_true=torch.cat(y_true)

        return val_loss.item(), y_true, y_pred



                
    
                
    


    