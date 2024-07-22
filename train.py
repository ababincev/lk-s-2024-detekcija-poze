from utilss import get_accuracy, plot_curve, keep_store_dict, store_dict_to_disk
from tqdm import tqdm
from test import test
import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision.io import read_image
from torch.utils.data import Dataset
from torchvision import transforms


def train(model, num_epochs, train_loader, store_dict, test_loader, device, loss_function, optimizer, batch_size):
    for epoch in range(num_epochs):
        train_running_loss = 0.0
        train_acc = 0.0

        model = model.train()
        
        for batch_num, (x, y) in tqdm(enumerate(train_loader)):
            
            x = x.to(device)
            y = y.to(device)
            
            y_hat = model(x)
            if y_hat is None:
                raise ValueError("Model output y_hat is None")
            loss = loss_function(y_hat.float(), y.float())
            
            train_running_loss += loss.detach().item()
            train_acc += get_accuracy(l1=y_hat, l2=y)
            
            
            if batch_num % batch_size == 0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        epoch_loss = train_running_loss / batch_num
        epoch_acc = train_acc / batch_num
        
        store_dict = keep_store_dict(curve=epoch_loss, label='train_loss', store_dict=store_dict)
        store_dict = keep_store_dict(curve=epoch_acc, label='train_acc', store_dict=store_dict)
        print('Epoch: %d | Loss: %.4f | Train Accuracy: %.2f' \
            %(epoch + 1, epoch_loss, epoch_acc))

        if test_loader is not None:
            test_acc = test(model=model, test_loader=test_loader, device=device)
            store_dict = keep_store_dict(curve=test_acc, label='test_acc', store_dict=store_dict)
        
    return store_dict