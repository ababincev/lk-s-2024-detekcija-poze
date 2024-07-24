from utilss import get_accuracy_train, get_accuracy_test, plot_curve, keep_store_dict, store_dict_to_disk
from tqdm import tqdm
from test import test
import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision.io import read_image
from torch.utils.data import Dataset
from data_loader import DataLoader1
from torchvision import transforms
import time


def train(model, num_epochs, train_loader, store_dict, test_loader, device, loss_function, optimizer, batch_size):
    for epoch in range(num_epochs):
        train_running_loss = 0.0
        train_acc = 0.0
        model = model.train()
        train_data_loader = DataLoader(train_loader, batch_size=batch_size, shuffle=True)
        print(train_data_loader)
        valid_data_loader = DataLoader(test_loader, batch_size=batch_size, shuffle=True)
        #train_loader.used = []
        for batch_num, (x, y) in tqdm(enumerate(train_data_loader)):

            x = x.to(device)
            y = y.to(device)

            y_hat = model(x)
            if y_hat is None:
                raise ValueError("Model output y_hat is None")
            loss = loss_function(y_hat.float(), y.float())

            train_running_loss += loss.detach().item()
            train_acc += get_accuracy_train(l1=y_hat, l2=y)

            #optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            #             # Measure time for transferring data to the device
            #             start_time = time.time()
            #             x = x.to(device)
            #             y = y.to(device)
            #             end_time = time.time()
            #             print(f"Time to transfer data to device: {end_time - start_time:.6f} seconds")

            #             # Measure time for forward pass
            #             start_time = time.time()
            #             y_hat = model(x)
            #             if y_hat is None:
            #                 raise ValueError("Model output y_hat is None")
            #             end_time = time.time()
            #             print(f"Time for forward pass: {end_time - start_time:.6f} seconds")

            #             # Measure time for loss computation
            #             start_time = time.time()
            #             loss = loss_function(y_hat.float(), y.float())
            #             end_time = time.time()
            #             print(f"Time for loss computation: {end_time - start_time:.6f} seconds")

            #             train_running_loss += loss.detach().item()

            #             # Measure time for accuracy computation
            #             start_time = time.time()
            #             train_acc += get_accuracy(l1=y_hat, l2=y)
            #             end_time = time.time()
            #             print(f"Time for accuracy computation: {end_time - start_time:.6f} seconds")

            #             # Measure time for backward pass
            #             start_time = time.time()
            #             loss.backward()
            #             end_time = time.time()
            #             print(f"Time for backward pass: {end_time - start_time:.6f} seconds")

            #             # Measure time for optimizer step
            #             start_time = time.time()
            #             optimizer.step()
            #             end_time = time.time()
            #             print(f"Time for optimizer step: {end_time - start_time:.6f} seconds")

            #             # Optional: measure time for zeroing gradients
            #             #start_time = time.time()
            #             #optimizer.zero_grad()
            #             #end_time = time.time()
            #             #print(f"Time to zero gradients: {end_time - start_time:.6f} seconds")
            
        #time for storing dict
        #start_time = time.time()
        for param in model.parameters():
            param1 = param.cpu()
            store_dict = keep_store_dict(curve=param1, label='after_optimizer_step', store_dict=store_dict)
        #end_time = time.time()
        #print(f"time for storing dict: {end_time - start_time:.6f} seconds")
            
            
        epoch_loss = train_running_loss / batch_num
        epoch_acc = train_acc / batch_num
        store_dict = keep_store_dict(curve=epoch_loss, label='train_loss', store_dict=store_dict)
        store_dict = keep_store_dict(curve=epoch_acc, label='train_acc', store_dict=store_dict)
        print('Epoch: %d | Loss: %.4f | Train Accuracy: %.2f' \
              %(epoch + 1, epoch_loss, epoch_acc))
        v = time.asctime().split()
        v = '_'.join(v)
        torch.save(model,'/notebooks/lk-s-2024-detekcija-poze/model_'+ v +'.pth')

        if test_loader is not None:
            test_acc = test(model=model, test_loader=test_loader, device=device)
            store_dict = keep_store_dict(curve=test_acc, label='test_acc', store_dict=store_dict)
        
    return store_dict