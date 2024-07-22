from utilss import get_accuracy, plot_curve, keep_store_dict, store_dict_to_disk
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt

def test(model, test_loader, device):
    # Put the model in evaluation mode. 
    # Tells the model not to compute gradients. Increases inference speed.
    model.eval()
    test_acc = 0.0
    with torch.no_grad():
        for batch_num, (x, y) in tqdm(enumerate(test_loader, 0)):
            # Put the data to the appropriate device.
            x = x.to(device)
            y = y.to(device)
            y_hat = model(x)
            # Do inference. Forwad pass with the model.
            test_acc += get_accuracy(y_hat, y)
        print(f'test acc = {test_acc / batch_num}')            
    return test_acc / batch_num