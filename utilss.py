<<<<<<< Updated upstream
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import json
import os


def sample_to_pil_image(sample: np.ndarray, image_shape):
    image = np.reshape(sample, newshape=image_shape)
    image = PIL.Image.fromarray(np.uint8(image*255))
    if image.size[0] < 140:
        new_size = (140, 140) 
    return image.resize(size=new_size)


def get_accuracy(y_hat, y):
    corrects = (torch.max(y_hat, 1)[1].view(y.size()).data == y.data).sum()
    accuracy = 100.0 * corrects / y_hat.shape[0]
    return accuracy.item()


def plot_curve(curves: Tuple[List], labels: Tuple[List], plot_name):
    
    fig, ax = plt.subplots(figsize=[6.4, 3])

    max_len = 0
    for curve in curves:
        if len(curve) > max_len:
            max_len = len(curve)
    
    x_axis = list(range(max_len))
    
    for curve, label in zip(curves, labels):
        ax.plot(x_axis, curve, label=label)
    
    ax.set_xlabel('epochs')
    ax.set_ylabel(plot_name)
    ax.legend()
    fig.tight_layout()
    return fig


def keep_store_dict(curve, label, store_dict: Dict=None):
    
    if store_dict is None:
        store_dict = {}
    
    if not isinstance(curve, list):
        if torch.is_tensor(curve):
            curve = curve.detach().numpy()
        
        if isinstance(curve, np.ndarray):
            curve = curve.tolist()
        else:
            curve = [curve]

    if label in store_dict:    
        store_dict[label].extend(curve)
    else:
        store_dict[label] = curve
    
    return store_dict


def store_dict_to_disk(file_path, store_dict: Dict, overwrite=False):
    
    if overwrite is False and os.path.isfile(file_path):
        prev_dict = load_dict_from_disk(file_path=file_path)
    else:
        prev_dict = None

    if prev_dict is not None:
        for key, val in store_dict.items():
            if key in prev_dict:
                prev_val: List = prev_dict[key]
                prev_val.extend(val)
                store_dict[key] = prev_val
    
    with open(file_path, 'w') as f:
        json.dump(store_dict, f, indent=4)


def load_dict_from_disk(file_path):
    with open(file_path, 'r') as f:
        store_dict = json.load(f)
    return store_dict
=======
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import json
import os


def sample_to_pil_image(sample: np.ndarray, image_shape):
    image = np.reshape(sample, newshape=image_shape)
    image = PIL.Image.fromarray(np.uint8(image*255))
    if image.size[0] < 140:
        new_size = (140, 140) 
    return image.resize(size=new_size)


"""def get_accuracy(targets, predictions):
    predictions = predictions.view(-1)
    targets = targets.view(-1)
    predictions = torch.round(predictions)
    correct = (predictions == targets).sum().item()
    total = targets.size(0)
    accuracy = correct / total
    return accuracy"""

def get_accuracy(l1, l2):
    list1 = l1.cpu()
    list1 = list1.detach().numpy()
    list2 = l2.cpu()
    list2 = list2.detach().numpy()
    size = list1.shape
    acc = 0

    koordsx1 = np.zeros((size[0]))
    koordsy1 = np.zeros((size[0]))
    koodrsx2 = np.zeros((size[0]))
    koordsy2 = np.zeros((size[0]))

    for i in range(size[0]):
        for j in range(size[1]):
            for k in range(size[2]):
                if list1[i][j][k] == 1:
                    koordsx1[i] += k 
                    koordsy1[i] += j 
                if list2[i][j][k] == 1:
                    koodrsx2[i] += k
                    koordsy2[i] += j

        koordsx1[i] /= size[1]**2
        koodrsx2[i] /= size[1]**2
        koordsy1[i] /= size[1]**2
        koordsy2[i] /= size[1]**2

    distancex = koodrsx2 - koordsx1
    distancey = koordsy2 - koordsy1
    euclidian_distance = np.sqrt(np.power(distancex, np.array([2] * distancex.shape[0])) + 
                                 np.power(distancey, np.array([2] * distancex.shape[0])))
    for i in range(len(list1)):
        if euclidian_distance[i] <= 3:
            acc += 1
    return acc / size[0]

def plot_curve(curves: Tuple[List], labels: Tuple[List], plot_name):
    
    fig, ax = plt.subplots(figsize=[6.4, 3])

    max_len = 0
    for curve in curves:
        if len(curve) > max_len:
            max_len = len(curve)
    
    x_axis = list(range(max_len))
    
    for curve, label in zip(curves, labels):
        ax.plot(x_axis, curve, label=label)
    
    ax.set_xlabel('epochs')
    ax.set_ylabel(plot_name)
    ax.legend()
    fig.tight_layout()
    return fig


def keep_store_dict(curve, label, store_dict: Dict=None):
    
    if store_dict is None:
        store_dict = {}
    
    if not isinstance(curve, list):
        if torch.is_tensor(curve):
            curve = curve.detach().numpy()
        
        if isinstance(curve, np.ndarray):
            curve = curve.tolist()
        else:
            curve = [curve]

    if label in store_dict:    
        store_dict[label].extend(curve)
    else:
        store_dict[label] = curve
    
    return store_dict


def store_dict_to_disk(file_path: str, store_dict: Dict, overwrite: bool = False):
    # Load existing dictionary if not overwriting
    if not overwrite and os.path.isfile(file_path):
        prev_dict = load_dict_from_disk(file_path)
    else:
        prev_dict = {}

    # Merge existing dictionary with the new one
    for key, val in store_dict.items():
        if key in prev_dict:
            # Ensure the existing value is a list before extending
            if isinstance(prev_dict[key], list):
                prev_dict[key].extend(val)  # Extend the existing list
            else:
                # If it's not a list, convert it to a list
                prev_dict[key] = [prev_dict[key]] + val
        else:
            # If the key does not exist, just add it
            prev_dict[key] = val
    
    # Write the merged dictionary back to disk
    with open(file_path, 'w') as f:
        json.dump(prev_dict, f, indent=4)


def load_dict_from_disk(file_path: str) -> Dict:
    if not os.path.isfile(file_path):
        return {}

    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {file_path}: {e}")
        return {}
>>>>>>> Stashed changes
