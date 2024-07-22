import numpy as np
import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from initializers import gaussian_initializer, constant_initializer



class ModelAVE(nn.Module):
    def __init__(self):
        super(ModelAVE, self).__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=9, stride=8)

    def forward(self, x):
        center_map = x[:, 3:, :, :]
        return self.avg_pool(center_map)



class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        self.conv1_stage1 = nn.Conv2d(3, 128, kernel_size=9, padding=4)
        self.conv2_stage1 = nn.Conv2d(128, 128, kernel_size=9, padding=4)
        self.conv3_stage1 = nn.Conv2d(128, 128, kernel_size=9, padding=4)
        self.conv4_stage1 = nn.Conv2d(128, 32, kernel_size=5, padding=2)
        self.conv5_stage1 = nn.Conv2d(32, 512, kernel_size=9, padding=4)
        self.conv6_stage1 = nn.Conv2d(512, 512, kernel_size=1)
        self.conv7_stage1 = nn.Conv2d(512, 15, kernel_size=1)
        self.pool_center_lower = None
        
    def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    gaussian_initializer()(m.weight)
                    if m.bias is not None:
                        constant_initializer()(m.bias)

    def forward(self, x):
        image = x[:, :3, :, :]
        x1 = F.relu(self.conv1_stage1(image))
        x1 = F.max_pool2d(x1, kernel_size=3, stride=2)
        x1 = F.relu(self.conv2_stage1(x1))
        x1 = F.max_pool2d(x1, kernel_size=3, stride=2)
        x1 = F.relu(self.conv3_stage1(x1))
        x1 = F.max_pool2d(x1, kernel_size=3, stride=2)
        x1 = F.relu(self.conv4_stage1(x1))
        x1 = F.relu(self.conv5_stage1(x1))
        x1 = F.relu(self.conv6_stage1(x1))
        x1 = self.conv7_stage1(x1)
        return x1
    



class Model2(nn.Module):
    def __init__(self):
        super(Model2, self).__init__()
        self.conv1_stage2 = nn.Conv2d(3, 128, kernel_size=9, padding=4)
        self.conv2_stage2 = nn.Conv2d(128, 128, kernel_size=9, padding=4)
        self.conv3_stage2 = nn.Conv2d(128, 128, kernel_size=9, padding=4)
        self.conv4_stage2 = nn.Conv2d(128, 32, kernel_size=5, padding=2)

    def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    gaussian_initializer()(m.weight)
                    if m.bias is not None:
                        constant_initializer()(m.bias)

    def forward(self, x):
        image = x[:, :3, :, :]
        x2 = F.relu(self.conv1_stage2(image))
        x2 = F.max_pool2d(x2, kernel_size=3, stride=2)
        x2 = F.relu(self.conv2_stage2(x2))
        x2 = F.max_pool2d(x2, kernel_size=3, stride=2)
        x2 = F.relu(self.conv3_stage2(x2))
        x3 = F.max_pool2d(x2, kernel_size=3, stride=2)
        x2 = F.relu(self.conv4_stage2(x3))
        return x2, x3
    


class Concatenated(nn.Module):
    def __init__(self, model1, model2, modelAVE):
        super(Concatenated, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.modelAVE = modelAVE
    
    def forward(self, x):
        image = x[:, :3, :, :]
        center_map = x[:, 3:, :, :]
        x1 = self.model1(image)
        x2 = self.model2(image)
        x_ave = self.modelAVE(center_map)
        concatenated_output = torch.cat([x1, x2, x_ave], dim=1)
        return concatenated_output

class ModelM(nn.Module):
    def __init__(self):
        super(ModelM, self).__init__()
        self.Mconv1 = nn.Conv2d(79, 128, kernel_size=11, padding=5)
        self.Mconv2 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mconv3 = nn.Conv2d(128, 128, kernel_size=11, padding=5)
        self.Mconv4 = nn.Conv2d(128, 128, kernel_size=1)
        self.Mconv5 = nn.Conv2d(128, 15, kernel_size=1)

    def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    gaussian_initializer()(m.weight)
                    if m.bias is not None:
                        constant_initializer()(m.bias)

    def forward(self, x):
        image = x[:, :3, :, :]
        x2 = F.relu(self.Mconv1_stage2(image))
        x2 = F.relu(self.Mconv2_stage2(x2))
        x2 = F.relu(self.Mconv3_stage2(x2))
        x2 = F.relu(self.Mconv4_stage2(x2))
        x2 = F.relu(self.Mconv5_stage2(x2))
        return x2

class Model3(nn.Module):
    def __init__(self):
        super(Model3, self).__init__()
        self.conv1_stage3 = nn.Conv2d(32, 32, kernel_size=5, padding=2)
        

    def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    gaussian_initializer()(m.weight)
                    if m.bias is not None:
                        constant_initializer()(m.bias)

    def forward(self, x):
        image = x[:, :3, :, :]
        x3 = F.relu(self.conv1_stage3(image))
        return x3



class Model(nn.Module):
    def __init__(self, model1, model2, model3, modelAVE, concatenated_stage2, concatenated_stage3, model2_M, model3_M):
        super(Model, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.modelAVE = modelAVE
        self.concatenated2 = concatenated_stage2
        self.concatenated3 = concatenated_stage3
        self.model2_M = model2_M
        self.model3_M = model3_M
        self.center_map = torch.nn.parameter.Parameter(np.random.randint(2, size=(368, 368, 20))) 
        

    def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    gaussian_initializer()(m.weight)
                    if m.bias is not None:
                        constant_initializer()(m.bias)
    
    def forward(self, x):
        output1 = self.model1(x)
        output2, input5 = self.model2(x)
        outputAVE = self.modelAVE(x)
        concatenated_output_stage2 = self.concatenated2(output1, output2, outputAVE)
        output3 = self.model2_M(concatenated_output_stage2)     
        output4 = self.model3(input5)
        concatenated_output_stage3 = self.concatenated3(output3, outputAVE, output4)
        output5 = self.model3_M
        concatenated_output = torch.cat([output1, output2, output3,concatenated_output_stage2, outputAVE, concatenated_output_stage2, concatenated_output_stage3, output4, output5], dim=1)
        return concatenated_output
        