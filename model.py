import torchvision.models as models
from torch import nn
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from params import params


class imageModel(nn.Module):
    def __init__(self):
        super(imageModel, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, 1)
    
    def forward(self, x):
        return self.model(x)
    
    def __str__(self):
        return str(self.model)
    
class ImageConvModel(nn.Module):
    def __init__(self):
        super(ImageConvModel, self).__init__()
        self.model = nn.Sequential()
        self.model.add_module('0', nn.Conv2d(2, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.model.add_module('0_1', nn.BatchNorm2d(64))
        self.model.add_module('1', nn.ReLU(inplace=True))
        self.model.add_module('2', nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False))
        self.model.add_module('2_1', nn.Dropout2d(0.2))
        self.model.add_module('3', nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.model.add_module('3_1', nn.BatchNorm2d(128))
        self.model.add_module('4', nn.ReLU(inplace=True))
        self.model.add_module('5', nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False))
        self.model.add_module('5_1', nn.Dropout2d(0.2))
        self.model.add_module('6', nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.model.add_module('6_1', nn.BatchNorm2d(256))
        self.model.add_module('7', nn.ReLU(inplace=True))
        self.model.add_module('8', nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False))
        self.model.add_module('8_1', nn.Dropout2d(0.2))
        self.model.add_module('9', nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
        self.model.add_module('9_1', nn.BatchNorm2d(512))
        self.model.add_module('10', nn.ReLU(inplace=True))
        self.model.add_module('10', nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False))
        self.model.add_module('10_1', nn.Dropout2d(0.2))
        self.model.add_module('11', nn.Flatten())

    def forward(self, x):
        return self.model(x)
    
    def __str__(self):
        return str(self.model)
    
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(722176, 32)
        self.lstm2 = nn.LSTM(32, 32)
        self.lstm3 = nn.LSTM(32, 32)

    def forward(self, x, prev_state=None):
        x, hc1 = self.lstm1(x, prev_state)
        x, hc2 = self.lstm2(x, hc1)
        x, hc3 = self.lstm3(x, hc2)
        return x, hc3

class ImuEncoder(nn.Module):
    def __init__(self):
        super(ImuEncoder, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.linear_1 = nn.Linear(6, 16)
        self.linear_2 = nn.Linear(16, 32)
        self.linear_3 = nn.Linear(32, 64)
        self.linear_4 = nn.Linear(64, 128)
        self.linear_5 = nn.Linear(128, 256)
    
    def forward(self, x):
        x = self.linear_1(x)
        x = self.relu(x)
        x = self.linear_2(x)
        x = self.relu(x)
        x = self.linear_3(x)
        x = self.relu(x)
        x = self.linear_4(x)
        x = self.relu(x)
        x = self.linear_5(x)
        x = self.relu(x)
        return x

class DeepVIO(nn.Module):
    def __init__(self):
        super(DeepVIO, self).__init__()
        self.imageModel = ImageConvModel()
        self.imuEncoder = ImuEncoder()
        self.lstmModel = LSTMModel()
        self.linear = nn.Linear(32, 7)

    
    def forward(self, x_images, x_imu, prev_state=None):
        batch_size = x_images.size(0)
        seq_len = x_images.size(1)

        # Reshape for image model
        images_model_input = x_images.view(batch_size*seq_len, 2, params.img_h, params.img_w)
        x = self.imageModel(images_model_input)
        x_imu = self.imuEncoder(x_imu)
        x = x.view(batch_size, seq_len, -1)
        # print(x.shape)
        # Reorder the dimensions to [seq_len, batch_size, 1, -1]
        x = x.permute(1, 0, 2)
        x_imu = x_imu.permute(1, 0, 2)
        # Reshape the tensor to [seq_len, batch_size, -1]
        x = x.contiguous().view(seq_len, batch_size, -1)
        # print(x.shape, x_imu.shape)

        x = torch.concat([x, x_imu], axis=2)
        # print(x.shape)
        # RNN
        out, hc = self.lstmModel(x, prev_state)
        # print(out.shape)
        pose = self.linear(out)
        pose = pose.permute(1, 0, 2)
        # print("pose", pose.shape)

        angles = pose[:, :, 3:]
        translation = pose[:, :, :3]
        return angles, translation, hc
    
    def get_loss(self, x_images, x_imu, y, prev_state=None):
        angles, translation, _ = self.forward(x_images, x_imu, prev_state)
        y = y.to(params.device)
        # print("angles", angles.shape, "translation", translation.shape, "y", y.shape)
        angle_loss = torch.nn.functional.mse_loss(angles, y[:,:,3:])
        translation_loss = torch.nn.functional.mse_loss(translation, y[:,:,:3])
        loss = params.angular_loss_weight * angle_loss + translation_loss
        return loss, angle_loss, translation_loss

    def step(self, x_images, x_imu, y, optimizer, prev_state=None):
        optimizer.zero_grad()
        loss, angle_loss, translation_loss = self.get_loss(x_images, x_imu, y, prev_state)
        loss.backward()
        optimizer.step()
        return loss, angle_loss, translation_loss    
    

    def train_model(self, data_loader, optimizer):
        self.train()
        loss_ang_mean_train = 0
        loss_trans_mean_train = 0
        losses = {'total': [], 'angle': [], 'translation': []}
        iter_num = 0
        for x_images, x_imu, y in tqdm(data_loader):
            x_images = x_images.to(params.device)
            x_imu = x_imu.to(params.device)
            y = y.to(params.device)
            loss, angle_loss, translation_loss = self.step(x_images, x_imu, y, optimizer)
            losses['total'].append(loss.item())
            losses['angle'].append(angle_loss.item())
            losses['translation'].append(translation_loss.item())
            iter_num += 1
            if iter_num % 20 == 0:
                message = f'Iteration: {iter_num}, Loss: {loss:.3f}, angle: {100*angle_loss:.4f}, trans: {translation_loss:.3f}'
                f = open(params.record_path, 'a')
                f.write(message+'\n') 
                print(message)
                f.close()
        loss_ang_mean_train /= len(data_loader)
        loss_trans_mean_train /= len(data_loader)
        loss_mean_train = loss_ang_mean_train + loss_trans_mean_train

        return loss_mean_train, loss_ang_mean_train, loss_trans_mean_train

    def validate_model(self, data_loader):
        self.eval()
        loss_ang_mean_val = 0
        loss_trans_mean_val = 0
        losses = {'total': [], 'angle': [], 'translation': []}

        for x_images, x_imu, y in tqdm(data_loader):
            x_images = x_images.to(params.device)
            x_imu = x_imu.to(params.device)
            y = y.to(params.device)
            loss, angle_loss, translation_loss = self.get_loss(x_images, x_imu, y)
            losses['total'].append(loss.item())
            losses['angle'].append(angle_loss.item())
            losses['translation'].append(translation_loss.item())
            loss_ang_mean_val += angle_loss.item()
            loss_trans_mean_val += translation_loss.item()
        loss_ang_mean_val /= len(data_loader)
        loss_trans_mean_val /= len(data_loader)
        loss_mean_val = loss_ang_mean_val + loss_trans_mean_val
        return loss_mean_val, loss_ang_mean_val, loss_trans_mean_val
    
    def __str__(self):
        return str(self.imageModel) + '\n' + str(self.lstmModel)
    

    
    
if __name__ == '__main__':
    model = DeepVIO()
    print(model)
