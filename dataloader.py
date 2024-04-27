import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from torchvision import transforms


class RandomNoiseDataset(Dataset):
    def __init__(self, num_samples, seq_len):
        self.num_samples = num_samples
        self.seq_len = seq_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x_image = torch.randn(self.seq_len, 1, 960, 752)
        x_imu = torch.randn(self.seq_len, 6)
        y = torch.randn(self.seq_len, 7)
        return x_image, x_imu, y


class EurocDataset(Dataset):
    def __init__(self, file_path, df_metadata, seq_len, transform=None):
        self.file_path = file_path
        self.df_metadata = df_metadata
        self.transform = transform
        self.seq_len = seq_len

    def __len__(self):
        return self.df_metadata.shape[0] - self.seq_len

    def __getitem__(self, idx):
        image_tensors, imu_tensors, pose_tensors = [], [], []

        for i in range(self.seq_len):
            image_tensors.append(self._get_image_tensor(idx + i))
            imu_tensors.append(self._get_imu_tensor(idx + i))
            pose_tensors.append(self._get_pose_tensor(idx + i))

        combined_image_tensor = torch.stack(image_tensors, dim=0)
        combined_imu_tensor = torch.stack(imu_tensors, dim=0)
        combined_pose_tensor = torch.stack(pose_tensors, dim=0)

        return combined_image_tensor, combined_imu_tensor, combined_pose_tensor

    def _get_image_tensor(self, idx):
        file_name_cam0 = self.df_metadata.iloc[idx].cam0
        file_name_cam1 = self.df_metadata.iloc[idx].cam1

        image_cam0 = Image.open(self.file_path + 'cam0/' + file_name_cam0).convert("L")
        image_cam1 = Image.open(self.file_path + 'cam1/' + file_name_cam1).convert("L")

        tensor_cam0 = self.transform(image_cam0)
        tensor_cam1 = self.transform(image_cam1)

        concatenated_tensor = torch.cat((tensor_cam0, tensor_cam1), dim=1)
        # print('concatenated tensor shape:', concatenated_tensor.shape)

        return concatenated_tensor

    def _get_imu_tensor(self, idx):
        imu_list = self.df_metadata.iloc[idx][['w_RS_S_x [rad s^-1]', 'w_RS_S_y [rad s^-1]', 'w_RS_S_z [rad s^-1]',
                                               'a_RS_S_x [m s^-2]', 'a_RS_S_y [m s^-2]', 'a_RS_S_z [m s^-2]']].to_list()
        return torch.tensor(imu_list)

    def _get_pose_tensor(self, idx):
        pose_list = self.df_metadata.iloc[idx][[' p_RS_R_x [m]', ' p_RS_R_y [m]', ' p_RS_R_z [m]',
                                                ' q_RS_w []', ' q_RS_x []', ' q_RS_y []', ' q_RS_z []']].to_list()
        return torch.tensor(pose_list)


def convert_to_tensor(df, file_path, seq_len):
    dataset = EurocDataset(file_path=file_path,
                           df_metadata=df,
                           seq_len=seq_len,
                           transform=transforms.Compose([transforms.ToTensor(),
                                                         ]))
    return dataset


if __name__ == "__main__":
    df_meta = pd.read_csv('dataset/mh_01.csv')
    image_folder = 'dataset/mh_01/'
    seq_len = 3
    batch_size = 4

    train_dataset = convert_to_tensor(df=df_meta,
                                      file_path=image_folder,
                                      seq_len=seq_len)

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    for data in trainloader:
        x_image, x_imu, y = data
        print(f"image shape: {x_image.shape}")
        print(f"imu shape: {x_imu.shape}")
        print(f"pose shape: {y.shape}")
        break

    # Create dataset
    # dataset = RandomNoiseDataset(num_samples=100, seq_len=10)
    #
    # # Create dataloader
    # dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
    #
    # # Usage
    # for batch in dataloader:
    #     x_image, x_imu, y = batch
    #     print(f"x_image shape: {x_image.shape}")
    #     print(f"x_imu shape: {x_imu.shape}")
    #     print(f"y shape: {y.shape}")
    #     break
