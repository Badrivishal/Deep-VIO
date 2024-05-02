import torch
import pandas as pd
import matplotlib.pyplot as plt
from model import DeepVIO
from dataloader import convert_to_tensor
from params import params
from tqdm import tqdm
import numpy as np

def test_model(model, data_loader):
    model.eval()
    total_position_error = 0
    total_quaternion_error = 0
    num_samples = 0
    print("Model is running")
    # ground_truth_positions = []
    # estimated_positions = []
    ground_truth_positions = np.empty((0, 3))
    estimated_positions = np.empty((0, 3))

    # Iterate through the test data loader
    with torch.no_grad():
        for x_images, x_imu, y in tqdm(data_loader):
            # Move data to GPU if available
            x_images = x_images.to(params.device)
            x_imu = x_imu.to(params.device)
            y = y.to(params.device)

            # Forward pass to get model predictions
            angles_pred, translation_pred, _ = model.forward(x_images, x_imu)


            position_diff = translation_pred - y[:, :, :3]
            position_error = torch.norm(position_diff, dim=2).mean()
            total_position_error += position_error.item()

            quaternion_diff = angles_pred - y[:, :, 3:]
            quaternion_error = torch.atan2(torch.norm(quaternion_diff, dim=2), 1.0 - torch.norm(quaternion_diff, dim=2)).mean()
            total_quaternion_error += quaternion_error.item()

            num_samples += x_images.size(0)

            y_np = y[:, :, :3].view(-1, 3).cpu().numpy()
            translation_pred_np = translation_pred.reshape(-1, 3).cpu().numpy()

            ground_truth_positions = np.concatenate((ground_truth_positions, y_np), axis=0)
            estimated_positions = np.concatenate((estimated_positions, translation_pred_np), axis=0)


            # Save ground truth and estimated positions for plotting
            # ground_truth_positions = np.concatenate((ground_truth_positions, y[:, :, :3].cpu().numpy()), axis=0)
            # estimated_positions = np.concatenate((estimated_positions, translation_pred.cpu().numpy()), axis=0)

    # Calculate average positional error and quaternion error
    avg_position_error = total_position_error / num_samples
    avg_quaternion_error = total_quaternion_error / num_samples

    return avg_position_error, avg_quaternion_error, ground_truth_positions, estimated_positions


if __name__ == "__main__":
    # Load model
    model = DeepVIO()
    model = model.to(params.device)
    model.load_state_dict(torch.load("D:\\1st Sem WPI\\Deep Learning\\Final Project\\Deep VIO\\experiments\\experiment_2_lr1e-4_decay5e-4_0.4drouput\\models\\DeepVIO.model_epoch_6.model.pth"))

    print("Model loaded")
    # Prepare test dataset
    df_meta = pd.read_csv('dataset/mh_01.csv')  # Update the path if necessary
    float64_cols = list(df_meta.select_dtypes(include='float64'))
    df_meta[float64_cols] = df_meta[float64_cols].astype('float32')

    test_dataset = convert_to_tensor(df=df_meta, file_path='dataset/mh_01/', seq_len=params.seq_len)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=params.batch_size, shuffle=False)

    # Test model
    position_error, quaternion_error, ground_truth_positions, estimated_positions = test_model(model, test_loader)
    print(f"Average Positional Error: {position_error}")
    print(f"Average Quaternion Error: {quaternion_error}")

    print("Plotting ground truth and estimated positions")

    # Define the planes for plotting
    planes = [('X', 'Y', 0, 1), ('Y', 'Z', 1, 2), ('Z', 'X', 2, 0)]

    # Plot ground truth and estimated positions over time
    for plane, (axis1, axis2, idx1, idx2) in enumerate(planes):
        plt.figure(figsize=(10, 10))
        plt.scatter(ground_truth_positions[:, idx1], ground_truth_positions[:, idx2], label='Ground Truth', alpha=0.5)
        plt.scatter(estimated_positions[:, idx1], estimated_positions[:, idx2], label='Estimation', alpha=0.5)
        plt.xlabel(axis1)
        plt.ylabel(axis2)
        plt.title(f'{plane} Position Plot')
        plt.legend()
        plt.grid(True)
        plt.show()