import torch
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional
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
        num_batch = 0
        for x_images, x_imu, y in tqdm(data_loader):
            # Move data to GPU if available
            x_images = x_images.to(params.device)
            x_imu = x_imu.to(params.device)
            y = y.to(params.device)

            # Forward pass to get model predictions
            angles_pred, translation_pred, _ = model.forward(x_images, x_imu)


            position_error = torch.nn.functional.mse_loss(translation_pred, y[:, :, :3])
            total_position_error += position_error.item()

            quaternion_diff = angles_pred - y[:, :, 3:]
            quaternion_error = torch.atan2(torch.norm(quaternion_diff, dim=2), 1.0 - torch.norm(quaternion_diff, dim=2)).mean()
            total_quaternion_error += quaternion_error.item()

            num_samples += x_images.size(0)
            num_batch += 1

            y_np = y[:, :, :3].view(-1, 3).cpu().numpy()
            translation_pred_np = translation_pred.reshape(-1, 3).cpu().numpy()

            ground_truth_positions = np.concatenate((ground_truth_positions, y_np), axis=0)
            estimated_positions = np.concatenate((estimated_positions, translation_pred_np), axis=0)


            # Save ground truth and estimated positions for plotting
            # ground_truth_positions = np.concatenate((ground_truth_positions, y[:, :, :3].cpu().numpy()), axis=0)
            # estimated_positions = np.concatenate((estimated_positions, translation_pred.cpu().numpy()), axis=0)

    # Calculate average positional error and quaternion error
    avg_position_error = total_position_error / num_batch
    avg_quaternion_error = total_quaternion_error / num_samples

    return avg_position_error, avg_quaternion_error, ground_truth_positions, estimated_positions


if __name__ == "__main__":
    # Load model
    model = DeepVIO()
    model = model.to(params.device)
    model.load_state_dict(torch.load("D:\\1st Sem WPI\\Deep Learning\\Final Project\\Deep VIO\\experiments\\experiment_small_imu_encoder_local2\\models\\DeepVIO.model_epoch_0.model.pth"))

    print("Model loaded")
    # Prepare test dataset
    df_meta = pd.read_csv('dataset/mh_01.csv')  # Update the path if necessary
    float64_cols = list(df_meta.select_dtypes(include='float64'))
    df_meta[float64_cols] = df_meta[float64_cols].astype('float32')

    test_dataset = convert_to_tensor(df=df_meta, file_path='dataset/mh_01/', seq_len=params.seq_len)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=params.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Test model
    position_error, quaternion_error, ground_truth, predictions = test_model(model, test_loader)
    print(f"Average Positional Error: {position_error}")
    print(f"Average Quaternion Error: {quaternion_error}")

    print("Plotting ground truth and estimated positions")
    print('Mean Squared Error:', np.mean((ground_truth - predictions) ** 2))

    # Define the planes for plotting
    # XY plot
    plt.figure()
    plt.scatter(ground_truth[:, 0], ground_truth[:, 1], label='Ground Truth', marker='.')
    plt.scatter(predictions[:, 0], predictions[:, 1], label='Predictions', marker='.')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('XY Plot')
    plt.legend()
    plt.show()

    # YZ plot
    plt.figure()
    plt.scatter(ground_truth[:, 1], ground_truth[:, 2], label='Ground Truth', marker='.')
    plt.scatter(predictions[:, 1], predictions[:, 2], label='Predictions', marker='.')
    plt.xlabel('Y')
    plt.ylabel('Z')
    plt.title('YZ Plot')
    plt.legend()
    plt.show()

    # ZX plot
    plt.figure()
    plt.scatter(ground_truth[:, 2], ground_truth[:, 0], label='Ground Truth', marker='.')
    plt.scatter(predictions[:, 2], predictions[:, 0], label='Predictions', marker='.')
    plt.xlabel('Z')
    plt.ylabel('X')
    plt.title('ZX Plot')
    plt.legend()
    plt.show()