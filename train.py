from model import DeepVIO
from params import params
from torch.utils.data import DataLoader
from dataloader import RandomNoiseDataset
import torch

def get_data_loaders(batch_size):
    dataset = RandomNoiseDataset(num_samples=1000, seq_len=5)
    val_dataset = RandomNoiseDataset(num_samples=200, seq_len=5)
    
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    return train_dataloader, val_dataloader


if __name__ == "__main__":
    torch.cuda.empty_cache()

    train_dataloader, val_dataloader = get_data_loaders(params.batch_size)


    m_deepvio = DeepVIO()
    m_deepvio = m_deepvio.to(params.device)

    optimizer = torch.optim.Adam(m_deepvio.parameters(), lr=params.optim_lr, betas=(0.9, 0.999), weight_decay=params.optim_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=params.optim_lr_step, gamma=params.optim_lr_decay_factor)


    if params.resume:
        m_deepvio.load_state_dict(torch.load(params.load_model_path))
        optimizer.load_state_dict(torch.load(params.load_optimizer_path))
        print('Model loaded from {}'.format(params.load_model_path))
        print('Optimizer loaded from {}'.format(params.load_optimizer_path))

    print('Training model...')

    min_loss_train = 1e10
    min_loss_val = 1e10

    for epoch in range(params.epochs):
        print('Epoch: {}'.format(epoch))
        train_loss, train_loss_ang, train_loss = m_deepvio.train_model(train_dataloader, optimizer)
        val_loss, val_loss_ang, val_loss = m_deepvio.validate_model(val_dataloader)
        scheduler.step()
        print('Train Loss: {:.3f}, Angle Loss: {:.4f}, Translation Loss: {:.3f}'.format(train_loss, train_loss_ang, train_loss))

        print("Saving model...")
        torch.save(m_deepvio.state_dict(), params.load_model_path, _use_new_zipfile_serialization=True)
        torch.save(optimizer.state_dict(), params.load_optimizer_path, _use_new_zipfile_serialization=True)


