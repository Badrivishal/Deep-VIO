from model import DeepVIO
from params import params
from torch.utils.data import DataLoader
from dataloader import convert_to_tensor
import torch
import pandas as pd


def get_data_loaders_split(df_meta, image_folder, seq_len, batch_size):
    split_idx = int(0.8 * df_meta.shape[0])

    df_train = df_meta.iloc[:split_idx]
    df_valid = df_meta.iloc[split_idx:]
    print('train set:', df_train.shape[0])
    print('valid set:', df_valid.shape[0])

    train_dataset = convert_to_tensor(df=df_train,
                                      file_path=image_folder,
                                      seq_len=seq_len)
    valid_dataset = convert_to_tensor(df=df_valid,
                                      file_path=image_folder,
                                      seq_len=seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader


def get_data_loader_whole_set(df_meta, image_folder, seq_len, batch_size):
    dataset = convert_to_tensor(df=df_meta,
                                file_path=image_folder,
                                seq_len=seq_len)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return loader


if __name__ == "__main__":

    torch.cuda.empty_cache()

    seq_len = 10
    batch_size = 6

    # prepare meta data -- TRAIN SET: MH01
    df_meta_train = pd.read_csv('dataset/mh_01.csv')
    float64_cols = list(df_meta_train.select_dtypes(include='float64'))
    df_meta_train[float64_cols] = df_meta_train[float64_cols].astype('float32')
    train_image_folder = 'dataset/mh_01/'

    train_dataloader = get_data_loader_whole_set(df_meta=df_meta_train,
                                                 image_folder=train_image_folder,
                                                 seq_len=seq_len,
                                                 batch_size=batch_size)

    # prepare meta data -- VALIDATION SET: MH02
    df_meta_valid = pd.read_csv('dataset/mh_02.csv')
    float64_cols = list(df_meta_valid.select_dtypes(include='float64'))
    df_meta_valid[float64_cols] = df_meta_valid[float64_cols].astype('float32')
    valid_image_folder = 'dataset/mh_02/'

    valid_dataloader = get_data_loader_whole_set(df_meta=df_meta_valid,
                                                 image_folder=valid_image_folder,
                                                 seq_len=seq_len,
                                                 batch_size=batch_size)


    m_deepvio = DeepVIO()
    m_deepvio = m_deepvio.to(params.device)

    optimizer = torch.optim.Adam(m_deepvio.parameters(), lr=params.optim_lr, betas=(0.9, 0.999),
                                 weight_decay=params.optim_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=params.optim_lr_step,
                                                gamma=params.optim_lr_decay_factor)

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
        train_loss, train_loss_ang, train_trans_loss = m_deepvio.train_model(train_dataloader, optimizer)
        val_loss, val_loss_ang, val_trans_loss = m_deepvio.validate_model(valid_dataloader)
        scheduler.step()
        message = "Epoch: {}, Train Loss: {:.3f}, Angle Loss: {:.4f}, Translation Loss: {:.3f}, Validation Loss: {:.3f}, Angle Loss: {:.4f}, Translation Loss: {:.3f}".format(
            epoch, train_loss, train_loss_ang, train_trans_loss, val_loss, val_loss_ang, val_trans_loss)
        f = open(params.record_path, 'a')
        f.write(message + '\n')
        print(message)
        f.close()
        # print('Train Loss: {:.3f}, Angle Loss: {:.4f}, Translation Loss: {:.3f}'.format(train_loss, train_loss_ang, train_loss))
        # print("Validation Loss: {:.3f}, Angle Loss: {:.4f}, Translation Loss: {:.3f}".format(val_loss, val_loss_ang, val_loss))

        print("Saving model...")
        torch.save(m_deepvio.state_dict(), params.save_model_path + "_epoch_{}".format(epoch) + ".model.pth")
        torch.save(optimizer.state_dict(), params.save_model_path + "_epoch_{}".format(epoch) + ".optimizer.pth")
