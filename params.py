import os
import torch

class Parameters():
    def __init__(self):

        # if torch.backends.mps.is_available():
        #     self.device = torch.device('mps')
        # else:
        #     self.device =torch.device('cpu')
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print("Using GPU")
        else:
            self.device = torch.device('cpu')
            print("Using CPU")

        # Path to the dataset. Please modify this before running
        #TODO Update data directories
        self.data_dir = '/home/mingyuy/DeepVO/dataset'
        self.image_dir = self.data_dir + '/sequences/'
        self.pose_dir = self.data_dir + '/poses/'
        
        # # List of train paths, valid paths and test paths
        # self.train_video = ['00', '02', '08', '09']
        # self.valid_video = ['03', '05']
        # self.test_video = ['03', '04', '05', '06', '07', '10']

        self.seq_len = 3           # Image sequence length
        self.overlap = 1           # overlap between adjacent sampled image sequences

        # Data Preprocessing
        self.img_w = 752   # 
        self.img_h = 960   # 

        # Data Augmentation (horizontal flip)
        self.is_hflip = False
        
        # Neural network settings
        self.rnn_input_size = 90246
        self.rnn_hidden_size = 256
        self.conv_dropout = (0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.5)
        self.rnn_dropout_out = 0.2
        self.rnn_dropout_between = 0.2 
        self.batch_norm = True
        self.angular_loss_weight = 1

        # Training settings
        self.epochs = 130
        self.batch_size = 5
        self.pin_mem = True
        self.optim_lr = 1e-4
        self.optim_decay = 5e-3
        self.optim_lr_decay_factor = 0.1
        self.optim_lr_step = 60

        # Pretrain, Resume training
        self.resume = False
        self.resume_t_or_v = '.latest'
        
        # Paths to save and load the model
        self.experiment_name = 'experiment_colab_new'
        self.save_path = 'experiments/{}'.format(self.experiment_name)

        # self.name = 't{}_v{}_im{}x{}_s{}_b{}'.format(''.join(self.train_video), ''.join(self.test_video), self.img_h, self.img_w, self.seq_len, self.batch_size)
        # self.name += '_flip' if self.is_hflip else ''
        self.name = "DeepVIO"

        # self.load_model_path = '{}/models/{}.model{}'.format(self.save_path, self.name, self.resume_t_or_v)
        self.save_model_path = '{}/models/{}'.format(self.save_path, self.name)
        self.load_model_path = "D:/1st Sem WPI/Deep Learning/Final Project/Working model/DeepVIO.model.latest"
        self.load_optimizer_path = 'D:/1st Sem WPI/Deep Learning/Final Project/Working model/DeepVIO.optimizer.latest'
        self.record_path = '{}/records/{}.txt'.format(self.save_path, self.name)
        self.save_model_path = '{}/models/{}.model'.format(self.save_path, self.name)
        self.save_optimzer_path = '{}/models/{}.optimizer'.format(self.save_path, self.name)

        if not os.path.isdir(os.path.dirname(self.record_path)):
            os.makedirs(os.path.dirname(self.record_path))
        if not os.path.isdir(os.path.dirname(self.save_model_path)):
            os.makedirs(os.path.dirname(self.save_model_path))
        if not os.path.isdir(os.path.dirname(self.save_optimzer_path)):
            os.makedirs(os.path.dirname(self.save_optimzer_path))


params = Parameters()