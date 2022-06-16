from signal import default_int_handler
from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--print-freq', type=int, default=100, help='frequency of showing training results')
        self.parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--continue_train', action='store_true', help='continue train: load the latest model')
        self.parser.add_argument('--phase', type=str, default='train', help='options: train; val; test; etc.')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load. set to latest to use the latest cached model')
        self.parser.add_argument('--niter', type=int, default=100, help='number of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=100, help='number of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
        self.parser.add_argument('--no_lsgan', action='store_true', help='do not use least square GAN, use vanilla GAN')
        self.parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated image')
        self.isTrain = True