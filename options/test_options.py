from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--ntest', type=int, default=float('inf'), help='number of test examples')
        self.parser.add_argument('--results_dir', type=str, default='./results/', help='results are stored in this directory')
        self.parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        self.parser.add_argument('--phase', type=str, default='options: train; val; test; etc.' )
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load')
        self.parser.add_argument('--nimage', type=int, default=50, help='number of test images to run')
        self.isTrain = False