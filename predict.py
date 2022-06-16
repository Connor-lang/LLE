from options.test_options import TestOptions
from data.dataloader import CreateDataLoader
from models.enlighten_gan import EnlightenGANModel
from pdb import set_trace as st
import util.util as util

opt = TestOptions().parse()
opt.nThreads = 1
opt.batchSize = 1
opt.serial_batches = True
opt.no_flip = True

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = EnlightenGANModel()
print(len(dataset))
for i, data in enumerate(dataset):
    model.set_input(data)
    visuals = model.predict()
    img_path = model.get_image_paths()
    util.save_image(visuals, img_path)