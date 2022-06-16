import logging
import torch 
import time 
import os
import datetime
from data.dataloader import CreateDataLoader
from models.enlighten_gan import EnlightenGANModel
from options.train_options import TrainOptions

if __name__ == "__main__":

    dated_filename = "./logs//enlightenGAN/" + str(datetime.date.today()) + ".log"
    os.makedirs(os.path.dirname(dated_filename), exist_ok=True)

    
    logging.basicConfig(
        filemode='w', 
        level=logging.INFO,
        filename=dated_filename,
        format="%(levelname)s: %(asctime)s - %(message)s"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using: {device}")
    logging.info(f"Using: {device}")

    opt = TrainOptions().parse()
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('Number of training images = %d' % dataset_size)
    logging.info(f"Number of training images = {dataset_size}")

    model = EnlightenGANModel()

    total_steps = 0

    for epoch in range(1, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        print("Epoch start")

        for i, data in enumerate(dataset):
            print("Enumerating dataset")
            total_steps += opt.batchSize
            epoch_iter = total_steps - dataset_size * (epoch - 1)
            model.set_input(data)
            model.optimize_parameters(epoch)
            
            if total_steps % opt.print_freq == 0:
                errors = model.get_current_errors(epoch)
                t = time.time() - epoch_start_time
                logging.info('----------Epoch: %d, Iter: %d, time: %.3f----------' % (epoch, epoch_iter, t))
                for k, v in errors.items():
                    logging.info('%s: %.3f' % (k, v))
            
            if total_steps % opt.save_latest_freq == 0:
                logging.info('saving the latest model (epoch: %d, total_steps: %d)' % (epoch, total_steps))
                model.save('latest')
            
        if epoch % opt.save_epoch_freq == 0:
            logging.info('saving the latest model (epoch: %d, total_steps: %d)' % (epoch, total_steps))
            model.save('latest')
            model.save(epoch)
        
        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

        if opt.new_lr:
            if epoch == opt.niter:
                model.update_learning_rate()
            elif epoch == (opt.niter + 20):
                model.update_learning_rate()
            elif epoch == (opt.niter + 70):
                model.update_learning_rate()
            elif epoch == (opt.niter + 90):
                model.update_learning_rate()
                model.update_learning_rate()
                model.update_learning_rate()
                model.update_learning_rate()
        else:
            if epoch > opt.niter:
                model.update_learning_rate()