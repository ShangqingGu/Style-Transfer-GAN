import random
import time
import datetime
import sys

from torch.autograd import Variable
from torchvision.utils import save_image

import torch
import numpy as np


class ReplayBuffer():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0,1) > 0.5:
                    i = random.randint(0, self.max_size-1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))

class LambdaLR():
    def __init__(self, epoch_num, epoch_start, decay_start_epoch):
        assert ((epoch_num - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.epoch_num = epoch_num
        self.epoch_start = epoch_start
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + 1 + self.epoch_start - self.decay_start_epoch)/(self.epoch_num - self.decay_start_epoch)


def sample_images(args,G_AB,G_BA, test_dataloader, epoch, batches_done):
    imgs = next(iter(test_dataloader))
    real_X_A = Variable(imgs['X'].type(torch.FloatTensor).cuda())
    real_Y_B = Variable(imgs['Y'].type(torch.FloatTensor).cuda())

    fake_X_B = G_AB(real_X_A) 
    recov_X_A = G_BA(fake_X_B)
    idt_Y_B = G_AB(real_Y_B)

    fake_Y_A = G_BA(real_Y_B)
    recov_Y_B = G_AB(fake_Y_A)
    idt_X_A = G_BA(real_X_A)

    img_sample = torch.cat((real_X_A.data ,fake_X_B.data,recov_X_A.data,idt_Y_B.data,
                            real_Y_B.data ,fake_Y_A.data,recov_Y_B.data,idt_X_A.data), 0)
    save_image(img_sample, './%s-%s/%s/%s-%s.png' % (args.exp_name, args.dataset_name, args.img_result_dir, batches_done, epoch), nrow=4, normalize=True)
