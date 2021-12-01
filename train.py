import math
import torch
import torch.nn as nn
import numpy as np
import time
import matplotlib
from torchvision import transforms
from torch.autograd import grad

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from glob import glob
from dataset import *
from network import *


class Train:
    def __init__(self, train_path='data/train', result_path='result', from_signal_activate=True, mixing=False,
                 init_signal_size=8,
                 max_signal_size=4096,
                 code_size=512,
                 num_epochs=500000, phase=30000,
                 batch_size=4, light=True,
                 input_nc=1, output_nc=1, lr=1e-4, weight_decay=1e-4, decay_flag=True, device='cuda:0',
                 resume=False):
        self.train_path = train_path
        self.result_path = result_path
        self.from_signal_activate = from_signal_activate
        self.mixing = mixing
        self.init_signal_size = init_signal_size
        self.max_signal_size = max_signal_size
        self.code_size = code_size
        self.num_epochs = num_epochs
        self.phase = phase
        self.batch_size = batch_size
        self.light = light
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.lr = lr
        self.weight_decay = weight_decay
        self.decay_flag = decay_flag
        self.device = device
        self.resume = resume

        self.used_batch = 0
        self.resolution = 0

    def get_train_data(self):
        Traindata = GetData(self.train_path)
        train_data = Traindata.get_data()
        return train_data

    def dataload(self, train_data, signal_size):
        train_transform = transforms.Compose([
            #             transforms.ToTensor(),
            transforms.Resize([1, signal_size]),
            #             transforms.Normalize(mean=0.5, std=0.5)
        ])
        self.train_loader = DataLoader(GetDataset(train_data, transform=train_transform),
                                       batch_size=self.batch_size, shuffle=True)

    def build_model(self):
        self.generator = StyledGenerator(code_dim=self.code_size, n_mlp=8).to(self.device)
        self.discriminator = Discriminator(fused=True, from_signal_activate=self.from_signal_activate).to(self.device)

    def define_optim(self):
        self.g_optim = torch.optim.Adam(
            self.generator.parameters(), lr=self.lr, weight_decay=self.weight_decay, betas=(0.0, 0.99)
        )
        self.d_optim = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                                        betas=(0.0, 0.99))

    def save_model(self, path, step):
        folder = os.path.exists(path)
        if not folder:
            os.makedirs(path)
        params = {}
        params['generator'] = self.generator.state_dict()
        params['discriminator'] = self.discriminator.state_dict()
        params['g_optim'] = self.g_optim.state_dict()
        params['d_optim'] = self.d_optim.state_dict()
        params['used_batch'] = self.used_batch
        params['resolution'] = self.resolution
        torch.save(params, os.path.join(path, 'model_params_%08d.pt' % step))

    def load_model(self, path, step):
        params = torch.load(os.path.join(path, 'model_params_%08d.pt' % step))
        self.generator.load_state_dict(params['generator'])
        self.discriminator.load_state_dict(params['discriminator'])
        self.g_optim.load_state_dict(params['g_optim'])
        self.d_optim.load_state_dict(params['d_optim'])
        self.used_batch = params['used_batch']
        self.resolution = params['resolution']

    def train(self):
        # init
        self.generator.train(), self.discriminator.train()
        self.generator.requires_grad_(False)
        self.discriminator.requires_grad_(True)

        d_loss, g_loss, grad_loss = 0, 0, 0
        start_num = 1
        step = int(math.log2(self.init_signal_size)) - 2
        max_step = int(math.log2(self.max_signal_size)) - 2
        self.used_batch = 0
        self.resolution = 4 * 2 ** step
        final_progress = False
        train_data = self.get_train_data()

        # load model
        if self.resume:
            model_list = glob(os.path.join(self.result_path, 'model', '*.pt'))
            if len(model_list) != 0:
                model_list.sort()
                start_num = int(model_list[-1].split('_')[-1].split('.')[0])
                self.load_model(os.path.join(self.result_path, 'model'), start_num)
                step = int(math.log2(self.resolution)) - 2
                print("load success!")

        start_time = time.time()

        for num in range(start_num, self.num_epochs + 1):
            # train
            self.discriminator.zero_grad()
            alpha = min(1, 1 / self.phase * (self.used_batch + 1))

            if (self.resolution == self.init_signal_size and not self.resume) or final_progress:
                alpha = 1

            if self.used_batch > self.phase * 2:
                self.used_batch = 0
                step += 1

                if step > max_step:
                    step = max_step
                    final_progress = True
                else:
                    alpha = 0

                self.resolution = 4 * 2 ** step
                self.dataload(train_data=train_data, signal_size=self.resolution)
                train_iter = iter(self.train_loader)

            try:
                real_signal = train_iter.next()
            except:
                self.dataload(train_data=train_data, signal_size=self.resolution)
                train_iter = iter(self.train_loader)
                real_signal = train_iter.next()

            self.used_batch += real_signal.shape[0]
            real_signal = real_signal.to(dtype=torch.float, device=self.device)

            real_predict = self.discriminator(real_signal, step=step, alpha=alpha)
            real_predict = real_predict.mean() - 0.001 * (real_predict ** 2).mean()
            (-real_predict).backward()

            if self.mixing and random.random() < 0.9:
                gen_in11, gen_in12, gen_in21, gen_in22 = torch.randn(
                    4, self.batch_size, self.code_size, device=self.device
                ).chunk(4, 0)
                gen_in1 = [gen_in11.squeeze(0), gen_in12.squeeze(0)]
                gen_in2 = [gen_in21.squeeze(0), gen_in22.squeeze(0)]

            else:
                gen_in1, gen_in2 = torch.randn(2, self.batch_size, self.code_size, device=self.device).chunk(
                    2, 0
                )
                gen_in1 = gen_in1.squeeze(0)
                gen_in2 = gen_in2.squeeze(0)

            fake_signal = self.generator(gen_in1, step=step, alpha=alpha)
            fake_predict = self.discriminator(fake_signal, step=step, alpha=alpha)

            fake_predict = fake_predict.mean()
            fake_predict.backward()

            eps = torch.rand(self.batch_size, 1, 1).to(self.device)
            x_hat = eps * real_signal.data + (1 - eps) * fake_signal.data
            x_hat.requires_grad = True
            hat_predict = self.discriminator(x_hat, step=step, alpha=alpha)
            grad_x_hat = grad(
                outputs=hat_predict.sum(), inputs=x_hat, create_graph=True
            )[0]
            grad_penalty = (
                    (grad_x_hat.view(grad_x_hat.size(0), -1).norm(2, dim=1) - 1) ** 2
            ).mean()
            grad_penalty = 10 * grad_penalty
            grad_penalty.backward()
            if num % 10 == 0:
                grad_loss = grad_penalty.item()
                d_loss = (-real_predict + fake_predict).item()

            self.d_optim.step()

            self.generator.zero_grad()
            self.generator.requires_grad_(True)
            self.discriminator.requires_grad_(False)

            fake_signal = self.generator(gen_in2, step=step, alpha=alpha)

            predict = self.discriminator(fake_signal, step=step, alpha=alpha)

            loss = -predict.mean()

            if num % 10 == 0:
                g_loss = loss.item()

            loss.backward()
            self.g_optim.step()

            self.generator.requires_grad_(False)
            self.discriminator.requires_grad_(True)

            if num % 10 == 0:
                print("[%5d/%5d] time: %4.4f signal_size: %d d_loss: %.8f, g_loss: %.8f, grad_loss: %.8f" % (
                    num, self.num_epochs, time.time() - start_time, self.resolution, d_loss, g_loss, grad_loss))

            if num % 5000 == 0:
                train_sample_num = 5
                self.discriminator.eval(), self.generator.eval()
                test_gen_in = torch.randn(train_sample_num, 512).to(self.device)
                test_fake_signal = self.generator(test_gen_in, step=step, alpha=alpha)

                path = os.path.join(os.path.join(self.result_path, 'train'), str(num) + '-' + str(self.resolution))
                folder = os.path.exists(path)
                if not folder:
                    os.makedirs(path)
                plt.cla()
                np.savetxt(os.path.join(path, str(num) + '-' + str(self.resolution) + '.txt'),
                           test_fake_signal[0][0].cpu().detach().numpy())
                plt.plot(test_fake_signal[0][0].cpu().detach().numpy())
                plt.savefig(os.path.join(path, str(num) + '-' + str(self.resolution) + '.png'), dpi=600)

            self.discriminator.train(), self.generator.train()

            if num % 50000 == 0:
                self.save_model(os.path.join(self.result_path, 'model'), num)

            if num % 1000 == 0:
                params = {}
                params['generator'] = self.generator.state_dict()
                params['discriminator'] = self.discriminator.state_dict()
                params['g_optim'] = self.g_optim.state_dict()
                params['d_optim'] = self.d_optim.state_dict()
                params['used_batch'] = self.used_batch
                params['resolution'] = self.resolution
                torch.save(params, os.path.join('model_params_latest.pt'))


if __name__ == '__main__':
    gan = Train(train_path='data/train', result_path='result', from_signal_activate=False, mixing=False,
                init_signal_size=8,
                max_signal_size=4096,
                code_size=512,
                num_epochs=1000000, phase=2000000,
                batch_size=100, light=True,
                input_nc=1, output_nc=1, lr=1e-4, weight_decay=1e-4, decay_flag=True, device='cuda:2',
                resume=True)
    gan.build_model()
    gan.define_optim()
    gan.train()
    print("training finished!")