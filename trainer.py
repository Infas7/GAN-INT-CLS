import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from txt2image_dataset import Text2ImageDataset
from models.gan_factory import gan_factory
from utils import Utils, Logger
from PIL import Image
import os

from inception_score import get_inception_score

from time import time
import shutil


device = 'cuda:1'


class Trainer(object):
    def __init__(self, type, dataset, split, lr, diter, save_path, l1_coef, l2_coef, pre_trained_gen, pre_trained_disc, batch_size, num_workers, epochs, eval_batch_size, eval_interval, ds):
        with open('config.yaml', 'r') as f:
            config = yaml.safe_load(f)

        self.generator = gan_factory.generator_factory(type).to(device)
        self.discriminator = gan_factory.discriminator_factory(
            type) .to(device)

        if pre_trained_disc:
            self.discriminator.load_state_dict(torch.load(pre_trained_disc))
        else:
            self.discriminator.apply(Utils.weights_init)

        if pre_trained_gen:
            self.generator.load_state_dict(torch.load(pre_trained_gen))
        else:
            self.generator.apply(Utils.weights_init)

        self.dataset_name = dataset
        self.ds = ds
        self.eval_interval = eval_interval
        self.eval_batch_size = eval_batch_size

        if self.dataset_name == 'birds':
            self.dataset = Text2ImageDataset(
                config['birds_dataset_path'], config['birds_dataset_path_full'], split=split)
            self.val_dataset = Text2ImageDataset(
                config['birds_dataset_path'], config['birds_dataset_path_full'], split=1)
            self.temp_dataset = Text2ImageDataset(
                'birds/birds.hdf5', config['birds_dataset_path_full'], split=0)
        elif self.dataset_name == 'flowers':
            self.dataset = Text2ImageDataset(
                config['flowers_dataset_path'], config['flowers_dataset_path_full'], split=split)
            self.val_dataset = Text2ImageDataset(
                config['flowers_dataset_path'], config['flowers_dataset_path_full'], split=1)
            self.temp_dataset = Text2ImageDataset(
                'flowers/flowers_50.hdf5', config['flowers_dataset_path_full'], split=0)
        else:
            print('Dataset not supported, please select either birds or flowers.')
            exit()

        self.noise_dim = 100
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.lr = lr
        self.beta1 = 0.5
        self.num_epochs = epochs
        self.DITER = diter

        self.l1_coef = l1_coef
        self.l2_coef = l2_coef

        print(len(self.dataset))

        self.logger = Logger()

        self.data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True,
                                      num_workers=self.num_workers, drop_last=True)

        self.val_data_loader = DataLoader(self.val_dataset, batch_size=self.eval_batch_size, shuffle=False,
                                          num_workers=self.num_workers)

        self.temp_dataloader = DataLoader(self.temp_dataset, batch_size=512, shuffle=True,
                                          num_workers=self.num_workers, drop_last=True)

        self.optimD = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.optimG = torch.optim.Adam(
            self.generator.parameters(), lr=self.lr, betas=(self.beta1, 0.999))

        self.checkpoints_path = 'checkpoints'
        self.save_path = save_path
        self.type = type

    def al_method(self):
        print('Before AL', len(self.dataset))
        temp_idxs = []

        noise = torch.randn(self.batch_size, 100, 1, 1).to(device)

        with torch.no_grad():
            for sample in self.temp_dataloader:
                right_embed = sample['right_embed'].float().to(device)

                fake_images = self.generator(right_embed, noise)

                outputs, _ = self.discriminator(
                    fake_images, right_embed)

                _, idxs = torch.topk(-outputs, k=1)  # 8
                temp_idxs += [sample['key'][x] for x in idxs.tolist()]

        self.dataset.dataset_keys = list(
            set(self.dataset.dataset_keys + temp_idxs))

        if self.dataset_name == 'birds':
            self.dataset = Text2ImageDataset(
                "birds/birds_10.hdf5", "birds/birds.hdf5", split=0, precompiled_keys=self.dataset.dataset_keys)
        else:
            self.dataset = Text2ImageDataset(
                "flowers/flowers_10.hdf5", "flowers/flowers.hdf5", split=0, precompiled_keys=self.dataset.dataset_keys)

        self.data_loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True,
                                      num_workers=self.num_workers, drop_last=True)
        print('After AL', len(self.dataset))

    def train(self, cls=False):

        # if self.type == 'wgan':
        #     self._train_wgan(cls)
        # elif self.type == 'gan':
        self._train_gan(cls)

    # def _train_wgan(self, cls):
    #     one = torch.FloatTensor([1])
    #     mone = one * -1

    #     one = Variable(one).to(device)
    #     mone = Variable(mone).to(device)

    #     gen_iteration = 0
    #     for epoch in range(self.num_epochs):
    #         iterator = 0
    #         data_iterator = iter(self.data_loader)

    #         while iterator < len(self.data_loader):

    #             if gen_iteration < 25 or gen_iteration % 500 == 0:
    #                 d_iter_count = 100
    #             else:
    #                 d_iter_count = self.DITER

    #             d_iter = 0

    #             # Train the discriminator
    #             while d_iter < d_iter_count and iterator < len(self.data_loader):
    #                 d_iter += 1

    #                 for p in self.discriminator.parameters():
    #                     p.requires_grad = True

    #                 self.discriminator.zero_grad()

    #                 sample = next(data_iterator)
    #                 iterator += 1

    #                 right_images = sample['right_images']
    #                 right_embed = sample['right_embed']
    #                 wrong_images = sample['wrong_images']

    #                 right_images = Variable(right_images.float()).to(device)
    #                 right_embed = Variable(right_embed.float()).to(device)
    #                 wrong_images = Variable(wrong_images.float()).to(device)

    #                 outputs, _ = self.discriminator(right_images, right_embed)
    #                 real_loss = torch.mean(outputs)
    #                 real_loss.backward(mone)

    #                 if cls:
    #                     outputs, _ = self.discriminator(
    #                         wrong_images, right_embed)
    #                     wrong_loss = torch.mean(outputs)
    #                     wrong_loss.backward(one)

    #                 noise = Variable(torch.randn(right_images.size(
    #                     0), self.noise_dim), volatile=True).to(device)
    #                 noise = noise.view(noise.size(0), self.noise_dim, 1, 1)

    #                 fake_images = Variable(
    #                     self.generator(right_embed, noise).data)
    #                 outputs, _ = self.discriminator(fake_images, right_embed)
    #                 fake_loss = torch.mean(outputs)
    #                 fake_loss.backward(one)

    #                 # NOTE: Pytorch had a bug with gradient penalty at the time of this project development
    #                 # , uncomment the next two lines and remove the params clamping below if you want to try gradient penalty
    #                 # gp = Utils.compute_GP(self.discriminator, right_images.data, right_embed, fake_images.data, LAMBDA=10)
    #                 # gp.backward()

    #                 d_loss = real_loss - fake_loss

    #                 if cls:
    #                     d_loss = d_loss - wrong_loss

    #                 self.optimD.step()

    #                 for p in self.discriminator.parameters():
    #                     p.data.clamp_(-0.01, 0.01)

    #             # Train Generator
    #             for p in self.discriminator.parameters():
    #                 p.requires_grad = False
    #             self.generator.zero_grad()
    #             noise = Variable(torch.randn(
    #                 right_images.size(0), 100)).to(device)
    #             noise = noise.view(noise.size(0), 100, 1, 1)
    #             fake_images = self.generator(right_embed, noise)
    #             outputs, _ = self.discriminator(fake_images, right_embed)

    #             g_loss = torch.mean(outputs)
    #             g_loss.backward(mone)
    #             g_loss = - g_loss
    #             self.optimG.step()

    #             gen_iteration += 1

    #             self.logger.log_iteration_wgan(
    #                 epoch, gen_iteration, d_loss, g_loss, real_loss, fake_loss)

    #         if (epoch+1) % 50 == 0:
    #             Utils.save_checkpoint(
    #                 self.discriminator, self.generator, self.checkpoints_path, epoch)

    def _train_gan(self, cls):
        criterion = nn.BCELoss()
        l2_loss = nn.MSELoss()
        l1_loss = nn.L1Loss()

        max_inception_score = 0

        t = time()

        noise = torch.randn(
            self.batch_size * 2 if self.ds else self.batch_size, 100, 1, 1).to(device)

        real_labels = torch.ones(noise.size(0)).to(device)
        fake_labels = torch.zeros(noise.size(0)).to(device)
        smoothed_real_labels = torch.FloatTensor(
            Utils.smooth_label(real_labels.to(device).cpu().numpy(), -0.1)).to(device)

        for epoch in range(1, self.num_epochs+1):

            if self.ds and 400 > epoch > 300 and epoch % 20 == 1:
                print('Inserting data')
                self.al_method()

            d, g, dx, dgx = [], [], [], []

            for sample in self.data_loader:

                right_images = sample['right_images'].float().to(device)
                right_embed = sample['right_embed'].float().to(device)

                # wrong_images = sample['wrong_images'].float().to(device)
                # inter_embed = sample['inter_embed'].float().to(device)

                if self.ds:
                    new_embed = sample['new_embed'].float().to(device)
                    new_images = sample['new_images'].float().to(device)

                    right_images = torch.cat((right_images, new_images), 0)
                    right_embed = torch.cat((right_embed, new_embed), 0)

                # ======== One sided label smoothing ==========
                # Helps preventing the discriminator from overpowering the
                # generator adding penalty when the discriminator is too confident
                # =============================================

                # Train the discriminator
                self.discriminator.zero_grad()

                outputs, activation_real = self.discriminator(
                    right_images, right_embed)
                real_loss = criterion(
                    outputs[:self.batch_size], smoothed_real_labels[:self.batch_size])

                if self.ds:
                    real_loss += criterion(
                        outputs[self.batch_size:], smoothed_real_labels[self.batch_size:])  # * 10

                real_score = outputs

                dx.append(real_score.mean())

                # if cls:
                #     outputs, _ = self.discriminator(wrong_images, right_embed)
                #     wrong_loss = criterion(outputs, fake_labels)

                fake_images = self.generator(right_embed, noise)
                outputs, _ = self.discriminator(fake_images, right_embed)
                fake_loss = criterion(
                    outputs[:self.batch_size], fake_labels[:self.batch_size])

                if self.ds:
                    fake_loss += criterion(
                        outputs[self.batch_size:], fake_labels[self.batch_size:])  # * 10

                fake_score = outputs

                dgx.append(fake_score.mean())

                d_loss = real_loss + fake_loss
                d.append(d_loss)

                # if cls:
                #     d_loss += wrong_loss

                d_loss.backward()
                self.optimD.step()

                # Train the generator

                self.generator.zero_grad()

                fake_images = self.generator(right_embed, noise)

                outputs, activation_fake = self.discriminator(
                    fake_images, right_embed)
                _, activation_real = self.discriminator(
                    right_images, right_embed)

                activation_fake = torch.mean(activation_fake, 0)
                activation_real = torch.mean(activation_real, 0)

                # ======= Generator Loss function============
                # This is a customized loss function, the first term is the regular cross entropy loss
                # The second term is feature matching loss, this measure the distance between the real and generated
                # images statistics by comparing intermediate layers activations
                # The third term is L1 distance between the generated and real images, this is helpful for the conditional case
                # because it links the embedding feature vector directly to certain pixel values.
                # ===========================================
                g_loss = criterion(outputs[:self.batch_size], real_labels[:self.batch_size]) \
                    + self.l2_coef * l2_loss(activation_fake[:self.batch_size], activation_real[:self.batch_size]) \
                    + self.l1_coef * \
                    l1_loss(fake_images[:self.batch_size],
                            right_images[:self.batch_size])

                if self.ds:
                    g_loss += (criterion(outputs[self.batch_size:], real_labels[self.batch_size:])
                               + self.l2_coef *
                               l2_loss(
                                   activation_fake[self.batch_size:], activation_real[self.batch_size:])
                               + self.l1_coef *
                               l1_loss(fake_images[self.batch_size:],
                                       right_images[self.batch_size:]))  # * 10

                # fake_images = self.generator(inter_embed, noise)
                # outputs, activation_fake = self.discriminator(
                #     fake_images, inter_embed)

                # g_loss += criterion(outputs, real_labels)

                g.append(g_loss)

                g_loss.backward()
                self.optimG.step()

            self.logger.log_iteration_gan(
                epoch, sum(d)/len(d), sum(g)/len(g), sum(dx)/len(dx), sum(dgx)/len(dgx))
            print('Time :', time() - t)

            if (epoch) % self.eval_interval == 0:
                try:
                    shutil.rmtree('results0')
                except:
                    pass

                Utils.save_checkpoint(
                    self.discriminator, self.generator, self.checkpoints_path, self.save_path, epoch)
                self.predict('results0')
                inception_score = get_inception_score('results0/*', device)[0]
                max_inception_score = max(max_inception_score, inception_score)
                print('IS:', inception_score, 'MAX IS:', max_inception_score)

    def predict(self, results):
        noise = torch.randn(self.eval_batch_size, 100, 1, 1).to(device)

        if not os.path.exists(f'{results}/'):
            os.makedirs(f'{results}/')

        with torch.no_grad():
            for sample in self.val_data_loader:
                right_embed = sample['right_embed'].float().to(device)
                txt = sample['txt']

                fake_images = self.generator(right_embed, noise)

                for image, t in zip(fake_images, txt):
                    im = Image.fromarray(image.data.mul_(127.5).add_(
                        127.5).byte().permute(1, 2, 0).cpu().numpy())
                    im.save(
                        f'{results}/{t.replace("/", "")[:100]}.jpg')
