import torch
from torch.autograd import Variable
from collections import OrderedDict
from .models import BaseModel
from networks.networks import NetworksFactory
from losses.model_loss import *
from utils import util
import numpy as np


class FineGrainedGAN(BaseModel):
    def __init__(self, opt):
        super(FineGrainedGAN, self).__init__(opt)
        self._name = 'FineGrainedGAN'

        # create networks
        self._init_create_networks()

        # prefetch variables
        self._init_prefetch_inputs()

        # init
        self._init_losses()

    def _init_create_networks(self):
        # Generator
        self._G = self._create_generator()
        self._G.init_weights()
        # Discriminator
        self._D = self._create_discriminator()
        self._D.init_weights()

        # init train variables
        if self._is_train:
            self._init_train_vars()

        # load networks and optimizers
        if not self._is_train or self._opt.load_epoch > 0:
            self.load()

        if len(self._gpu_ids) > 1:
            self._G = torch.nn.DataParallel(self._G, device_ids=self._gpu_ids)
        self._G.cuda()
        if len(self._opt.gpu_ids) > 1:
            self._D = torch.nn.DataParallel(self._D, device_ids=self._opt.gpu_ids)
        self._D.cuda()

        # init train variables
        self._init_train_vars()


    def _create_discriminator(self):
        # _opt.cond_nc = 17 by default
        return NetworksFactory.get_by_name(self._opt.discriminator_name, self._opt)

    def _create_generator(self):
        # _opt.cond_nc = 17 by default
        return NetworksFactory.get_by_name(self._opt.generator_name, self._opt)

    def _init_train_vars(self):
        self._current_lr_G = self._opt.lr_G
        self._current_lr_D = self._opt.lr_D

        # initialize optimizers
        self._optimizer_G = torch.optim.Adam(self._G.parameters(), lr=self._current_lr_G,
                                             betas=(self._opt.G_adam_b1, self._opt.G_adam_b2))
        self._optimizer_D = torch.optim.Adam(self._D.parameters(), lr=self._current_lr_D,
                                             betas=(self._opt.D_adam_b1, self._opt.D_adam_b2))

    def _init_prefetch_inputs(self):
        # _opt.batch_size = 4
        self._input_real_img = self._Tensor(self._opt.batch_size, 3, self._opt.image_size, self._opt.image_size)
        self._input_real_cond = self._Tensor(self._opt.batch_size, self._opt.cond_nc)
        self._input_desired_img = self._Tensor(self._opt.batch_size, 3, self._opt.image_size, self._opt.image_size)
        self._input_desired_cond = self._Tensor(self._opt.batch_size, self._opt.cond_nc)
        # self._input_cond_diff = self._Tensor(self._opt.batch_size, self._opt.cond_nc)
        self._input_real_img_path = None
        self._input_real_cond_path = None

    def _init_losses(self):
        # define loss functions
        self._criterion_identity = torch.nn.L1Loss().cuda()    # get_cycle_loss()
        self._criterion_D_cond = torch.nn.MSELoss().cuda()  # get_cond_loss()

        # init losses G
        self._loss_g_fake = Variable(self._Tensor([0]))
        self._loss_g_cond = Variable(self._Tensor([0]))

        self._loss_g_rec_cond = Variable(self._Tensor([0]))
        self._loss_g_idt = Variable(self._Tensor([0]))
        self._loss_g_cyc = Variable(self._Tensor([0]))
        self._loss_g_smooth = Variable(self._Tensor([0]))

        # init losses D
        self._loss_d_real = Variable(self._Tensor([0]))
        self._loss_d_cond = Variable(self._Tensor([0]))
        self._loss_d_fake = Variable(self._Tensor([0]))
        self._loss_d_gp = Variable(self._Tensor([0]))

    def set_input(self, input):
        self._input_real_img.resize_(input['real_img'].size()).copy_(input['real_img'])
        self._input_real_cond.resize_(input['real_cond'].size()).copy_(input['real_cond'])
        self._input_desired_img.resize_(input['desired_img'].size()).copy_(input['desired_img'])
        self._input_desired_cond.resize_(input['desired_cond'].size()).copy_(input['desired_cond'])
        self._input_real_img_path = input['real_img_path']

        if len(self._gpu_ids) > 0:
            self._input_real_img = self._input_real_img.cuda(self._gpu_ids[0], async=True)
            self._input_real_cond = self._input_real_cond.cuda(self._gpu_ids[0], async=True)
            self._input_desired_cond = self._input_desired_cond.cuda(self._gpu_ids[0], async=True)

    def set_train(self):
        self._G.train()
        self._D.train()
        self._is_train = True

    def set_eval(self):
        self._G.eval()
        self._is_train = False

    def get_image_paths(self):
        return OrderedDict([('real_img', self._input_real_img_path)])

    def forward(self, keep_data_for_visuals=False, return_estimates=False):
        if not self._is_train:
            with torch.no_grad():
                # convert tensor to variables
                real_img = Variable(self._input_real_img)#, requires_grad=True)
                real_cond = Variable(self._input_real_cond)#, requires_grad=True)
                desired_img = Variable(self._input_desired_img)#, requires_grad=False)
                desired_cond = Variable(self._input_desired_cond)#, requires_grad=True)

                # generate fake images
                fake_imgs = self._G.forward(real_img, desired_cond - real_cond)
                d_fake_desired_img_prob, d_fake_desired_img_cond = self._D.forward(fake_imgs.detach())
                self._loss_g_fake = self._compute_loss_D(d_fake_desired_img_prob, False) * self._opt.lambda_D_prob
                self._loss_g_cond = self._criterion_D_cond(d_fake_desired_img_cond, self._desired_cond) * self._opt.lambda_D_cond
                self._loss_g_smooth = self._compute_loss_smooth(fake_imgs) * self._opt.lambda_smooth
                # Reconstruction error
                fake_imgs_rec = self._G.forward(real_img, real_cond - real_cond)
                self._loss_g_idt = self._criterion_identity(fake_imgs_rec, self._real_img) * self._opt.lambda_rec_l1

                # Cycle Reconstruction
                cycle_rec_img = self._G.forward(fake_imgs, real_cond - desired_cond)
                self._loss_g_cyc = self._criterion_identity(cycle_rec_img, self._real_img) * self._opt.lambda_cyc_l1


                imgs, data = None, None
                # keep data for visualization
                if keep_data_for_visuals:
                    self._vis_real_img = util.tensor2im(real_img.data)
                    self._vis_desired_img = util.tensor2im(desired_img.data)

                    self._vis_fake_img = util.tensor2im(fake_imgs.data)
                    self._vis_fake_img_rec = util.tensor2im(fake_imgs_rec.data)
                    self._vis_cycle_rec_img = util.tensor2im(cycle_rec_img.data)

                    self._vis_real_cond = self._input_real_cond.cpu()[0, ...].numpy()
                    self._vis_desired_cond = self._input_desired_cond.cpu()[0, ...].numpy()

                if return_estimates:
                    # generate images
                    # ---- original
                    im_real_img = util.tensor2im(real_img.data)
                    im_desired_img = util.tensor2im(desired_img.data)
                    # ---- fake images
                    im_fake_imgs = util.tensor2im(fake_imgs.data)
                    # ---- keep cond same fake images
                    im_fake_imgs_rec = util.tensor2im(fake_imgs_rec.data)
                    # ---- cycle reconstruction images
                    im_cycle_rec_imgs = util.tensor2im(cycle_rec_img.data)

                    im_concat_img = np.concatenate([im_real_img, im_desired_img,
                                                    im_fake_imgs, im_fake_imgs_rec, im_cycle_rec_imgs],
                                                   1)

                    imgs = OrderedDict([('real_img', im_real_img),
                                        ('desired_emotion_img', im_desired_img),
                                        ('fake_imgs', im_fake_imgs),
                                        ('fake_imgs_rec', im_fake_imgs_rec),
                                        ('cycle_rec', im_cycle_rec_imgs),
                                        ('concat', im_concat_img),
                                        ])
                    data = OrderedDict([('real_path', self._input_real_img_path),
                                        ('desired_cond', desired_cond.data[0,...].cpu().numpy().astype('str'))
                                        ])

                    return imgs, data

    def optimize_parameters(self, train_discriminator=True, train_generator=True, keep_data_for_visuals=False):
        if self._is_train:
            # convert Tensor to Variables
            self._B = self._input_real_img.size(0)
            self._real_img = Variable(self._input_real_img)#, requires_grad=True)
            self._real_cond = Variable(self._input_real_cond)#, requires_grad=True)
            self._desired_img = Variable(self._input_desired_img)#, requires_grad=True)
            self._desired_cond = Variable(self._input_desired_cond)#, requires_grad=True)
            # self._cond_diff = Variable(self._input_cond_diff)

            # train D
            if train_discriminator:
                loss_D, fake_imgs_masked = self._forward_D()
                self._optimizer_D.zero_grad()
                loss_D.backward()
                self._optimizer_D.step()

                loss_D_gp = self._gradient_penalty_D(fake_imgs_masked)
                self._optimizer_D.zero_grad()
                loss_D_gp.backward()
                self._optimizer_D.step()

            # train G
            if train_generator:
                loss_G = self._forward_G(keep_data_for_visuals)
                self._optimizer_G.zero_grad()
                loss_G.backward()
                self._optimizer_G.step()

    def _forward_D(self):
        # D(real_I)
        d_real_img_prob, d_real_img_cond = self._D.forward(self._real_img)  # out_real = self.conv1(h) out_aux = self.conv2(h)
        # real or fake, output is 0 or 1, use the output of D
        self._loss_d_real = self._compute_loss_D(d_real_img_prob, True) * self._opt.lambda_D_prob
        self._loss_d_cond = self._criterion_D_cond(d_real_img_cond, self._real_cond) * self._opt.lambda_D_cond

        # generate fake images
        fake_imgs = self._G.forward(self._real_img, self._desired_cond - self._real_cond)
        # D(fake_I)
        d_fake_desired_img_prob, d_fake_desired_img_cond = self._D.forward(fake_imgs.detach())
        self._loss_d_fake = self._compute_loss_D(d_fake_desired_img_prob, False) * self._opt.lambda_D_prob
        self._loss_d_fake_cond = self._criterion_D_cond(d_fake_desired_img_cond, self._desired_cond) * self._opt.lambda_D_cond

        return self._loss_d_real + self._loss_d_cond + self._loss_d_fake, fake_imgs

    def _gradient_penalty_D(self, fake_imgs_masked):
        # interpolate sample
        alpha = torch.rand(self._B, 1, 1, 1).cuda().expand_as(self._real_img)
        interpolated = Variable(alpha * self._real_img.data + (1 - alpha) * fake_imgs_masked.data, requires_grad=True)
        interpolated_prob, _ = self._D(interpolated)

        # compute gradients
        grad = torch.autograd.grad(outputs=interpolated_prob,
                                   inputs=interpolated,
                                   grad_outputs=torch.ones(interpolated_prob.size()).cuda(),
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        # penalize gradients
        grad = grad.view(grad.size(0), -1)
        grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
        self._loss_d_gp = torch.mean((grad_l2norm - 1) ** 2) * self._opt.lambda_D_gp

        return self._loss_d_gp

    def _compute_loss_D(self, estim, is_real):
        return -torch.mean(estim) if is_real else torch.mean(estim)

    def _forward_G(self, keep_data_for_visuals):
        # ---- adversarial loss
        # G(I_yr|yg)  generate fake images with desired cond
        fake_imgs = self._G.forward(self._real_img, self._desired_cond - self._real_cond)
        d_fake_img_prob, d_fake_img_cond = self._D.forward(fake_imgs)
        self._loss_g_fake = self._compute_loss_D(d_fake_img_prob, True) * self._opt.lambda_D_prob
        self._loss_g_cond = self._criterion_D_cond(d_fake_img_cond, self._desired_cond) * self._opt.lambda_G_fake_cond
        self._loss_g_smooth = self._compute_loss_smooth(fake_imgs) * self._opt.lambda_smooth

        fake_imgs_rec = self._G.forward(self._real_img, self._real_cond - self._real_cond)
        self._loss_g_idt = self._criterion_identity(fake_imgs_rec, self._real_img) * self._opt.lambda_rec_l1

        # cycle
        cycle_rec_img = self._G.forward(fake_imgs, self._real_cond - self._desired_cond)
        self._loss_g_cyc = self._criterion_identity(cycle_rec_img, self._real_img) * self._opt.lambda_cyc_l1

        # keep data for visualization
        if keep_data_for_visuals:
            self._vis_real_img = util.tensor2im(self._input_real_img)
            self._vis_desired_img = util.tensor2im(self._desired_img)
            self._vis_fake_img = util.tensor2im(fake_imgs.data)
            self._vis_fake_img_rec = util.tensor2im(fake_imgs_rec.data)
            self._vis_cycle_rec_img = util.tensor2im(cycle_rec_img.data)
            self._vis_real_cond = self._input_real_cond.cpu()[0, ...].numpy()
            self._vis_desired_cond = self._input_desired_cond.cpu()[0, ...].numpy()

        loss = self._loss_g_fake + self._loss_g_cond + self._loss_g_smooth + self._loss_g_idt + self._loss_g_cyc

        return loss

    def _compute_loss_smooth(self, mat):
        return torch.sum(torch.abs(mat[:, :, :, :-1] - mat[:, :, :, 1:])) + \
               torch.sum(torch.abs(mat[:, :, :-1, :] - mat[:, :, 1:, :]))

    def get_current_errors(self):
        loss_dict = OrderedDict([('g_fake', self._loss_g_fake.item()),
                                 ('g_cond', self._loss_g_cond.item()),
                                 ('g_smooth', self._loss_g_smooth.item()),
                                 ('g_rec', self._loss_g_idt.item()),
                                 ('g_cyc', self._loss_g_cyc.item()),
                                 ('d_real', self._loss_d_real.item()),
                                 ('d_cond', self._loss_d_cond.item()),
                                 ('d_fake', self._loss_d_fake.item()),
                                 ('d_gp', self._loss_d_gp.item()),
                                 ])

        return loss_dict

    def get_current_scalars(self):
        return OrderedDict([('lr_G', self._current_lr_G), ('lr_D', self._current_lr_D)])

    def get_current_visuals(self):
        # visuals return dictionary
        visuals = OrderedDict()
        visuals['1_input_img'] = self._vis_real_img
        visuals['2_desired_img'] = self._vis_desired_img
        visuals['3_fake_img'] = self._vis_fake_img
        visuals['4_fake_img_rec'] = self._vis_fake_img_rec
        visuals['5_rec_real_img'] = self._vis_cycle_rec_img

        return visuals

    def save(self, label):
        # save networks
        self._save_network(self._G, 'G', label)
        self._save_network(self._D, 'D', label)

        # save optimizers
        self._save_optimizer(self._optimizer_G, 'G', label)
        self._save_optimizer(self._optimizer_D, 'D', label)

    def load(self):
        load_epoch = self._opt.load_epoch

        # load G
        self._load_network(self._G, 'G', load_epoch)

        if self._is_train:
            # load D
            self._load_network(self._D, 'D', load_epoch)
            # load optimizers
            self._load_optimizer(self._optimizer_G, 'G', load_epoch)
            self._load_optimizer(self._optimizer_D, 'D', load_epoch)

    def update_learning_rate(self, epoch_i):
        # updated learning rate G
        lr_decay_G = self._opt.lr_G / self._opt.nepochs_decay
        self._current_lr_G -= lr_decay_G
        for param_group in self._optimizer_G.param_groups:
            param_group['lr'] = self._current_lr_G
        print('update G learning rate: %f -> %f' % (self._current_lr_G + lr_decay_G, self._current_lr_G))

        # update learning rate D
        lr_decay_D = self._opt.lr_D / self._opt.nepochs_decay
        self._current_lr_D -= lr_decay_D
        for param_group in self._optimizer_D.param_groups:
            param_group['lr'] = self._current_lr_D
        print('update D learning rate: %f -> %f' % (self._current_lr_D + lr_decay_D, self._current_lr_D))

    def _l1_loss_with_target_gradients(self, input, target):
        return torch.sum(torch.abs(input - target)) / input.data.nelement()
