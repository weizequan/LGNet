import torch
from .base_model import BaseModel
from . import networks
from .loss import l1_loss_mask, VGG16FeatureExtractor, style_loss, perceptual_loss, TV_loss

# global + local + global network
class Pix2PixGLGModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='batch', netG='unet_256', preprocess='resize', no_dropout=True, load_size=256, is_mask=True, gan_mode='nogan')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G', 'G_content', 'G_style', 'G_tv']
        if self.opt.gan_mode != 'nogan':
            self.loss_names += ['D_real', 'D_fake', 'G_GAN']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['masked_images1', 'merged_images1', 'images', 'merged_images2', 'merged_images3']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['G1', 'G2', 'G3']
        if self.opt.gan_mode != 'nogan':
            self.model_names += ['D']
        # define networks (both generator and discriminator)
        self.netG1 = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG1, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, gpu_ids=self.gpu_ids)
        self.netG2 = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG2, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, gpu_ids=self.gpu_ids)
        self.netG3 = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG3, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, gpu_ids=self.gpu_ids)

        if self.isTrain:  # define a discriminator;
            if opt.gan_mode != 'nogan':
                self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                              opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G1 = torch.optim.Adam(self.netG1.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_G2 = torch.optim.Adam(self.netG2.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_G3 = torch.optim.Adam(self.netG3.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            if opt.gan_mode != 'nogan':
                self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_G1)
            self.optimizers.append(self.optimizer_G2)
            self.optimizers.append(self.optimizer_G3)

            self.lossNet = VGG16FeatureExtractor()
            self.lossNet.cuda(opt.gpu_ids[0])

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.images = input['A' if AtoB else 'B'].to(self.device)
        self.masks = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.masked_images1 = self.images * (1 - self.masks) + self.masks 
        self.output_images1 = self.netG1(torch.cat((self.masked_images1, self.masks), 1))
        self.merged_images1 = self.images * (1 - self.masks) + self.output_images1 * self.masks
        self.output_images2 = self.netG2(torch.cat((self.merged_images1, self.masks), 1))
        self.merged_images2 = self.images * (1 - self.masks) + self.output_images2 * self.masks
        self.output_images3 = self.netG3(torch.cat((self.merged_images2, self.masks), 1))
        self.merged_images3 = self.images * (1 - self.masks) + self.output_images3 * self.masks

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(self.merged_images1.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False, is_disc=True)
        # Real
        pred_real = self.netD(self.images)
        self.loss_D_real = self.criterionGAN(pred_real, True, is_disc=True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        
        loss_G_valid = l1_loss_mask(self.output_images1*(1-self.masks), self.images*(1-self.masks), (1-self.masks))
        loss_G_hole = l1_loss_mask(self.output_images1*self.masks, self.images*self.masks, self.masks)
        self.loss_G = loss_G_valid + 6 * loss_G_hole

        loss_G_valid2 = l1_loss_mask(self.output_images2*(1-self.masks), self.images*(1-self.masks), (1-self.masks))
        loss_G_hole2 = l1_loss_mask(self.output_images2*self.masks, self.images*self.masks, self.masks)
        self.loss_G = self.loss_G + loss_G_valid2 + 6 * loss_G_hole2

        
        real_B_feats2 = self.lossNet(self.images)
        fake_B_feats2 = self.lossNet(self.output_images2)
        comp_B_feats2 = self.lossNet(self.merged_images2)

        self.loss_G_tv = TV_loss(self.merged_images2 * self.masks)
        self.loss_G_style = style_loss(real_B_feats2, fake_B_feats2) + style_loss(real_B_feats2, comp_B_feats2)
        self.loss_G_content = perceptual_loss(real_B_feats2, fake_B_feats2) + perceptual_loss(real_B_feats2, comp_B_feats2)
        self.loss_G = self.loss_G + 0.05*self.loss_G_content + 120*self.loss_G_style + 0.1*self.loss_G_tv
    
        loss_G_valid3 = l1_loss_mask(self.output_images3*(1-self.masks), self.images*(1-self.masks), (1-self.masks))
        loss_G_hole3 = l1_loss_mask(self.output_images3*self.masks, self.images*self.masks, self.masks)
        self.loss_G = self.loss_G + loss_G_valid3 + 6 * loss_G_hole3

        fake_B_feats3 = self.lossNet(self.output_images3)
        comp_B_feats3 = self.lossNet(self.merged_images3)

        self.loss_G_tv3 = TV_loss(self.merged_images3 * self.masks)
        self.loss_G_style3 = style_loss(real_B_feats2, fake_B_feats3) + style_loss(real_B_feats2, comp_B_feats3)
        self.loss_G_content3 = perceptual_loss(real_B_feats2, fake_B_feats3) + perceptual_loss(real_B_feats2, comp_B_feats3)
        self.loss_G = self.loss_G + 0.05*self.loss_G_content3 + 120*self.loss_G_style3 + 0.1*self.loss_G_tv3

        # G(A) should fake the discriminator

        if self.opt.gan_mode != 'nogan':
            pred_fake = self.netD(self.merged_images1)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True, is_disc=False)
            self.loss_G = self.loss_G + 0.1 * self.loss_G_GAN
            
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)

        if self.opt.gan_mode != 'nogan':
            # update D
            self.set_requires_grad(self.netD, True)  # enable backprop for D
            self.optimizer_D.zero_grad()     # set D's gradients to zero
            self.backward_D()                # calculate gradients for D
            self.optimizer_D.step()          # update D's weights

            # update G
            self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G

        # update G
        self.optimizer_G1.zero_grad()        # set G's gradients to zero
        self.optimizer_G2.zero_grad()        # set G's gradients to zero
        self.optimizer_G3.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G1.step()             # udpate G's weights
        self.optimizer_G2.step()             # udpate G's weights
        self.optimizer_G3.step()             # udpate G's weights