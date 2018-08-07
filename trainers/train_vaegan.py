import os
import numpy as np
import torch
from torch.nn import functional as F
from torch.optim.lr_scheduler import ExponentialLR

from utils.misc import overrides
from .utils import BaseTrainer, kl_gauss_unag, \
  kl_cat_unag, sse_loss


def zero_grads(nets):
  for key in nets: nets[key].zero_grad()


class Trainer(BaseTrainer):
  @overrides(BaseTrainer)
  def put_in_work(self):
    """Puts in work like a man possessed."""
    epochs = int(np.ceil(self.args.steps / len(self.dataloader)))
    step = 0

    margin = 0.35
    equilibrium = 0.68
    
    decay_lr = 0.9
    enc_lr_sched = ExponentialLR(self.opt['enc'], gamma=decay_lr)
    dec_lr_sched = ExponentialLR(self.opt['dec'], gamma=decay_lr)
    disc_lr_sched = ExponentialLR(self.opt['disc'], gamma=decay_lr)

    for _ in range(epochs):
      for x in self.dataloader:
        step += 1
        if step > self.args.steps: return
    
        # gather relevant statistics
        x = x.to(self.device)
        disc_real_out, x_feat = self.nets['disc'](x)
        x_sample = self.nets['vae'].sample(len(x)).detach()
        disc_sample_out, x_sample_feat = self.nets['disc'](x_sample)
        x_recon, z_post, dist_params = self.nets['vae'](x)
        disc_recon_out, x_recon_feat = self.nets['disc'](x_recon)
      
        # feature-wise sse
        fw_sse = 0.5*(x_feat - x_sample_feat).pow(2).sum() + \
          0.5*(x_feat - x_recon_feat).pow(2).sum()
      
        # KL for gaussian
        kl_cont_dw = torch.empty(0).to(self.device)
        cont_cap_loss = 0
        if 'cont' in dist_params.keys():
          mu, logvar = dist_params['cont']
          kl_cont_dw = kl_gauss_unag(mu, logvar).sum(0)
          cont_cap_loss = self.get_cap_loss(kl_cont_dw.sum(), step)
 
        # KL for categorical
        kl_cats = torch.empty(0).to(self.device)
        cat_cap_loss = 0
        if 'cat' in dist_params.keys():
          for logits in dist_params['cat']:
            kl_cat = kl_cat_unag(logits).sum(1).sum()
            kl_cats = torch.cat((kl_cats, kl_cat.view(1)))
          cat_cap_loss = self.get_cap_loss(kl_cats.sum(), step)
  
        # setup losses for different subnetworks
        loss_enc = fw_sse + cont_cap_loss + cat_cap_loss

        loss_dec = -(torch.log(disc_sample_out + 1e-10) + \
          torch.log(disc_recon_out + 1e-10)).sum()
        loss_dec = 0.5*self.args.fw_coeff*fw_sse + \
          (1 - self.args.fw_coeff)*loss_dec

        loss_disc = -(torch.log(disc_real_out + 1e-10) + \
          torch.log(1 - disc_sample_out + 1e-10) + \
          torch.log(1 - disc_recon_out + 1e-10)).sum()
          
        # selectively disable the decoder or the disc if they are unbalanced
        train_disc, train_dec = True, True
        if disc_real_out.mean() < equilibrium-margin or \
            disc_sample_out.mean() < equilibrium-margin:
          train_disc = False
        if disc_real_out.mean() > equilibrium+margin or \
            disc_sample_out.mean() > equilibrium+margin:
          train_dec = False
        if not (train_dec and train_disc):
          train_disc, train_dec = True, True
        
        # optimize
        zero_grads(self.nets)
        loss_enc.backward(retain_graph=True)
        self.opt['enc'].step()
  
        zero_grads(self.nets)
        if train_dec:
          loss_dec.backward(retain_graph=True)
          self.opt['dec'].step()
        
        zero_grads(self.nets)
        if train_disc:
          loss_disc.backward()
          self.opt['disc'].step()

        # log...
        enc_lr = self.opt['enc'].param_groups[0]['lr']
        dec_lr = self.opt['dec'].param_groups[0]['lr']
        disc_lr = self.opt['disc'].param_groups[0]['lr']

        self.logger.log_val('enc_lr', enc_lr, runavg=False)
        self.logger.log_val('dec_lr', dec_lr, runavg=False)
        self.logger.log_val('disc_lr', disc_lr, runavg=False)

        self.logger.log_val('pw_sse', sse_loss(x, x_recon).item())
        self.logger.log_val('fw_sse', fw_sse.item())
        self.logger.log_val('loss_enc', loss_enc.item())
        self.logger.log_val('loss_dec', loss_dec.item())
        self.logger.log_val('loss_disc', loss_disc.item())
        self.logger.log_val('cont_kl', kl_cont_dw.data.cpu().numpy())
        self.logger.log_val('cat_kl', kl_cats.data.cpu().numpy())
          
        if not step % self.args.log_interval:
          self.logger.print(step)
        
        if not step % self.args.save_interval:
          filepath = os.path.join(self.args.logdir, 'model.ckpt')
          torch.save(self.nets['vae'], filepath)

          self.logger.save(step)
          self.vis.traverse(step)
          self.vis.recon(step)

      enc_lr_sched.step()
      dec_lr_sched.step()
      disc_lr_sched.step()

