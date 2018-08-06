import os
import numpy as np
import torch
from torch.nn import functional as F

from .utils import BaseTrainer, kl_gauss_unag, \
  kl_cat_unag, sse_loss, overrides  


def l1_loss(source, target):
  return torch.abs(source - target).mean()


class Trainer(BaseTrainer):
  @overrides(BaseTrainer)
  def put_in_work(self):
    """Puts in work like a man possessed."""
    epochs = int(np.ceil(self.args.steps / len(self.dataloader)))
    step = 0
    for _ in range(epochs):
      for x in self.dataloader:
        step += 1
        if step > self.args.steps: return
    
        # 1. optimize GAN discriminator
        x = x.to(self.device)
        x_recon_disc, _ = self.nets['disc'](x)
        real_loss = l1_loss(x_recon_disc, x)
        x_fake = self.nets['vae'].sample(len(x)).detach()
        x_fake_recon, _ = self.nets['disc'](x_fake)
        gan_disc_loss = real_loss - self.args.k*l1_loss(x_fake_recon, x_fake)
      
        self.opt['disc'].zero_grad()
        gan_disc_loss.backward()
        self.opt['disc'].step()

        # 2. optimize VAE decoder / BEGAN generator
        x_recon, z_post, dist_params = self.nets['vae'](x)
        _, x_recon_disc_fmap = self.nets['disc'](x_recon, full=False)
        _, x_disc_fmap = self.nets['disc'](x, full=False)
        fw_sse = sse_loss(x_recon_disc_fmap, x_disc_fmap)
        pw_bce = F.binary_cross_entropy(
          x_recon, x, reduce=False).mean(0).sum()  # for logging
      
        x_fake = self.nets['vae'].sample(len(x)).detach()
        x_fake_recon, _ = self.nets['disc'](x_fake)
        gan_gen_loss = l1_loss(x_fake_recon, x_fake)

        self.opt['dec'].zero_grad()
        gan_dec_loss = self.args.fw_coeff*fw_sse + gan_gen_loss
        gan_dec_loss.backward(retain_graph=True)
        self.opt['dec'].step()

        # 3. optimize VAE encoder
        # KL for gaussian
        kl_cont_dw = torch.empty(0).to(self.device)
        cont_cap_loss = 0
        if 'cont' in dist_params.keys():
          mu, logvar = dist_params['cont']
          kl_cont_dw = kl_gauss_unag(mu, logvar).mean(0)
          cont_cap_loss = self.get_cap_loss(kl_cont_dw.sum(), step)
 
        # KL for categorical
        kl_cats = torch.empty(0).to(self.device)
        cat_cap_loss = 0
        if 'cat' in dist_params.keys():
          for logits in dist_params['cat']:
            kl_cat = kl_cat_unag(logits).sum(1).mean()
            kl_cats = torch.cat((kl_cats, kl_cat.view(1)))
          cat_cap_loss = self.get_cap_loss(kl_cats.sum(), step)

        self.opt['enc'].zero_grad()
        enc_loss = fw_sse + cont_cap_loss + cat_cap_loss
        enc_loss.backward(retain_graph=True)
        self.opt['enc'].step()
  
        # 4. equilibrium maintenance 
        balance = (self.args.gamma*real_loss - gan_gen_loss).item()
        self.args.k += self.args.lambda_k*balance
        self.args.k = max(min(1, self.args.k), 0)
        M = real_loss.item() + np.abs(balance)

        # log...
        self.logger.log_val('pw_bce', pw_bce.item())
        self.logger.log_val('fw_sse', fw_sse.item())
        self.logger.log_val('M', M)
        self.logger.log_val('cont kl', kl_cont_dw.data.cpu().numpy())
        self.logger.log_val('cat kl', kl_cats.data.cpu().numpy())
          
        if not step % self.args.log_interval:
          self.logger.print(step)
        
        if not step % self.args.save_interval:
          filepath = os.path.join(self.args.logdir, 'model.ckpt')
          torch.save(self.nets['vae'], filepath)

          self.logger.save(step)
          self.vis.traverse(step)
          self.vis.recon(step)

