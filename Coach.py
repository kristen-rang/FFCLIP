import os
import clip
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
import datetime
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from data.Dynamic_dataset import StyleDataset
from edit.utils import trans_img
from tqdm import tqdm
import losses.clip_loss as clip_loss
from losses.ID.id_loss import IDLoss
from losses.BG_loss import BackgroundLoss
from edit.dynamic_mapper import EditModel
from torch.optim.lr_scheduler import MultiStepLR
# from losses.discriminator import NLayerDiscriminator, weights_init, adopt_weight, vanilla_d_loss
import shutil
import torch.nn.functional as F


class Coach:
    def __init__(self, opts):
        self.global_step = 1

        # Initialize logger
        self.configs = OmegaConf.load(opts)
        if self.configs.resume.pretrain:
            base_dir = self.configs.resume.result_dir
        else:
            now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            base_dir = os.path.join(self.configs.train.base_dir, now)
        if not os.path.exists(base_dir):
            os.makedirs(base_dir, exist_ok=True)
        self.check_dir = os.path.join(base_dir, self.configs.train.checkpoint_dir)
        self.image_dir = os.path.join(base_dir, self.configs.train.image_dir)
        self.txt_path = os.path.join(base_dir, self.configs.train.txt_path)
        os.makedirs(self.check_dir, exist_ok=True)
        os.makedirs(self.image_dir, exist_ok=True)
        shutil.copy('./edit/dynamic_mapper.py', os.path.join(base_dir, 'dynamic_mapper.py'))
        shutil.copy('./edit/modules.py', os.path.join(base_dir, 'modules.py'))
        shutil.copy('./Coach.py', os.path.join(base_dir, 'Coach.py'))
        shutil.copy('./configs/edit.yaml', os.path.join(base_dir, 'edit.yaml'))
        # Initialize save interval
        self.best_val_loss = None
        if self.configs.train.save_interval is None:
            self.configs.train.save_interval = self.configs.train.max_epoch

        # 0. set up distributed device
        self.world_size = int(os.environ['WORLD_SIZE'])
        self.rank = int(os.environ["RANK"])
        self.local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(self.local_rank)
        dist.init_process_group(backend="nccl", init_method='env://', rank=self.rank, world_size=self.world_size)
        self.device = torch.device("cuda", self.local_rank)
        torch.manual_seed(42)

        # Initialize network

        model = EditModel(self.configs)
        if self.configs.resume.pretrain:
            ckpt = torch.load(self.configs.resume.resume_dir)["state_dict"]
            resume_ckpt = {k.split('.', 1)[-1]: v for k, v in ckpt.items()}
            model.load_state_dict(resume_ckpt, strict=True)
        model.train()
        self.model = DDP(model.to(self.device), device_ids=[self.local_rank], output_device=self.local_rank,
                         find_unused_parameters=True)

        self.clip_model, _ = clip.load(self.configs.clip_path, device="cuda")
        self.clip_model = self.clip_model.eval().float().cuda()
        for param in self.clip_model.parameters():
            param.requires_grad = False

        # Initialize loss
        if self.configs.loss.id_lambda > 0:
            self.id_loss = IDLoss(self.configs.ID_path).to(self.device).eval()
        if self.configs.loss.clip_lambda > 0:
            self.clip_loss = clip_loss.CLIPLoss()
        if self.configs.loss.x_l2_lambda > 0:
            self.x_l2_loss = nn.MSELoss().to(self.device).eval()
        if self.configs.loss.bgloss_lambda > 0:
            self.bgloss = BackgroundLoss(self.configs).eval()
        if self.configs.loss.discrim_lambda > 0:
            self.discriminator = NLayerDiscriminator(input_nc=3,
                                                     n_layers=3,
                                                     use_actnorm=False,
                                                     ndf=64
                                                     ).apply(weights_init)

        # optimizer  ##########################################################
        params = list(model.att_mapper.parameters())
        if self.configs.loss.discrim_lambda > 0:
            params += list(self.discriminator.parameters())
        self.optimizer = torch.optim.Adam(params, lr=self.configs.train.learning_rate, eps=1e-8)

        self.schedular = MultiStepLR(self.optimizer,
                                     [int(x * self.configs.train.max_epoch) for x in self.configs.train.lr_steps_rel],
                                     gamma=self.configs.train.lr_gamma)

        # Initialize dataset
        train_dataset = StyleDataset(self.configs, "train")
        test_dataset = StyleDataset(self.configs, "test")

        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                        shuffle=True)  #######################
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, shuffle=False)
        self.trainloader = DataLoader(train_dataset, batch_size=self.configs.data.batch_size, num_workers=0,
                                      pin_memory=True, sampler=train_sampler)
        self.testloader = DataLoader(test_dataset, batch_size=self.configs.data.batch_size, num_workers=0,
                                     pin_memory=True, sampler=test_sampler)




    ################### classifier loss ######################

    def train(self):
        self.model.train()
        while self.global_step < self.configs.train.max_epoch:
            for batch_idx, batch in enumerate(tqdm(self.trainloader)):
                self.optimizer.zero_grad()
                attributes, discirption, text, w = batch
                with torch.no_grad():
                    img = self.model.module.generator([w.cuda()], input_is_latent=True, randomize_noise=False,
                                                      truncation=1).to(self.device)

                w_hat = self.model(w, attributes.float())
                img_hat = self.model.module.generator([w_hat], input_is_latent=True, randomize_noise=False,
                                                      truncation=1).to(self.device)

                loss, loss_dict = self.calc_loss(w.cuda(), w_hat.cuda(), img, img_hat, discirption, text)

                # with torch.autograd.detect_anomaly():
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.configs.train.gradient_clipping)

                self.optimizer.step()
                self.print_metrics(loss_dict, prefix='train')
            with torch.no_grad():
                # Logging related
                if self.global_step % self.configs.train.image_interval == 0 and self.rank == 0:
                    self.parse_and_log_images(img, img_hat, title='images_train', text=text)
                if self.global_step % self.configs.train.board_interval == 0 and self.rank == 0:
                    self.print_metrics(loss_dict, prefix='train')
                    self.log_metrics(loss_dict, prefix='train')

            torch.cuda.empty_cache()
            # Validation related
            if self.global_step % self.configs.train.val_interval == 0 or self.global_step == self.configs.train.max_epoch and self.rank == 0:
                with torch.no_grad():
                    val_loss_dict = None
                    val_loss_dict = self.validate()
                    if val_loss_dict and (self.best_val_loss is None or val_loss_dict[
                        'loss'] < self.best_val_loss) and torch.distributed.get_rank() == 0:
                        self.best_val_loss = val_loss_dict['loss']
                        self.checkpoint_me(val_loss_dict, is_best=True)

            if self.global_step % self.configs.train.save_interval == 0 or self.global_step == self.configs.train.max_epoch and self.rank == 0:
                if val_loss_dict is not None:
                    self.checkpoint_me(val_loss_dict, is_best=False)
                else:
                    self.checkpoint_me(loss_dict, is_best=False)

            if self.global_step == self.configs.train.max_epoch:
                print('OMG, finished training!')
                break
            self.schedular.step()
            self.global_step += 1

    def validate(self):
        with torch.no_grad():
            agg_loss_dict = []
            self.model.eval()
            for batch_idx, batch in enumerate(self.testloader):
                val_attributes, val_discirption, val_text, w = batch
                w_hat = self.model(w, val_attributes.float())
                with torch.no_grad():
                    img_hat = self.model.module.generator([w_hat], input_is_latent=True, randomize_noise=False,
                                                          truncation=1).to(self.device)
                    img = self.model.module.generator([w.cuda()], input_is_latent=True, randomize_noise=False,
                                                      truncation=1).to(self.device)

                loss, cur_loss_dict = self.calc_loss(w.cuda(), w_hat.cuda(), img, img_hat, val_discirption, val_text)
                agg_loss_dict.append(cur_loss_dict)
                del cur_loss_dict
            self.parse_and_log_images(img, img_hat, title='images_val', text=val_text)
        val_loss_dict = self.aggregate_loss_dict(agg_loss_dict)
        self.log_metrics(val_loss_dict, prefix='test')
        self.print_metrics(val_loss_dict, prefix='test')
        self.model.train()
        return val_loss_dict

    def checkpoint_me(self, loss_dict, is_best):
        save_name = 'best_model.pt' if is_best else 'epoch_{}.pt'.format(self.global_step)
        save_dict = self.__get_save_dict()
        checkpoint_path = os.path.join(self.check_dir, save_name)
        torch.save(save_dict, checkpoint_path)
        with open(self.txt_path, 'a') as f:
            if is_best:
                f.write(
                    '**Best**: Step - {}, Loss - {:.3f} \n{}\n'.format(self.global_step, self.best_val_loss, loss_dict))
            else:
                f.write('Step - {}, \n{}\n'.format(self.global_step, loss_dict))

    def calc_loss(self, w, w_hat, x, x_hat, discirption, text):
        loss_dict = {}
        loss = 0.0
        if self.configs.loss.id_lambda > 0:
            loss_id, _ = self.id_loss(x_hat.contiguous(), x.contiguous())
            loss_dict['loss_id'] = float(loss_id)
            loss = loss_id * self.configs.loss.id_lambda

        if self.configs.loss.clip_lambda > 0:
            #############
            loss_clip = 0.0
            for i in range(x_hat.shape[0]):

                if text[i] in self.age:
                    loss_clip = loss_clip + self.clip_loss(x_hat[i].unsqueeze(0).float().contiguous(),
                                                           discirption[i].contiguous()).mean() * self.configs.loss.age_lambda
                
                elif text[i] in self.emotion:
                    loss_clip = loss_clip + self.clip_loss(x_hat[i].unsqueeze(0).float().contiguous(),
                                                           discirption[i].contiguous()).mean() * self.configs.loss.emotion_lambda
                
                else:
                    loss_clip = loss_clip + self.clip_loss(x_hat[i].unsqueeze(0).float().contiguous(),
                                                           discirption[i].contiguous()).mean() * self.configs.loss.clip_lambda

            loss_clip = loss_clip.to(self.device)
            loss_dict['loss_clip'] = float(loss_clip)
            loss += loss_clip

        if self.configs.loss.latent_lambda > 0:
            rec_loss1 = torch.abs(w.contiguous() - w_hat.contiguous()).mean().to(self.device)

            loss_dict['loss_latent'] = float(rec_loss1)
            loss += rec_loss1 * self.configs.loss.latent_lambda

        if self.configs.loss.x_l2_lambda > 0:
            loss_l2_x = torch.abs(x.contiguous() - x_hat.contiguous()).mean()


            loss_dict['loss_l2_x'] = float(loss_l2_x)
            loss += loss_l2_x * self.configs.loss.x_l2_lambda

        if self.configs.loss.discrim_lambda > 0:
            self.discriminator = self.discriminator.cuda()
            logits_real = self.discriminator(x.contiguous().detach())
            logits_fake = self.discriminator(x_hat.contiguous().detach())
            disc_factor = adopt_weight(1.0, self.global_step, threshold=self.configs.train.disc_start)
            d_loss = disc_factor * vanilla_d_loss(logits_real, logits_fake)
            loss_dict['d_loss'] = float(d_loss)
            loss += d_loss * self.configs.loss.discrim_lambda

        if self.configs.loss.embd_lambda > 0:
            img_1 = F.interpolate(x, size=(224, 224), mode='bicubic',
                                  align_corners=True)  # torch.Size([B, 3, 224, 224])
            img_2 = F.interpolate(x_hat, size=(224, 224), mode='bicubic', align_corners=True)
            embd1 = self.clip_model.encode_image(img_1.contiguous())
            embd2 = self.clip_model.encode_image(img_2.contiguous())
            loss_l1_embd = torch.abs(embd1 - embd2).mean()
            loss_dict['loss_l1_embd'] = float(loss_l1_embd)
            loss += loss_l1_embd * self.configs.loss.embd_lambda

        if self.configs.loss.bgloss_lambda > 0:
            self.bgloss = self.bgloss.cuda()
            loss_background = self.bgloss(x.contiguous(), x_hat.contiguous())

            loss_dict['loss_background'] = float(loss_background)
            loss += loss_background * self.configs.loss.bgloss_lambda


        loss_dict['loss'] = float(loss)  # .cpu().detach())
        return loss, loss_dict

    def aggregate_loss_dict(self, agg_loss_dict):
        mean_vals = {}
        for output in agg_loss_dict:
            for key in output:
                mean_vals[key] = mean_vals.setdefault(key, []) + [output[key]]
        for key in mean_vals:
            if len(mean_vals[key]) > 0:
                mean_vals[key] = sum(mean_vals[key]) / len(mean_vals[key])
            else:
                print('{} has no value'.format(key))
                mean_vals[key] = 0
        return mean_vals

    def log_metrics(self, metrics_dict, prefix):
        for key, value in metrics_dict.items():
            # pass
            print(f"step: {self.global_step} \t metric: {prefix}/{key} \t value: {value}")
        # self.logger.add_scalar('{}/{}'.format(prefix, key), value, self.global_step)

    def print_metrics(self, metrics_dict, prefix):
        print('Metrics for {}, step {}'.format(prefix, self.global_step))
        for key, value in metrics_dict.items():
            print('\t{} = '.format(key), value)

    def parse_and_log_images(self, x, x_hat, title, text, index=None):
        text = '_'.join(text)
        if index is None:
            path = os.path.join(self.image_dir, title, f'{str(self.global_step) + text}.jpg')
        else:
            path = os.path.join(self.image_dir, title, f'{str(self.global_step) + text}_{str(index).zfill(5)}.jpg')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torchvision.utils.save_image(torch.cat([x.detach().cpu(), x_hat.detach().cpu()]), path,
                                     normalize=True, scale_each=True, range=(-1, 1), nrow=x.shape[0])

    def __get_save_dict(self):
        save_dict = {
            'state_dict': self.model.state_dict(),
            'opts': vars(self.configs)
        }
        return save_dict


if __name__ == '__main__':
    opts = 'configs.yaml'
    coach = Coach(opts)
    coach.train()

