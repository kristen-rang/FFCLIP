import os
import torchvision.utils
import torch
import cv2
from configs.test_options import TestOptions
import numpy as np
import clip
from get_latents import Latent_make
from edit.dynamic_mapper import EditModel

opts = TestOptions().parse()
img_list = [os.path.join(opts.img_dir, i) for i in os.listdir(opts.img_dir)]
model = EditModel(opts)
tmp_ckpt = torch.load(opts.ckpt_dir, map_location='cuda')['state_dict']
ckpt = {k.split('.', 1)[-1]: v for k, v in tmp_ckpt.items()}
model.load_state_dict(ckpt, strict=True)
model = model.cuda()
clip_model, _ = clip.load(opts.clip_path, device="cuda")
clip_model = clip_model.eval().float().cuda()

attribute = []
if not os.path.exists(opts.result_dir):
    os.makedirs(opts.result_dir, exist_ok=True)
if opts.description:
    text = opts.description
    description = clip.tokenize(text).cuda()
    attribute.append(clip_model.encode_text(description).unsqueeze(0))
    
    
else:
    #description_list = ['Bald', 'Pink hair','Make up','Angry','Bowl cut hairstyle','Beard Blond mohawk hair','aged 10','age 90']
    description_list = ['Blue eyes', 'jewfro','dreadlocks','Disgust']
    #description_list = ['brown afro hair','brown Hi-top Fade hair','bald make up']
    #description_list = ['Red hair']
    
    for text in description_list:
        description = clip.tokenize(text).cuda()
        attribute.append(clip_model.encode_text(description).unsqueeze(0))
    

def run_img_inference():
    for path in img_list:
        raw_img = cv2.imread(path)
        img = raw_img[:, :, (2, 1, 0)]
        img = img[np.newaxis, :, :, :]
        with torch.no_grad():
            w, w_hat = model(img, attribute)
            img_hat = model.generator([w_hat], input_is_latent=True, randomize_noise=False,
                                      truncation=1)
            tmp = path.rsplit('.', 1)[0]
            save_path = os.path.join(opts.result_dir, f'e4e_{text}_{tmp}.jpg')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torchvision.utils.save_image(torch.cat([img.detach().cpu(), img_hat.detach().cpu()]), save_path,
                                         normalize=True, scale_each=True, range=(-1, 1))

def run_latent_inference():
    if not opts.latents_dir:
        latent_model = Latent_make(opts)
        imgs = None
        for path in img_list:
            raw_img = cv2.imread(path)
            img = raw_img[:, :, (2, 1, 0)]
            if img.shape[1] != 1024:
                img = cv2.resize(img,(1024,1024))
            img = img[np.newaxis, :, :, :]

            if imgs is not None:
                imgs = np.concatenate((imgs,img),axis=0)
            else:
                imgs = img
                
        latents = latent_model(imgs)
        #torch.save(latents,'test_imgs.pt')
    else:
        #latents = torch.from_numpy(np.load(opts.latents_dir)).cuda()
        latents = torch.from_numpy(np.load(opts.latents_dir)).cuda()
    girl_list = [924,927,922,938] #latents.shape[0]
    l1 = [76,72,198,17,2048,2022]
    # 80 88
    for i in range(2400,2500):
        with torch.no_grad():
            img = model.generator([latents[i].unsqueeze(0)], input_is_latent=True, randomize_noise=False, truncation=1)
            img_hat = []
        for attri in attribute:
            with torch.no_grad():
                w_hat = model(latents[i].unsqueeze(0), attri)
                print(w_hat.shape)
                img_hat.append(model.generator([w_hat], input_is_latent=True, randomize_noise=False, truncation=1))
                
        img_hat = torch.cat(img_hat)

        name = str(i).zfill(5)+'.jpg'
        save_path = os.path.join(opts.result_dir,name)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torchvision.utils.save_image(torch.cat([img.detach().cpu(), img_hat.detach().cpu()]), save_path,
                                     normalize=True, scale_each=True, range=(-1, 1),nrow=len(attribute)+1)

if __name__ == '__main__':
    run_latent_inference()
