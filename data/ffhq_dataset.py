from torch.utils.data import Dataset
import numpy as np
import torch
import random
import os
import clip
import cv2


class StyleDataset(Dataset):
    def __init__(self, opts, status='train'):
        self.opts = opts
        self.status = status
        self.emotion_list = [ 'Sad', 'Surprise', 'Fear',  'Contempt']*2+['Neutral', 'happy','Disgust', 'Anger']
        self.age_gender_list = ['Woman Aged ' + i for i in ['40','80','10','20']] + ['Man Aged '+ i for i in ['40','80','10','20']]
        self.hair_color_list = [i + ' hair' for i in ['Purple', 'Green', 'Orange', 'Yellow', 'Blue', 'Gray', 'Brown', 'Black', 'White','Blond', 'Pink']] + ['Beard'] + ['curly hair'] #'Red',
        self.eyes_list = ['Blue eyes','Green eyes'] 
        self.face_list = ['Bald'] + ['make up'] + ['closed eyes'] + ['wearing glasses']
        self.famous_man = ['Taylor Swift']
        self.hair_style_list = [i + ' hairstyle' for i in ['afro', 'bowl cut', 'mohawk', 'jewfro', 'dreadlocks', 'cornrows', 'Hi-top Fade', 'Bob Cut']]
        self.description_list = self.emotion_list + self.age_gender_list  + self.face_list * 2  + self.famous_man * 2  + self.eyes_list+ self.hair_color_list+ self.hair_style_list
        

        if status == 'train':
            self.w = torch.from_numpy(np.load(opts.data.train_latents))
            self.length = self.w.shape[0]

        else:

            self.w = torch.from_numpy(np.load(opts.data.test_latents))
            self.length = self.w.shape[0]

        self.clip_model, _ = clip.load(opts.clip_path, device='cpu')
        self.clip_model = self.clip_model.eval().float()

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        selected_description = np.random.choice(self.description_list)
        with torch.no_grad():
            description = clip.tokenize(selected_description)
            embedding = self.clip_model.encode_text(description)

        return embedding, description.long(), selected_description, self.w[index]


if __name__ == '__main__':
    from omegaconf import OmegaConf

    opts = './configs/edit.yaml'
    args = OmegaConf.load(opts)
    test_dataset = StyleDataset(args, 'test')
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=2)

    for i, batch in enumerate(testloader):
        img, attri, discrip, selected_description = batch
        print(img.shape)
        print(attri.shape)
        print(discrip.shape)
        print(selected_description)
