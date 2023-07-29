import torch
import numpy as np
from torch.utils.data import Dataset
import sys
sys.path.append("..")
import os
import clip
import random

class StyleDataset(Dataset):
    def __init__(self, opts, flag):
        self.opts = opts
        self.flag = flag
        self.color_list = [ i + ' car' for i in ['Red','Blue','Gray','Black','White','silver']]
        self.style_list = ['Jeep','Sports car','From Sixties']
        self.description_list = self.color_list + self.style_list
        
        self.clip_model,_ = clip.load(opts.clip_path,device='cpu')
        self.clip_model = self.clip_model.eval().float()
        
        if flag == 'train':
            train_dir = opts.data.train_dir
            self.latent = torch.from_numpy(np.load(train_dir)).cpu()
            self.length = self.latent.shape[0]

        else:
            test_dir = opts.data.test_dir
            self.latent = torch.from_numpy(np.load(test_dir)).cpu()
            self.length = self.latent.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        selected_description = np.random.choice(self.description_list)
        with torch.no_grad():
            description = clip.tokenize(selected_description)
            embedding = self.clip_model.encode_text(description)

    
        return embedding, description.long(),selected_description,self.latent[item] 


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument('--test_dir', default='/home/wcr/datasets/cars_test')

    args = parser.parse_args()
    train_dataset = StyleDataset(args, 'test')
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=3)
    
    for i, batch in enumerate(trainloader):
         print(batch.shape)
         break

