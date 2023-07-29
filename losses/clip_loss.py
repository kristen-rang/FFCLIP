import torch
import clip


model_path = '../pretrained_models/ViT-B-32.pt'
class CLIPLoss(torch.nn.Module):
    def __init__(self):
        super(CLIPLoss, self).__init__()
        self.upsample = torch.nn.Upsample(scale_factor=7)
        self.avg_pool = torch.nn.AvgPool2d(kernel_size=32)  # 8
        self.model, self.preprocess = clip.load(model_path, device='cuda') 

    def forward(self, image, text):
        device = "cuda" if torch.cuda.is_available() else "cpu" 
        self.model = self.model.eval().to(device)
        self.model = self.model.float()
        image = image.to(device)
        text = text.to(device)
        image = self.avg_pool(self.upsample(image))
        similarity = (1 - self.model(image, text)[0] / 100).mean()
        return similarity.float()


