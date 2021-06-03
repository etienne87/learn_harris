import os
import cv2
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import kornia
import tqdm

import torchvision
from torchvision.models.resnet import BasicBlock


def loader(im_path, w, h):
    im = cv2.imread(im_path)
    # assert im is not None, im_path
    im = cv2.resize(im, (w,h), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(im).permute(2,0,1)


def normalize_tiles(tensor, num_stds=6, num_dims=2, real_min_max=True):
    shape = tensor.shape[:-num_dims]
    trflat = tensor.view(*shape, -1)
    mu, std = trflat.mean(dim=-1), trflat.std(dim=-1)[0] * num_stds
    mu = mu[(...,) + (None,) * num_dims]
    std = std[(...,) + (None,) * num_dims]
    low, high = mu - std, mu + std
    tensor = torch.min(tensor, high)
    tensor = torch.max(tensor, low)
    if real_min_max:
        trflat = tensor.view(*shape, -1)
        low, high = trflat.min(dim=-1)[0], trflat.max(dim=-1)[0]
        low = low[(...,) + (None,) * num_dims]
        high = high[(...,) + (None,) * num_dims]

    return (tensor - low) / (high - low + 1e-5)

def make_grid(x):
    batch_size = len(x)
    nrows = 2 ** ((batch_size.bit_length() - 1) // 2)
    return torchvision.utils.make_grid(x,nrow=nrows).detach().cpu().permute(1, 2, 0).numpy().astype(np.uint8)


def corner_layer(input_data, k=0.04, input_is_image=False, debug=False):
    """
    - https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/
      py_features_harris/py_features_harris.html#harris-corners
    - https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/
      py_shi_tomasi/py_shi_tomasi.html#shi-tomasi

    - http://math.colgate.edu/~wweckesser/math312Spring06/handouts/IMM_2x2linalg.pdf
    - https://en.wikipedia.org/wiki/Quadratic_equation
    """

    def g(x):
        return kornia.filters.gaussian_blur2d(x, (7, 7), (1.0, 1.0))
    if input_is_image:
        gradients = kornia.filters.spatial_gradient(input_data, 'sobel')
        dx = gradients[:, :, 0]
        dy = gradients[:, :, 1]
    else:
        dx = input_data[:, 0:1]
        dy = input_data[:, 1:2]

    dx2: torch.Tensor = g(dx ** 2)
    dy2: torch.Tensor = g(dy ** 2)
    dxy: torch.Tensor = g(dx * dy)

    det_m: torch.Tensor = dx2 * dy2 - dxy * dxy
    trace_m: torch.Tensor = dx2 + dy2
    # harris
    scores: torch.Tensor = det_m - k * (trace_m ** 2)
    return scores.float()


class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2,reduction='mean'):
        super(FocalLoss, self).__init__(weight,reduction=reduction)
        self.gamma = gamma
        self.weight = weight #weight parameter will act as the alpha parameter to balance class weights

    def forward(self, input, target):

        ce_loss = F.cross_entropy(input, target,reduction=self.reduction,weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss





def main(path, lr=0.001, save_path='harris.ckpt', batch_size=32, viz_batch_size=8, height=128, width=128, epochs=10, hidden=16, num_workers=2, device='cuda:0', resume=True):
    dataset = torchvision.datasets.ImageFolder(path, loader=lambda path:loader(path, width, height))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size, num_workers=num_workers)

    hidden=16
    net = nn.Sequential(
            nn.Conv2d(1,hidden,3,1,1),
            BasicBlock(hidden,hidden,1, norm_layer=nn.BatchNorm2d),
            BasicBlock(hidden,hidden,1, norm_layer=nn.BatchNorm2d),
            nn.Conv2d(hidden,4,3,1,1)
            )

    #net = nn.Conv2d(1,2,3,1,1)
    net.to(device)

    if os.path.exists(save_path) and resume:
        net.load_state_dict(torch.load(save_path))

    optim = torch.optim.AdamW(net.parameters(), lr)

    criterion = FocalLoss(gamma=2.0)

    for epoch in range(epochs):
        with tqdm.tqdm(dataloader, total=len(dataloader)) as tq:
            for i, (x, _) in enumerate(tq):
                x = x.to(device)
                x = x.float()/255.0
                x = x.mean(dim=1).unsqueeze(1)
                b,c,h,w = x.shape

                gradients = kornia.filters.spatial_gradient(x, 'sobel').squeeze(1)
                t = corner_layer(gradients)

                optim.zero_grad()
                # experiment 1: we train sobel & apply harris response
                #grad = net(x)
                #grad = (grad-grad.min())/(grad.max()-grad.min()) - 0.5
                #print('range: ', grad.min().item(), grad.max().item())
                #y = corner_layer(grad)
                #loss = nn.functional.l1_loss(grad,gradients)
                #loss = nn.functional.l1_loss(y,t, reduce=False)
                #loss = loss.mean()
                #loss = loss[bin_mask].mean()


                # experiment 2: learn binary mask directly
                # bin_mask = (t > 6e-5)
                # target = bin_mask.squeeze().long()
                # logits = net(x)
                # loss = kornia.losses.binary_focal_loss_with_logits(logits, target, alpha=0.5, gamma=2.0, reduction='mean')
                # y = torch.sigmoid(logits)

                # experiment 3: learn quantization
                tsq = t.squeeze()
                target = torch.zeros((len(t),height,width), dtype=torch.long, device=device)
                target[tsq>6e-5]=1
                target[tsq>6e-4]=2
                target[tsq>6e-3]=3

                logits = net(x)
                loss = criterion(logits, target)

                #predict
                y = F.softmax(logits, dim=1)
                y = y[:,1]*6e-5 + y[:,2]*6e-4 + y[:,3]*6e-3
                y = y.unsqueeze(1)

                loss.backward()
                optim.step()

                tq.set_description('\rtrain_loss: %.11f'%loss.item())

                if i % 100 == 0:
                    #visualize a few of them
                    x = x[:viz_batch_size]
                    y = y[:viz_batch_size]
                    t = t[:viz_batch_size]

                    im_x = make_grid(x*255)

                    # to visualize harris-response
                    viz_t = normalize_tiles(t, 3)
                    viz_t = make_grid(viz_t*255)
                    viz_y = normalize_tiles(y, 3)
                    viz_y = make_grid(viz_y*255)

                    im_x[...,2] = np.maximum(im_x[...,2], viz_t[...,2])
                    im_x[...,1] = np.maximum(im_x[...,1], viz_y[...,1])

                    cv2.imshow('result', im_x)
                    cv2.waitKey(5)

        torch.save(net.state_dict(), save_path)


if __name__ == '__main__':
    import fire;fire.Fire(main)
