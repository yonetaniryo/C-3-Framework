from models.CC import CrowdCounter
import torch
from torch.autograd import Variable
from PIL import Image
from glob import glob
import numpy as np
from torchvision import transforms, datasets
import sys
import json
import os
from tqdm import tqdm


def main(params):

    H, W = params['image_size']
    mean_std = ([0.452016860247, 0.447249650955, 0.431981861591],[0.23242045939, 0.224925786257, 0.221840232611])

    data_transform = transforms.Compose([
        transforms.Resize((H, W)),
        transforms.ToTensor(),
        transforms.Normalize(*mean_std)])

    net = CrowdCounter([0], params['model'])
    net.load_state_dict(torch.load(params['model_path']))
    net.cuda()
    net.eval()

    video_list = np.sort(glob(params['dataset_path'] + '/*'))
    for v in video_list:
        print(v)
        outputdir = params['outputdir_prefix'] + '/%d_%d/' % (H, W)
        os.makedirs(outputdir, exist_ok=True)
        file_list = np.sort(glob(v + '/*.jpg'))

        imgs = torch.zeros(len(file_list), 3, H, W)
        for i, f in enumerate(tqdm(file_list)):
            imgs[i] = data_transform(Image.open(f))

        train_dataset = torch.utils.data.TensorDataset(imgs, torch.zeros(len(file_list)))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=False)
        pred_map = []
        for x, y in tqdm(train_loader):
            tmp = net.test_forward(x.cuda()).squeeze().detach().cpu().numpy()
            if(len(tmp.shape) == 2):
                tmp = tmp[np.newaxis]
            pred_map.append(tmp)
        pred_map = np.concatenate(pred_map)
        np.savez_compressed(outputdir + os.path.basename(v), pred_map)


if __name__ == '__main__':
    argv = sys.argv
    with open(argv[1]) as f:
        params = json.load(f)
    main(params)