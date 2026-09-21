import torch
#import HDPNet_model
import dataset
import os
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F
from imageio import imwrite
from tqdm import tqdm
# from methods.SINet_v2_Res2Net_GRA_NCD import Network as SINetV2
from methods.DCTNet import DCTNet
from methods.HDPNet_model import Model
from methods.zoomnet import ZoomNet
from methods.Network import Network as FEDER


if __name__ == '__main__':
    batch_size = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化模型并移动到设备
    net = DCTNet().to(device)
    
    root_path = './'
    ckpt_path = 'training/ckpt_save/model_26_loss_0.34456_best.pth'     # DCTNet的权重

    Dirs = [
        "dataset/TestDataset/CAMO",
        "dataset/TestDataset/COD10K",
        # "dataset/TestDataset/CHAMELEON",
        # "dataset/TestDataset/NC4K"
    ]

    result_save_root = os.path.join(root_path, "results/test_2")
    os.makedirs(result_save_root, exist_ok=True)

    # 加载模型权重
    print(f"Loading checkpoint from {root_path + ckpt_path}")
    pretrained_dict = torch.load(root_path + ckpt_path, map_location=device)
    
    # 根据不同模型处理权重加载
    if isinstance(pretrained_dict, dict) and 'model' in pretrained_dict:
        pretrained_dict = pretrained_dict['model']

    # net.load_state_dict({k.replace('module.',''):v for k,v in torch.load(root_path + ckpt_path).items()})
    net.load_state_dict(pretrained_dict)
    net.eval()

    # 测试每个数据集
    for Dir in Dirs:
        if not os.path.exists(Dir):
            print(f"Warning: Dataset directory {Dir} does not exist! Skipping...")
            continue

        save_path = os.path.join(result_save_root, Dir.split("/")[-1]) # results/CAMO
        os.makedirs(save_path, exist_ok=True)

        Dataset = dataset.TestDataset(Dir, 384) 
        Dataloader = DataLoader(Dataset, batch_size=batch_size, num_workers=0)

        with tqdm(Dataloader, desc=f"Testing on {Dir.split('/')[-1]}") as tepoch:
            for data in tepoch:
                img = data['img'].to(device)
                label = data['label'].to(device)
                name = os.path.basename(data['name'][0])
                # name = data['name'][0].split("\\")[-1]

                with torch.no_grad():
                    out = net(img)

                    # 处理不同模型的输出格式
                    if isinstance(out, dict):
                        # print("配置字典的所有键：", list(out.keys())) 
                        pred = out['sal']
                        # break
                        # pred = out['pred']
                    elif isinstance(out, (tuple, list)):
                        pred = out[-1]  # 使用最后一个输出
                    else:
                        pred = out  # 直接使用输出

                B, C, H, W = label.size()
                o = F.interpolate(pred, (H, W), mode='bilinear', align_corners=True)
                o = o.detach().cpu().numpy()[0, 0]
                o = (o - o.min()) / (o.max() - o.min() + 1e-8)
                o = (o * 255).astype(np.uint8)     

                save_file = os.path.join(save_path, name) # 可能有混用的\/的情况

                imwrite(save_file, o)

    print("Test finished!")
