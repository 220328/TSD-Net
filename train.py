from torch.utils.data import DataLoader
import os
import argparse
import torch
import time
import dataset
import loss
import pandas as pd
from tqdm import tqdm
from methods.SINet import SINet_ResNet50
from methods.MFFN import MFFN
from methods.zoomnet import ZoomNet
from methods.SINet_v2_Res2Net_GRA_NCD import Network as SINetV2
from methods.HDPNet_model import Model as HDPNet_model
from methods.DCTNet import DCTNet   


def parse_args():
    parser = argparse.ArgumentParser('--HDPNet--')
    parser.add_argument('--base_lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--batch_size_per_gpu', default=1, type=int, help='batch size per GPU')
    parser.add_argument("--resume", default=None)
    parser.add_argument('--gpu', default=0, type=int)
    parser.add_argument('--path', type=str, default="dataset/TrainDataset/", help='path to train dataset')
    parser.add_argument('--pretrain', type=str, default="data/pretrain/pvt_v2_b4.pth", help='path to pretrain model')
    parser.add_argument('--ft_for_MoCA', default=None, type=str, help='path to pretrain model')
    
    # 学习率调度配置
    parser.add_argument('--scheduler_type', type=str, default='warmup_cosine', 
                        choices=['warmup_cosine', 'warmup_restart', 'poly'],
                        help='learning rate scheduler type')
    parser.add_argument('--warmup_epochs', default=5, type=int, help='warmup epochs')
    parser.add_argument('--use_layer_decay', action='store_true', help='use layer-wise learning rate decay')
    parser.add_argument('--backbone_lr_ratio', default=0.1, type=float, help='backbone lr ratio relative to head')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int, help='gradient accumulation steps')

    # DDP configs:
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--epochs', default=30, type=int,
                        help='number of training epochs')
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    args = parser.parse_args()
    return args


def main(args):
    ### model ###
    net = DCTNet()                 

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 自动选择设备
    # 检查是否有可用的GPU
    if torch.cuda.is_available():
        print(f"使用GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU数量: {torch.cuda.device_count()}")
    else:
        print("使用CPU")

    # device = torch.device('cuda:0')
    print(f"cuda:{device}")
    net.to(device)
    
    # ============ 简化的优化器和学习率调度器 ============
    print(f"\n========== 训练配置 ==========")
    print(f"优化器: Adam")
    print(f"初始学习率: {args.base_lr}")
    print(f"权重衰减: {args.weight_decay}")
    print("=" * 30 + "\n")
    
    # 使用简化的Adam优化器
    optimizer = torch.optim.Adam(
        net.parameters(), 
        lr=args.base_lr
        # weight_decay=args.weight_decay
    )
    
    # 简化的学习率调度器：每50个epoch学习率减半
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=50,  # 每50个epoch
        gamma=0.5      # 学习率减半
    )


    ### data ###
    Dir = [args.path]
    Dataset = dataset.TrainDataset(Dir)
    Dataloader = DataLoader(Dataset, 
                          batch_size=args.batch_size_per_gpu, 
                          num_workers=1,  # 减少worker数量
                          collate_fn=dataset.my_collate_fn, 
                          drop_last=True, 
                          shuffle=True,
                          pin_memory=False)  # 关闭pin_memory以减少内存使用

    # torch.backends.cudnn.benchmark = True

    ### main loop ###
    # scheduler.last_epoch = - 1  # do not move
    epochs = args.epochs
    loss_result_curve = []
    t1 = time.time()
    best_epoch = 0
    best_loss_all = float('inf')  # 初始化为正无穷，确保第一个epoch的loss会被保存
    best_pth = dict()

    for curr_epoch in range(0, epochs + 1):
        # if curr_epoch == 50 or curr_epoch == 100:
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = param_group['lr'] * 0.1
        #         print("Learning rate:", param_group['lr'])

        net.train()

        # 打印当前学习率
        current_lrs = [param_group['lr'] for param_group in optimizer.param_groups]
        if len(current_lrs) == 1:
            print(f"Epoch {curr_epoch + 1}/{epochs} | LR: {current_lrs[0]:.2e}")
        else:
            print(f"Epoch {curr_epoch + 1}/{epochs} | LR: {current_lrs[0]:.2e} (head), {current_lrs[1]:.2e} (backbone)")
        running_loss_all, running_loss_m = 0., 0.
        count = 0
        with tqdm(Dataloader, unit="batch") as tepoch:
            for data in tepoch:
                tepoch.set_description(f"Epoch {curr_epoch + 1}")
                count += 1
                img = data['img'].to(device)
                label = data['label'].to(device)
                out = net(img)  # [b,1,384,384]

                # 根据模型类型处理输出
                if isinstance(out, dict):
                    # 字典类型输出（如HDPNet） only 'pred'
                    pred = out.get('pred', out.get('sal', list(out.values())[0]))
                elif isinstance(out, tuple):
                    # 元组类型输出
                    pred = out
                    # pred1 = out[0]  # 可能的辅助输出
                    # pred2 = out[1]
                else:
                    # 张量类型输出（如MFFN）
                    # pred = out[2]
                    pred = out

                # 调整预测输出大小以匹配标签
                if pred[-1].shape != label.shape:
                    # pred = torch.nn.functional.interpolate(pred, size=(label.shape[2], label.shape[3]), mode='bilinear', align_corners=True)
                    pred[-1] = torch.nn.functional.interpolate(pred[-1], size=(label.shape[2], label.shape[3]), mode='bilinear', align_corners=True)
                    pred[-2] = torch.nn.functional.interpolate(pred[-2], size=(label.shape[2], label.shape[3]), mode='bilinear', align_corners=True)

                # loss_initial = loss.structure_loss(pred1, label) + loss.structure_loss(pred2, label)  # 辅助损失
                all_loss, loss_m = loss.mse_structure_loss(pred[-1], label)
                # all_loss = all_loss + loss_initial * 0.5 # 结合主损失和辅助损失，调整权重以平衡两者的影响
                all_loss = loss.mse_loss(pred[0], label) + loss.mse_loss(pred[1], label) + \
                           loss.mse_loss(pred[2], label) + loss.mse_loss(pred[3], label) + all_loss # 

                optimizer.zero_grad()
                all_loss.backward()
                optimizer.step()

                running_loss_all += all_loss.item()
                running_loss_m += loss_m.item()

                # 清理内存
                del out, all_loss, loss_m
                torch.cuda.empty_cache()

                tepoch.set_postfix(loss_all=running_loss_all / count, loss_main=running_loss_m / count)

        # 更新最佳权重：如果当前epoch的loss更小，则更新
        if running_loss_all <= best_loss_all:
            best_loss_all = running_loss_all
            best_epoch = curr_epoch
            best_pth = net.state_dict()

        # 从第0轮开始，每5轮保存一次最佳权重
        if curr_epoch % 2 == 0:
            ckpt_save_root = "./training/ckpt_save"
            if not os.path.exists(ckpt_save_root):
                os.mkdir(ckpt_save_root)
            torch.save(best_pth,
                      os.path.join(ckpt_save_root, f"model_{best_epoch}_loss_{best_loss_all / count:.5f}.pth"))
            # 为下一个5周期的比较重置最佳loss，确保每个周期都能正确追踪最小值
            print(f"✓ Saved checkpoint: model_{best_epoch}_loss_{best_loss_all / count:.5f}.pth")
            best_loss_all = float('inf')

        loss_result_curve.append(running_loss_all / count)

        scheduler.step()
    
    t2 = time.time()
    total_time = t2 - t1
    print('training time: {:0>2}:{:0>2}\n'.format(int(total_time // 3600), int((total_time % 3600) // 60)))
    print(f'best training weight: model_{best_epoch}_loss_{best_loss_all / count:.5f}.pth')

    # 保存损失曲线
    df = pd.DataFrame(loss_result_curve)
    df.to_excel(os.path.join(ckpt_save_root, f'model_{epochs}-Epochs.xlsx'))


if __name__ == '__main__':
    args = parse_args()
    main(args)
