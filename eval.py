import os
import cv2
from tqdm import tqdm
from py_sod_metrics import MAE, Emeasure, Fmeasure, Smeasure, WeightedFmeasure

# method = 'HDPNet'
# method = 'SINetV2'
method = 'DCTNet'
# ckpt_save = "ckpt_save/model_108_loss_0.02126.pth"  # 替换为实际路径
# ckpt_save = "ckpt_save/SINet_v2_epoch_best.pth"
ckpt_save = 'checkpoints/v2/epoch_30_bestEpoch_30_loss_1.47634.pth'     # DCTNet的权重

for _data_name in ['CAMO', 'COD10K', 'NC4K', 'CHAMELEON']:
# for _data_name in ['CHAMELEON']:
    print("eval-dataset: {}".format(_data_name))
    mask_root = 'dataset/TestDataset/{}/{}/'.format(_data_name, "GT") # change path
    pred_root = 'results/test_2/{}/'.format(_data_name) # change path
    pred_name_list = sorted(os.listdir(pred_root))
    FM = Fmeasure()
    WFM = WeightedFmeasure()
    SM = Smeasure()
    EM = Emeasure()
    M = MAE()
    for pred_name in tqdm(pred_name_list, total=len(pred_name_list)):
        mask_path = os.path.join(mask_root, pred_name)
        pred_path = os.path.join(pred_root, pred_name)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        FM.step(pred=pred, gt=mask)
        WFM.step(pred=pred, gt=mask)
        SM.step(pred=pred, gt=mask)
        EM.step(pred=pred, gt=mask)
        M.step(pred=pred, gt=mask)

    fm = FM.get_results()["fm"]
    wfm = WFM.get_results()["wfm"]
    sm = SM.get_results()["sm"]
    em = EM.get_results()["em"]
    mae = M.get_results()["mae"]

    results = {
        "Smeasure": sm, 
        "wFmeasure": wfm,
        "adpEm": em["adp"],
        "meanEm": em["curve"].mean(),
        "maxEm": em["curve"].max(),
        "adpFm": fm["adp"],
        "meanFm": fm["curve"].mean(),
        "maxFm": fm["curve"].max(),
        "MAE": mae,
    }

    print(results)
    # 确保results目录存在
    os.makedirs("results", exist_ok=True)
    file = open("results/test_CHAMELEON/CHAMELEON.txt", "a")
    file.write( '\n' + method + ' ' + str(ckpt_save)  + ' ' + 'DataSet: ' + _data_name  + '\n' + str(results))

print("Eval finished!")
