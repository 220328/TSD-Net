# MVFNet

## USAGE

> The training and testing experiments are conducted using PyTorch with an NVIDIA 5090.
> Using Git Bash as the terminal.

### 1. Preparation

> Note that MVFNet is implemented with VSCode on Windows11.

- Creating a virtual environment in terminal: `python3 -m venv .venv `
- Activatiing the virtual environment: source `source .venv/Scripts/activate`
- Installing necessary packages: `pip install -r Requirements.txt`

### 2. Downloading Training and Testing Datasets

- Download the necessary datasets in "/datasets/TestDataset" or "/datasets/TrainDatasets". CAMO(250) is already included in TestDataset.

### 3. Training Config

- The pretrained model(pvt2) is stored in "/data/pretrain/pvt_v2_b4.pth"
- Run in terminal: `python3 train.py`
- Training weight is stored in "/training/ckpt_save"

### 4. Testing Config

- The best weight is stored in "/training/ckpt_save/model_26_loss_0.34456_best.pth"
- You can modify the `Dirs=[]` in `test.py` to change the test dataset.
- Run in terminal: `python3 test.py`. Testing results are stored in "results/YOUR_TEST_DATASET_NAME/"

### 5. Evaluation

> Note that before evaluating, testing is necessary.
- You can modify the Dict in Line 14 of `eval.py` to change the eval dataset.
- You can evaluate the results by running: `python3 eval.py`. The results will be saved in "/results/.txt"
