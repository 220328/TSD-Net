# Transformation-Induced Relative Spectral Discrepancy Learning for Camouflaged Object Detection

## Usage
The training and testing experiments are implemented with PyTorch and tested on NVIDIA RTX 5090.

### 1. Preparation
The project is developed using VSCode.
- Install required dependencies: 
```bash
pip install -r Requirements.txt
```
### 2. Downloading Training and Testing Datasets

- Download the training and testing datasets and place them under `/datasets/TestDataset` and `/datasets/TrainDatasets`.

### 3. Training Config

Pre-trained PVTv2 weights are located at `/data/pretrain/pvt_v2_b4.pth`
- Run the training script: 
```bash
python train.py
```
Checkpoints will be saved to `/training/ckpt_save/`.

### 4. Testing Config
- Modify `Dirs=[]` in `test.py` to switch between different test datasets.
- Run the testing script:
```bash
python test.py
```
Prediction results will be saved to `results/YOUR_TEST_DATASET_NAME/`.

### 5. Evaluation

Note that before evaluating, testing is necessary.
Run the evaluation script:
```bash
python eval.py
```
Quantitative evaluation metrics will be saved as a `.txt` file under `/results/`.