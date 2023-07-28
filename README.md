<meta name="robots" content="noindex">


# SIGMOD 2023 Artifact Submission - Paper ID: V1mod125

Welcome to the artifact documentation for our paper, **LightCTS: A Lightweight Framework for Correlated Time Series Forecasting**. This documentation outlines the steps required to reproduce our work.

**Authors:** Zhichen Lai, Dalin Zhang*, Huan Li*, Christian S. Jensen, Hua Lu, Yan Zhao

**Paper Link:** [LightCTS: A Lightweight Framework for Correlated Time Series Forecasting](https://dl.acm.org/doi/10.1145/3589270)

## Hardware information
We trained all models on a server with an NVIDIA Tesla P100 GPU. Additionally, we conducted some inference experiments on an x86 device with a 380 MHz CPU to emulate resource-restricted environments.

## Library Dependencies
We developed the code for experiments using Python 3.7.13 and PyTorch 1.13.0. You can install PyTorch following the instructions on the [PyTorch website](https://pytorch.org/get-started/locally/), tailored to your specific operating system, CUDA version, and computing platform. For example:

```bash
pip install torch==1.13.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
```

After successfully installing PyTorch, you can install the remaining dependencies using:

```bash
pip install -r requirements.txt
```

Please note, if you encounter issues installing `fvcore` directly using pip, you can install it from its GitHub repository using:

```bash
pip install git+https://github.com/facebookresearch/fvcore.git
```

## Dataset Preparation
We tested LightCTS on four public multi-step correlated time series forecasting datasets and two public single-step correlated time series forecasting datasets.

**Multi-Step Datasets:**

| Dataset  | Data Type   | Download Link |
|----------|-------------|--------------------|
| PEMS04   | Traffic Flow| [download](https://drive.google.com/drive/folders/154SVCXjnVj0983zlqzUQMRpAKfmYZwON?usp=drive_link) |
| PEMS08   | Traffic Flow| [download](https://drive.google.com/drive/folders/1U6wui2D1i7iw0vCAyVf5HizYMrg24wYw?usp=drive_link) |
| METR-LA  | Traffic Speed| [download](https://drive.google.com/drive/folders/1U4iuH7o_wunTD3oU1KSLT3Z3A1sE9jI0?usp=drive_link) |
| PEMS-BAY | Traffic Speed| [download](https://drive.google.com/drive/folders/1gSlL4cqmQ_9av7y5BJP-fzEPuoN7hQ9s?usp=drive_link) |

**Single-Step Datasets:**

| Dataset   | Data Type               | Download Link |
|-----------|-------------------------|--------------------|
| Solar     | Solar Power Production  | [download](https://drive.google.com/file/d/1TP6xDPXmf923YdhdRPdD1VwqtLATQqkS/view?usp=drive_link) |
| Electricity| Electricity Consumption | [download](https://drive.google.com/file/d/1x9nBW-RAXrubHCWeG6JLMtUaXl2iQznX/view?usp=drive_link) |

To download all the datasets in one run, please follow these instructions:
Install the download library `gdown`:

```bash
pip install gdown
```
Run the script to download all the datasets:
```bash
python data_downloading.py
```

After downloading the datasets, move them to the '\data' directory. The directory structure should appear as follows:

```text
data
   ├─METR-LA
   ├─PEMS-BAY
   ├─PEMS04
   ├─PEMS08
   ├─solar.txt
   ├─electricity.txt
```

## Reproducing Experiments

This section provides detailed steps to reproduce the multi-step and single-step forecasting experiments from our paper.

### Multi-Step Forecasting Experiments

#### Traffic Flow Forecasting Experiments
To replicate the multi-step traffic flow forecasting experiments presented in Table 5 of the paper, follow these instructions:

##### Model Training
```bash
python Multi-step/Traffic Flow/$DATASET_NAME/train_$DATASET_NAME.py --device='cuda:0'
#python Multi-step/Traffic Flow/PEMS04/train_PEMS04.py --device='cuda:0'
#python Multi-step/Traffic Flow/PEMS08/train_PEMS08.py --device='cpu'
```

After the training phase concludes, a log summarizing the best model's performance on the test set will appear:

```bash
On average: Test MAE: ..., Test MAPE: ..., Test RMSE: ...
```

##### Model Testing
```bash
python Multi-step/Traffic Flow/$DATASET_NAME/test_$DATASET_NAME.py --device='cuda:0' --checkpoint=$CKPT_PATH
#python Multi-step/Traffic Flow/PEMS04/test_PEMS04.py --device='cuda:0' --checkpoint='./checkpoint.pth'
#python Multi-step/Traffic Flow/PEMS08/test_PEMS08.py --device='cpu' --checkpoint='./checkpoint.pth'
```

After the testing phase concludes, a log summarizing the tested model's performance will appear:

```bash
On average: Test MAE: ..., Test MAPE: ..., Test RMSE: ...
```

##### Computing Lightness Metrics
```bash
python Multi-step/Traffic Flow/$DATASET_NAME/lightness_metrics_$DATASET_NAME.py
#python Multi-step/Traffic Flow/PEMS04/lightness_metrics_PEMS04.py
#python Multi-step/Traffic Flow/PEMS08/lightness_metrics_PEMS08.py
```

Upon completion, a log like the foolowing one will display the number of parameters and FLOPs:

```bash
| module                                                   | #parameters or shape   | #flops    |
|:-------------------------------------------------------- |:-----------------------|:----------|
| model                                                    | 0.185M                 | 0.147G    |
|  Filter_Convs                                            |  8.448K                |  23.892M  |
|   Filter_Convs.0                                         |   2.112K               |   9.431M  |
|    Filter_Convs.0.weight                                 |    (64, 16, 1, 2)      |           |
|    Filter_Convs.0.bias                                   |    (64,)               |           |

...
```

In the above commands, replace `$DATASET_NAME` with either `PEMS04` or `PEMS08`, and `$CKPT_PATH` with the path to the desired saved checkpoint. Adjust the `--device` option in the command line according to your available hardware. Note that the lightness metrics calculation only runs on CPU, so we set the device to 'cpu' by default.

### Traffic Speed Forecasting Experiments
To reproduce the multi-step traffic speed forecasting experiments presented in Table 6 of the paper, follow these instructions:

##### Model Training
```bash
python Multi-step/Traffic Speed/$DATASET_NAME/train_$DATASET_NAME.py --device='cuda:0'
#python Multi-step/Traffic Speed/METR-LA/train_METR-LA.py --device='cuda:0'
#python Multi-step/Traffic Speed/PEMS-BAY/train_PEMS-BAY.py --device='cpu'
```

After the training phase concludes, a log summarizing the best model's performance on the test set will appear:

```bash
Horizon 1, Test MAE: ..., Test MAPE: ..., Test RMSE: ...
Horizon 2, Test MAE: ..., Test MAPE: ..., Test RMSE: ...
Horizon 3, Test MAE: ..., Test MAPE: ..., Test RMSE: ...
...
...
Horizon 12, Test MAE: ..., Test MAPE: ..., Test RMSE: ...
On average: Test MAE: ..., Test MAPE: ..., Test RMSE: ...
```

Here, Horizon 3, Horizon 6, and Horizon 12 correspond to '15 mins', '30 mins', and '60 mins' in Table 6, respectively.

##### Model Testing
```bash
python Multi-step/Traffic Speed/$DATASET_NAME/test_$DATASET_NAME.py --device='cuda:0' --checkpoint=$CKPT_PATH
#python Multi-step/Traffic Speed/METR-LA/test_METR-LA.py --device='cuda:0' --checkpoint='./checkpoint.pth'
#python Multi-step/Traffic Speed/PEMS-BAY/test_PEMS-BAY.py --device='cuda:0' --checkpoint='./checkpoint.pth'
```

After the testing phase concludes, a log summarizing the tested model's performance will appear:

```bash
Horizon 1, Test MAE: ..., Test MAPE: ..., Test RMSE: ...
Horizon 2, Test MAE: ..., Test MAPE: ..., Test RMSE: ...
Horizon 3, Test MAE: ..., Test MAPE: ..., Test RMSE: ...
...
...
Horizon 12, Test MAE: ..., Test MAPE: ..., Test RMSE: ...
On average: Test MAE: ..., Test MAPE: ..., Test RMSE: ...
```

Here, Horizon 3, Horizon 6, and Horizon 12 correspond to '15 mins', '30 mins', and '60 mins' in Table 6, respectively.

##### Computing Lightness Metrics
```bash
python Multi-step/Traffic Speed/$DATASET_NAME/lightness_metrics__$DATASET_NAME.py
#python Multi-step/Traffic Speed/METR-LA/lightness_metrics_METR-LA.py
#python Multi-step/Traffic Speed/PEMS-BAY/lightness_metrics_PEMS-BAY.py
```
A similar log like the above traffic flow forecasting's lightness metrics will appear.

In the above commands, replace `$DATASET_NAME` with either `METR-LA` or `PEMS-BAY`, and `$CKPT_PATH` with the path to the desired saved checkpoint. Update `--device` in the command line according to your available hardware.

### Single-Step Forecasting Experiments

To replicate the single-step forecasting experiments presented in Table 7 of the paper, follow these instructions:

##### Model Training
```bash
python Single-step/$DATASET_NAME/train_$DATASET_NAME.py --horizon=3 --device='cuda:0'
#python Single-step/Solar/train_Solar.py --horizon=3 --device='cuda:0'
#python Single-step/Electricity/train_Electricity.py --horizon=3 --device='cpu'
```

After the training phase concludes, a log summarizing the best model's performance on the test set will appear:

```bash
On average: Test RRSE: ..., Test CORR ...
```

##### Model Testing
```bash
python Single-step/$DATASET_NAME/test_$DATASET_NAME.py --horizon=3 --device='cuda:0' --checkpoint=$CKPT_PATH
#python Single-step/Solar/test_Solar.py --horizon=3 --device='cuda:0' --checkpoint='./save.pt'
#python Single-step/Electricity/test_Electricity.py --horizon=3 --device='cpu' --checkpoint='./save.pt'
```

After the testing phase concludes, a log summarizing the tested model's performance will appear:

```bash
On average: Test RRSE: ..., Test CORR ...
```

##### Computing Lightness Metrics
```bash
python Single-step/$DATASET_NAME/test_$DATASET_NAME.py
#python Single-step/Solar/lightness_metrics_Solar.py
#python Single-step/Electricity/lightness_metrics_Electricity.py
```

A similar log like the above traffic flow forecasting's lightness metrics will appear.

In the above commands, replace `$DATASET_NAME` with either `Solar` or `Electricity`, and `$CKPT_PATH` with the path to the desired saved checkpoint. Update `--horizon` to the target future horizons, which are [3, 6, 12, 24] in Table 7, and update  `--device` in the command line to suit your hardware.

Please note that during the training process, the saved checkpoints will be stored in the '/logs' directory within each dataset's directory.

## Figure Drawing
After gathering the metrics results of each dataset, you can follow the instructions to draw the figures in the paper.
First, please install the library, Matplotlib, for figure drawing:
```bash
pip install matplotlib
```
Then, move to the `Figure_drawing`, modify the metrics results in the file, and then run the code to draw the figure.
```bash
python Figure_drawing/Figure_$FIGURE_$SUBFIGURE_drawing.py
#python Figure_drawing/Figure_5_a_drawing.py
#python Figure_drawing/Figure_6_b_drawing.py
```
In the above commands, replace `$FIGURE` and `$SUBFIGURE` with the number of Figure and Subfigure.

## Pretrained Models
If you prefer to skip the training process and directly access the pre-trained checkpoints for reproduction, we have provided a version of the codebase [here](https://github.com/AI4CTS/lightcts). This version is optimized for better execution of the pre-trained checkpoints.

## Contact
For any inquiries, please reach out to Zhichen Lai at zhla@cs.aau.dk.
