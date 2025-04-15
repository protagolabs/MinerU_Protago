# MinerU Protago

## Overview

This repository contains the code for the MinerU Protago project, which is adopted from the [MinerU](https://github.com/opendatalab/MinerU) project. In this project, we aim to extract the tables from the PDF documents from Orbit dataset and convert them into the HTML format, which can be used for the further processing. I personally use different python environments for MinerU and PaddleX.

## Installation

please check the [MinerU](https://github.com/opendatalab/MinerU) for the detailed installation. We use the this [tutorial](https://github.com/opendatalab/MinerU/blob/master/docs/README_Ubuntu_CUDA_Acceleration_en_US.md) to install the environment.

### Install Applications
In this project, I use the `MinerU==1.3.3`

```
conda create -n minerup133 python=3.10
conda activate minerup133
pip install -U magic-pdf[full]==1.3.3 --extra-index-url https://wheels.myhloli.com
magic-pdf --version # check the version, should be 1.3.3

```

### Download Models

```
pip install huggingface_hub
python scripts/download_models_hf.py
```
all the downloaded models will be saved in the `models` directory.

### Understand the Location of the Configuration File
After completing the "Download Models" step, the script will automatically generate a `magic-pdf.json` file in the user directory and configure the default model path. You can find the `magic-pdf.json` file in your user directory.
> [!TIP]
> The user directory for Linux is "/home/username".

> You need to check the `magic-pdf.json` file to use the models in the right `models` directory.


### First Run

Download a sample file from the repository and test it.

```sh
magic-pdf -p demo/small_ocr.pdf -o ./output
```

###Test CUDA Acceleration

If your graphics card has at least **8GB** of VRAM, follow these steps to test CUDA acceleration:

1. Modify the value of `"device-mode"` in the `magic-pdf.json` configuration file located in your home directory.
   ```json
   {
     "device-mode": "cuda"
   }
   ```
2. Test CUDA acceleration with the following command:
   ```sh
   magic-pdf -p small_ocr.pdf -o ./output
   ```

### Enable CUDA Acceleration for OCR (This step is deprecated, if you use the `MinerU>=1.3.0`.)


1. Download `paddlepaddle-gpu`. Installation will automatically enable OCR acceleration.
   ```sh
   python -m pip install paddlepaddle-gpu==3.0.0rc1 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
   ```
2. Test OCR acceleration with the following command:
   ```sh
   magic-pdf -p small_ocr.pdf -o ./output
   ```

## Data Preparation

### Orbit Dataset



The Orbit dataset is a collection of PDF documents with tables. There are two versions, one is a small version with 176 PDF documents, and the other is the larger version with 1000 PDF documents. All the code is tested on the small version.

You can download the small version dataset from [here](https://orbit-common-resources.s3.us-west-2.amazonaws.com/all_file_type_case/export_pdf.zip) and large one from [here](https://orbit-common-resources.s3.us-west-2.amazonaws.com/data4demo/pdf2.zip).


After downloading the dataset, you can unzip the files and put them in the `inputs/export_pdf` directory. It should contains the following files:

```
inputs/export_pdf/
├── pdf
│   ├── f_0AibR1dz.pdf
│   ├── ...
├── azure_pkl
│   ├── f_0AibR1dz.pkl
│   ├── ...
├── azure_pages
│   ├── f_0AibR1dz.pages.txt
│   ├── ... 
├── azure_blocks
│   ├── f_0AibR1dz.blocks.txt
│   ├── ...
├── index.xlsx
```
### Table Extraction

After downloading the dataset, you can run the following command to extract the tables from the PDF documents.

```bash
prepare.sh
```

In particular,  You can check the `prepare.sh` file to see the detailed command. After running the script, you will get the following files:

```
inputs/export_pdf
├── pdf
├── azure_pkl
├── azure_pages
├── azure_blocks
├── tables_from_pkl # tables extracted from pkl files
│   ├── f_0AibR1dz
│   │   ├── f_0AibR1dz.json
│   │   ├── images
│   │       ├── table_0_page_1.png
│   │       ├── ...
│   ├── ...
├── index.xlsx
outputs/orbit_data
├── images
│   ├── f_0AibR1dz_table_0_page_1.png
│   ├── ...
├── f_0AibR1dz_tables.json
├── ...
```

## Training and Inference

In this section, we will use the extracted data to fine-tune the [slanet_plus](https://arxiv.org/pdf/2210.05391) by using [PaddleX](https://github.com/PaddlePaddle/PaddleX). But before that, make sure you have the folder `images` and  file `final_output.txt` in the `outputs/orbit_data` directory.


```bash
cd PaddleX
```

### Install PaddleX

I follow the 2.2 自定义方式安装PaddleX [here](https://paddlepaddle.github.io/PaddleX/latest/installation/installation.html#21-dockerpaddlex) for detailed steps. In particular, I choose the `paddlex --install` option to install all the plugins. More details can be found in the [PaddleX](https://github.com/PaddlePaddle/PaddleX). 

### Fine-tune the slanet_plus

First of all, please copy the `final_output.txt` file and `images` folder to the `PaddleX/dataset/orbit_data_v1` directory. Then run the following command to fine-tune the slanet_plus:

```bash
cp -r  ../outputs/orbit_data_words/images dataset/orbit_data_v1/

cp ../outputs/orbit_data_words/final_output.txt  dataset/orbit_data_v1/

run.sh
```

More details can be found in the comments of the `run.sh` file. Also, the paddleX [documents](https://github.com/PaddlePaddle/PaddleX/blob/release/3.0-beta1/docs/module_usage/tutorials/ocr_modules/table_structure_recognition.md) is very helpful for the fine-tuning.














# Reference

[1] MinerU:https://github.com/opendatalab/MinerU

[2] SlANET https://arxiv.org/pdf/2210.05391 

[3] PaddleX: https://github.com/PaddlePaddle/PaddleX