# MinerU Protago

## Overview

This repository contains the code for the MinerU Protago project, which is adopted from the [MinerU](https://github.com/opendatalab/MinerU) project. In this project, we aim to extract the tables from the PDF documents from Orbit dataset and convert them into the HTML format, which can be used for the further processing. I personally use different python environments for MinerU and PaddleX.

## Dev Log

| Sample Size | Version | TED     | TED structure |
|-------------|---------|---------|----------------|
| ~176 (v1)        | 1.2.2   | 48.36%  | 92.66%         |
| ~176 (v1)        | 1.3.3   | **53.77%** | **92.99%**     |
| ~176 (v1)        | 1.3.3 + vlm   | **62.61** | **84.47%**     |
| ~1063 (v2)       | 1.2.2   | 45.77%  | 92.17%         |
| ~1063 (v2)     | 1.3.3   | **52.97%** | **92.39%**     |





## Installation

please check the [MinerU](https://github.com/opendatalab/MinerU) for the detailed installation. We use the this [tutorial](https://github.com/opendatalab/MinerU/blob/master/docs/README_Ubuntu_CUDA_Acceleration_en_US.md) to install the environment.

### Install Applications
In this project, I use the `MinerU==1.3.3`

```
conda create -n minerup133 python=3.10 -y
conda activate minerup133
pip install -e ".[full]"
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
> The user directory for Linux is "/home/username". For Windows, it is "C:\Users\username".
> You need to check the `magic-pdf.json` file to use the models in the right `models` directory.


### First Run

Download a sample file from the repository and test it.

```sh
magic-pdf -p demo/pdfs/small_ocr.pdf -o ./output
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
   magic-pdf -p demo/pdfs/small_ocr.pdf -o ./output
   ```

If you meet the error " raise AssertionError("Torch not compiled with CUDA enabled") ", you can try the following command to install the torch and torchvision:

```
pip install torch==2.6.0 torchvision==0.21.0 --extra-index-url https://download.pytorch.org/whl/cu118
```


### Enable CUDA Acceleration for OCR (Deprecated if you use the MinerU>=1.3.0)


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

You can download the datasets from Google Drive (requires sign-in):
- Small version: [Download here](https://drive.google.com/file/d/1PzmTsmBIAXAcUXQHjWwY6o6T0IjKMtct/view?usp=drive_link)
- Large version: [Download here](https://drive.google.com/file/d/11qRpGk8bbQfChQ6pOFdOnUqtkTZAd_yJ/view?usp=drive_link)

Alternatively, you can use gdown to download the datasets (requires Google Drive access):

```bash
# Install gdown if you haven't already
pip install gdown

# Download small version dataset "export_pdf.zip"
gdown "https://drive.google.com/uc?id=1PzmTsmBIAXAcUXQHjWwY6o6T0IjKMtct"

# Download large version dataset
gdown "https://drive.google.com/uc?id=11qRpGk8bbQfChQ6pOFdOnUqtkTZAd_yJ"

# Unzip the datasets
mkdir -p inputs
unzip export_pdf.zip -d inputs/
mv inputs/export_pdf inputs/orbit_v1
rm -r inputs/__MACOSX # clean up
mkdir -p inputs/raw_orbit_v2
unzip pdf4.zip -d inputs/raw_orbit_v2

cd inputs
bash preprocess_raw_orbit_v2.sh
cd ..
rm -r inputs/raw_orbit_v2

rm pdf4.zip
rm export_pdf.zip
```

> [!NOTE]
> Both download methods require access to the Google Drive files. If you don't have access, please contact the repository maintainers.

After downloading the dataset, you can unzip the files and put them in the `inputs/orbit_v1` directory. It should contains the following files:

```
inputs/orbit_v1/
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

### MinerU Table Extraction

```bash
mkdir -p outputs/
magic-pdf -p inputs/orbit_v1/pdf -o outputs/orbit_v1_mineru133_outputs -m ocr
```

### Table Extraction

After downloading the dataset, you can run the following command to extract the tables from the PDF documents.

```bash
mkdir -p comparison_results
bash extract.sh
```

In particular,  You can check the `extract.sh` file to see the detailed command. After running the script, you will get the following files:

```
comparison_results
|-- orbit100_azure_outputs_tables
|---- f_0AibR1dz.pages.tables.json
|---- ...
|-- orbit_v1_mineru133_outputs_tables
|---- f_0AibR1dz.tables.json
|---- ...
```

### Comparisons

```bash
bash comparison.sh
```

Check the "comparison.sh" for output destination and more result details.



















# Reference

[1] MinerU:https://github.com/opendatalab/MinerU

[2] SlANET https://arxiv.org/pdf/2210.05391 

[3] PaddleX: https://github.com/PaddlePaddle/PaddleX