# MinerU Protago

## Overview

This repository contains the code for the MinerU Protago project, which is adopted from the [MinerU](https://github.com/opendatalab/MinerU) project. In this project, we aim to extract the tables from the PDF documents from Orbit dataset and convert them into the HTML format, which can be used for the further processing. I personally use different python environments for MinerU and PaddleX.


## Devlog
we update the TED and TED structure calculation based on the official code in [PubTabNet](https://github.com/ibm-aur-nlp/PubTabNet). Further evaluations will be conducted on this.

| Sample Size  | Tools (version)  |    TED     | TED structure |
|------------- |------------------|------------|---------------|
| ~176 (v1)    |  MinerU (1.3.3)  |    64.51%  |    81.06%     |
| ~176 (v1)    |  Marker (1.3.3)  |    55.49%  |    65.786%    |

## Devlog (Deprecated)

| Sample Size | Version | TED     | TED structure |
|-------------|---------|---------|----------------|
| ~176 (v1)        | 1.2.2   | 48.36%  | 92.66%         |
| ~176 (v1)        | 1.3.3   | **53.77%** | **92.99%**     |
| ~176 (v1)        | marker   | 72.85% | 66.43%     |
| ~176 (v1)        | 1.3.10   | 53.48% | 92.84%     |
| ~176 (v1)        | 1.3.3 + vlm (md2html)   | **61.24%** | **94.71%**     |
| ~1063 (v2)       | 1.2.2   | 45.77%  | 92.17%         |
| ~1063 (v2)     | 1.3.3   | **52.97%** | **92.39%**     |
| ~1063 (v2)     | 1.3.10   | 52.16% | 92.20%     |



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
- v1 version: [Download here](https://drive.google.com/file/d/1PzmTsmBIAXAcUXQHjWwY6o6T0IjKMtct/view?usp=drive_link)
- v2 version: [Download here](https://drive.google.com/file/d/11qRpGk8bbQfChQ6pOFdOnUqtkTZAd_yJ/view?usp=drive_link)
- v3 version: [Download here](https://drive.google.com/file/d/1Uyb-ImPfH6UirS33mSHGkAyC836pwrgf/view?usp=drive_link)

Alternatively, you can use gdown to download the datasets (requires Google Drive access):

```bash
# Install gdown if you haven't already
pip install gdown


gdown "https://drive.google.com/uc?id=1PzmTsmBIAXAcUXQHjWwY6o6T0IjKMtct"

# Unzip the datasets
unzip export_pdf.zip -d inputs/
mv inputs/export_pdf inputs/orbit_v1
rm -r inputs/__MACOSX # clean up
rm export_pdf.zip

gdown "https://drive.google.com/uc?id=11qRpGk8bbQfChQ6pOFdOnUqtkTZAd_yJ"
mkdir -p inputs/raw_orbit_v2
unzip pdf4.zip -d inputs/raw_orbit_v2
cd inputs
bash preprocess_raw_orbit_v2.sh
cd ..
rm -r inputs/raw_orbit_v2
rm pdf4.zip

gdown "https://drive.google.com/uc?id=1Uyb-ImPfH6UirS33mSHGkAyC836pwrgf"
unzip raw_pdf5000.zip -d inputs/
mv inputs/raw_pdf5000 inputs/raw_orbit_v3
cd inputs
bash preprocess_raw_orbit_v3.sh
cd ..
rm -r inputs/raw_orbit_v3
rm raw_pdf5000.zip

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
mkdir -p logs/
magic-pdf -p inputs/orbit_v1/pdf -o outputs/orbit_v1_mineru133_outputs -m ocr > logs/orbit_v1_mineru133_outputs.log 2>&1
```

If you meet the error "MemoryError", you can try to change the following in "magic_pdf/tools/cli.py". This will disable the batch processing and process the file one by one.
```python
    if os.path.isdir(path):
        for doc_path in Path(path).glob('*'):
            if doc_path.suffix in pdf_suffixes + image_suffixes + ms_office_suffixes:
                parse_doc(doc_path)
    else:
        parse_doc(Path(path))
        
    # if os.path.isdir(path):
    #     doc_paths = []
    #     for doc_path in Path(path).glob('*'):
    #         if doc_path.suffix in pdf_suffixes + image_suffixes + ms_office_suffixes:
                # if doc_path.suffix in ms_office_suffixes:
                #     convert_file_to_pdf(str(doc_path), temp_dir)
                #     doc_path = Path(os.path.join(temp_dir, f'{doc_path.stem}.pdf'))
                # elif doc_path.suffix in image_suffixes:
                #     with open(str(doc_path), 'rb') as f:
                #         bits = f.read()
                #         pdf_bytes = fitz.open(stream=bits).convert_to_pdf()
                #     fn = os.path.join(temp_dir, f'{doc_path.stem}.pdf')
                #     with open(fn, 'wb') as f:
                #         f.write(pdf_bytes)
                #     doc_path = Path(fn)
                # doc_paths.append(doc_path)
        # datasets = batch_build_dataset(doc_paths, 4, lang)
        # batch_do_parse(output_dir, [str(doc_path.stem) for doc_path in doc_paths], datasets, method, debug_able, lang=lang)
    # else:
    #     parse_doc(Path(path))
```

### Table Extraction

After downloading the dataset, you can run the following command to extract the tables from the PDF documents.

```bash
mkdir -p comparison
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

Check the "comparison.sh" for output destination and more result details. The json file should contains something similar to the following:

```json
    "summary": {
        "total_files_processed": 176,
        "overall_average_similarity": 0.5377033703165067,
        "overall_average_structure_similarity": 0.9299762187917149,
        "files_with_tables": 176,
        "overall_average_similarity_tables_only": 0.5377033703165067,
        "overall_average_structure_similarity_tables_only": 0.9299762187917149,
        "timestamp": "20250429_153454"
    },
```


















# Reference

[1] MinerU:https://github.com/opendatalab/MinerU

[2] SlANET https://arxiv.org/pdf/2210.05391 

[3] PaddleX: https://github.com/PaddlePaddle/PaddleX
