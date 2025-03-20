# MinerU Protago

## Overview

This repository contains the code for the MinerU Protago project, which is adopted from the [MinerU](https://github.com/opendatalab/MinerU) project. In this project, we aim to extract the tables from the PDF documents from Orbit dataset and convert them into the HTML format, which can be used for the further processing.

## Data Preparation

### Orbit Dataset



The Orbit dataset is a collection of PDF documents with tables. There are two versions, one is a small version with 176 PDF documents, and the other is the larger version with 1000 PDF documents. All the code is tested on the small version.

You can download the small version dataset from [here](https://orbit-common-resources.s3.us-west-2.amazonaws.com/all_file_type_case/export_pdf.zip) and large one from [here](https://orbit-common-resources.s3.us-west-2.amazonaws.com/ljx_20241210/pdf.zip).


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







# Reference

[1] MinerU:https://github.com/opendatalab/MinerU

[2] SlANET https://arxiv.org/pdf/2210.05391 

[3] PaddleX: https://github.com/PaddlePaddle/PaddleX