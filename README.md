### Quick Start

#### 1. Install magic-pdf

```bash
conda create -n dev_tables 'python>3.10' -y # we tested on 3.13
conda activate dev_tables
pip install -e ".[full]"
magic-pdf --version # check the version, should be 1.3.10
pip install -U "marker-pdf[full]==1.6.2" # install marker-pdf for table rec
pip install -U "transformers==4.49.0"
```

#### 2. Download model weight files

```bash
pip install huggingface_hub
wget https://raw.githubusercontent.com/protagolabs/MinerU_Protago/refs/heads/dev_tables/download_models_hf.py -O download_models_hf.py
python download_models_hf.py
```





## Usage

### Command Line

```bash
magic-pdf -p demo/samples_tables.pdf -o output/mineru1310
```

#### dev_log

* [x] 06/30/2025 add the multithread features, special thanks to [@finger92](https://github.com/finger92)
* [x] 06/23/2025 add the layout_only features
* [x] 06/16/2025 add the "marker" table features
* [ ] speed up the "marker" table features

If you only need layout detection without OCR, formula recognition, or table recognition, you can use the `--layout_only` option for faster processing:

```bash
magic-pdf -p your_document.pdf -o output_dir --layout_only true # layout_detection only
```


This mode will:
- Perform only layout detection on the document
- Skip OCR (text recognition)
- Skip formula recognition  
- Skip table recognition
- Significantly reduce processing time for layout analysis tasks

> [!TIP]
> For more information about the output files, please refer to the [Output File Description](docs/output_file_en_us.md).


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
cd inputs/
bash ./download_datasets.sh

```

> [!NOTE]
> Both download methods require access to the Google Drive files. If you don't have access, please contact the repository maintainers.

