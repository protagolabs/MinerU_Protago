### Quick Start

#### 1. Install magic-pdf

```bash
conda create -n dev 'python>3.10' -y # we tested on 3.13
conda activate dev
pip install -e ".[full]"
magic-pdf --version # check the version, should be 1.3.10
# pip install -U "marker-pdf[full]==1.6.2" # install marker-pdf for table rec
# pip install -U "transformers==4.49.0"
```

#### 2. Download model weight files

```bash
pip install huggingface_hub
wget https://raw.githubusercontent.com/protagolabs/MinerU_Protago/refs/heads/dev/download_models_hf.py -O download_models_hf.py
python download_models_hf.py
```



## Usage

### Command Line

[Using MinerU via Command Line](https://mineru.readthedocs.io/en/latest/user_guide/usage/command_line.html)

#### dev_log

* [x] 06/30/2025 add the multithread features, special thanks to [@finger92](https://github.com/finger92)
* [x] 06/23/2025 add the layout_only features
* [x] 06/16/2025 add the "marker" table features
* [ ] speed up the "marker" table features

If you only need layout detection without OCR, formula recognition, or table recognition, you can use the `--layout_only` option for faster processing:

```bash
magic-pdf -p your_document.pdf -o output_dir --layout_only true # if the pdf file page number is less than 10, it will set layout_only=true by default
```


This mode will:
- Perform only layout detection on the document
- Skip OCR (text recognition)
- Skip formula recognition  
- Skip table recognition
- Significantly reduce processing time for layout analysis tasks

> [!TIP]
> For more information about the output files, please refer to the [Output File Description](docs/output_file_en_us.md).



