### Quick Start

#### 1. Install magic-pdf

```bash
conda create -n minerumarker 'python=3.10' -y
conda activate mineru1310
pip install -e ".[full]"
magic-pdf --version # check the version, should be 1.3.10
pip install -U "marker-pdf[full]==1.6.2" # install marker-pdf for table rec
```

#### 2. Download model weight files

```bash
pip install huggingface_hub
wget https://github.com/protagolabs/MinerU_Protago/blob/dev_tables/download_models_hf.py -O download_models_hf.py
python download_models_hf.py
```

#### 3. Modify the Configuration File for Additional Configuration

After completing the [2. Download model weight files](#2-download-model-weight-files) step, the script will automatically generate a `magic-pdf.json` file in the user directory and configure the default model path.
You can find the `magic-pdf.json` file in your 【user directory】.

> [!TIP]
> The user directory for Windows is "C:\\Users\\username", for Linux it is "/home/username", and for macOS it is "/Users/username".

You can modify certain configurations in this file to enable or disable features, such as table recognition:


> [!NOTE]
> If the following items are not present in the JSON, please manually add the required items and remove the comment content (standard JSON does not support comments).

```json
{
    // other config
    "layout-config": {
        "model": "doclayout_yolo" 
    },
    "formula-config": {
        "mfd_model": "yolo_v8_mfd",
        "mfr_model": "unimernet_small",
        "enable": true  // The formula recognition feature is enabled by default. If you need to disable it, please change the value here to "false".
    },
    "table-config": {
        "model": "rapid_table", 
        "sub_model": "slanet_plus",
        "enable": true, // The table recognition feature is enabled by default. If you need to disable it, please change the value here to "false".
        "max_time": 400
    }
}
```

### Using GPU

If your device supports CUDA and meets the GPU requirements of the mainline environment, you can use GPU acceleration. Please select the appropriate guide based on your system:

- [Ubuntu 22.04 LTS + GPU](docs/README_Ubuntu_CUDA_Acceleration_en_US.md)
- [Windows 10/11 + GPU](docs/README_Windows_CUDA_Acceleration_en_US.md)
- Quick Deployment with Docker
> [!IMPORTANT]
> Docker requires a GPU with at least 6GB of VRAM, and all acceleration features are enabled by default.
>
> Before running this Docker, you can use the following command to check if your device supports CUDA acceleration on Docker.
> 
> ```bash
> docker run --rm --gpus=all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
> ```
  ```bash
  wget https://github.com/opendatalab/MinerU/raw/master/docker/global/Dockerfile -O Dockerfile
  docker build -t mineru:latest .
  docker run -it --name mineru --gpus=all mineru:latest /bin/bash -c "echo 'source /opt/mineru_venv/bin/activate' >> ~/.bashrc && exec bash"
  magic-pdf --help
  ```

### Using NPU

If your device has NPU acceleration hardware, you can follow the tutorial below to use NPU acceleration:

[Ascend NPU Acceleration](docs/README_Ascend_NPU_Acceleration_zh_CN.md)

### Using MPS

If your device uses Apple silicon chips, you can enable MPS acceleration for your tasks.

You can enable MPS acceleration by setting the `device-mode` parameter to `mps` in the `magic-pdf.json` configuration file.

```json
{
    // other config
    "device-mode": "mps"
}
```


## Usage

### Command Line

[Using MinerU via Command Line](https://mineru.readthedocs.io/en/latest/user_guide/usage/command_line.html)

> [!TIP]
> For more information about the output files, please refer to the [Output File Description](docs/output_file_en_us.md).

### API

[Using MinerU via Python API](https://mineru.readthedocs.io/en/latest/user_guide/usage/api.html)


### Deploy Derived Projects

Derived projects include secondary development projects based on MinerU by project developers and community developers,  
such as application interfaces based on Gradio, RAG based on llama, web demos similar to the official website, lightweight multi-GPU load balancing client/server ends, etc.
These projects may offer more features and a better user experience.  
For specific deployment methods, please refer to the [Derived Project README](projects/README.md)


### Development Guide

TODO

# TODO

- [x] Reading order based on the model  
- [x] Recognition of `index` and `list` in the main text  
- [x] Table recognition
- [x] Heading Classification
- [ ] Code block recognition in the main text
- [ ] [Chemical formula recognition](docs/chemical_knowledge_introduction/introduction.pdf)
- [ ] Geometric shape recognition

# Known Issues

- Reading order is determined by the model based on the spatial distribution of readable content, and may be out of order in some areas under extremely complex layouts.
- Vertical text is not supported.
- Tables of contents and lists are recognized through rules, and some uncommon list formats may not be recognized.
- Code blocks are not yet supported in the layout model.
- Comic books, art albums, primary school textbooks, and exercises cannot be parsed well.
- Table recognition may result in row/column recognition errors in complex tables.
- OCR recognition may produce inaccurate characters in PDFs of lesser-known languages (e.g., diacritical marks in Latin script, easily confused characters in Arabic script).
- Some formulas may not render correctly in Markdown.

# FAQ

[FAQ in Chinese](docs/FAQ_zh_cn.md)

[FAQ in English](docs/FAQ_en_us.md)

# All Thanks To Our Contributors

<a href="https://github.com/opendatalab/MinerU/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=opendatalab/MinerU" />
</a>

# License Information

[LICENSE.md](LICENSE.md)

This project currently uses PyMuPDF to achieve advanced functionality. However, since it adheres to the AGPL license, it may impose restrictions on certain usage scenarios. In future iterations, we plan to explore and replace it with a more permissive PDF processing library to enhance user-friendliness and flexibility.

# Acknowledgments

- [PDF-Extract-Kit](https://github.com/opendatalab/PDF-Extract-Kit)
- [DocLayout-YOLO](https://github.com/opendatalab/DocLayout-YOLO)
- [StructEqTable](https://github.com/UniModal4Reasoning/StructEqTable-Deploy)
- [RapidTable](https://github.com/RapidAI/RapidTable)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [RapidOCR](https://github.com/RapidAI/RapidOCR)
- [PaddleOCR2Pytorch](https://github.com/frotms/PaddleOCR2Pytorch)
- [PyMuPDF](https://github.com/pymupdf/PyMuPDF)
- [layoutreader](https://github.com/ppaanngggg/layoutreader)
- [fast-langdetect](https://github.com/LlmKira/fast-langdetect)
- [pdfminer.six](https://github.com/pdfminer/pdfminer.six)

# Citation

```bibtex
@misc{wang2024mineruopensourcesolutionprecise,
      title={MinerU: An Open-Source Solution for Precise Document Content Extraction}, 
      author={Bin Wang and Chao Xu and Xiaomeng Zhao and Linke Ouyang and Fan Wu and Zhiyuan Zhao and Rui Xu and Kaiwen Liu and Yuan Qu and Fukai Shang and Bo Zhang and Liqun Wei and Zhihao Sui and Wei Li and Botian Shi and Yu Qiao and Dahua Lin and Conghui He},
      year={2024},
      eprint={2409.18839},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2409.18839}, 
}

@article{he2024opendatalab,
  title={Opendatalab: Empowering general artificial intelligence with open datasets},
  author={He, Conghui and Li, Wei and Jin, Zhenjiang and Xu, Chao and Wang, Bin and Lin, Dahua},
  journal={arXiv preprint arXiv:2407.13773},
  year={2024}
}
```

# Star History

<a>
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=opendatalab/MinerU&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=opendatalab/MinerU&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=opendatalab/MinerU&type=Date" />
 </picture>
</a>

# Magic-doc

[Magic-Doc](https://github.com/InternLM/magic-doc) Fast speed ppt/pptx/doc/docx/pdf extraction tool

# Magic-html

[Magic-HTML](https://github.com/opendatalab/magic-html) Mixed web page extraction tool

# Links

- [LabelU (A Lightweight Multi-modal Data Annotation Tool)](https://github.com/opendatalab/labelU)
- [LabelLLM (An Open-source LLM Dialogue Annotation Platform)](https://github.com/opendatalab/LabelLLM)
- [PDF-Extract-Kit (A Comprehensive Toolkit for High-Quality PDF Content Extraction)](https://github.com/opendatalab/PDF-Extract-Kit)
