### Quick Start

#### 1. Install magic-pdf

```bash
conda create -n minerumarker 'python=3.10' -y
conda activate minerumarker
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

#### 3. Modify the Configuration File for Additional Configuration

After completing previous step, the script will automatically generate a `magic-pdf.json` file in the user directory and configure the default model path.
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
    // "table-config": {
    //     "model": "rapid_table", 
    //     "sub_model": "slanet_plus",
    //     "enable": true, // The table recognition feature is enabled by default. If you need to disable it, please change the value here to "false".
    //     "max_time": 400
    // }
    "table-config": {
        "model": "marker_table",
        "enable": true,
        "max_time": 400
    },  
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


