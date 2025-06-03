# Guide: Changing Table Processing Models in MinerU

## 🎯 Overview
This guide shows you how to modify, replace, or add new table processing models in MinerU.

## 📋 Current Table Models Available

### 1. RapidTable (Default)
- **Model Name**: `rapid_table`
- **Implementation**: `magic_pdf/model/sub_modules/table/rapidtable/rapid_table.py`
- **Sub-models**: `slanet_plus`, `unitable`

### 2. TableMaster 
- **Model Name**: `tablemaster`
- **Implementation**: `magic_pdf/model/sub_modules/table/tablemaster/`

### 3. StructEqTable
- **Model Name**: `struct_eqtable`
- **Implementation**: `magic_pdf/model/sub_modules/table/structeqtable/`

## 🔧 Method 1: Configure Existing Models

### Change Table Model via Configuration

**Option A: Edit config file (`magic-pdf.json`)**
```json
{
    "table-config": {
        "model": "rapid_table",        # Change this to: tablemaster, struct_eqtable
        "sub_model": "slanet_plus",   # For rapid_table: slanet_plus, unitable
        "enable": true,
        "max_time": 400
    }
}
```

**Option B: Via CLI parameters**
```bash
# The table model is configured through the config file, not CLI directly
magic-pdf -p input.pdf -o output_dir -m auto
```

### Available Configuration Options

1. **RapidTable variants:**
   ```json
   {
       "table-config": {
           "model": "rapid_table",
           "sub_model": "slanet_plus",  # or "unitable"
           "enable": true,
           "max_time": 400
       }
   }
   ```

2. **TableMaster:**
   ```json
   {
       "table-config": {
           "model": "tablemaster", 
           "enable": true,
           "max_time": 400
       }
   }
   ```

3. **StructEqTable:**
   ```json
   {
       "table-config": {
           "model": "struct_eqtable",
           "enable": true, 
           "max_time": 400
       }
   }
   ```

## 🛠️ Method 2: Add Your Custom Table Model

### Step 1: Create Your Model Class

Create a new file: `magic_pdf/model/sub_modules/table/your_model/your_table_model.py`

```python
class YourCustomTableModel:
    def __init__(self, ocr_engine, model_path=None, **kwargs):
        """
        Initialize your custom table model
        Args:
            ocr_engine: OCR engine for text recognition
            model_path: Path to your model weights
            **kwargs: Additional configuration
        """
        self.ocr_engine = ocr_engine
        self.model_path = model_path
        # Initialize your model here
        
    def predict(self, image):
        """
        Process table image and return HTML
        Args:
            image: PIL Image or numpy array
            
        Returns:
            tuple: (html_code, table_cell_bboxes, logic_points, elapse_time)
        """
        # Your table processing logic here
        html_code = self.process_table(image)
        table_cell_bboxes = []  # Cell bounding boxes if available
        logic_points = []       # Logic points if available  
        elapse_time = 0        # Processing time
        
        return html_code, table_cell_bboxes, logic_points, elapse_time
        
    def process_table(self, image):
        """Your custom table processing implementation"""
        # Example: 
        # 1. Run OCR on the image
        # 2. Detect table structure
        # 3. Generate HTML
        pass
```

### Step 2: Register Your Model

**A. Add to constants** (`magic_pdf/config/constants.py`):
```python
class MODEL_NAME:
    # ... existing models ...
    YOUR_CUSTOM_TABLE = 'your_custom_table'
```

**B. Add to model initialization** (`magic_pdf/model/sub_modules/model_init.py`):
```python
def table_model_init(table_model_type, model_path, max_time, _device_='cpu', lang=None, table_sub_model_name=None):
    # ... existing conditions ...
    elif table_model_type == MODEL_NAME.YOUR_CUSTOM_TABLE:
        from magic_pdf.model.sub_modules.table.your_model.your_table_model import YourCustomTableModel
        atom_model_manager = AtomModelSingleton()
        ocr_engine = atom_model_manager.get_atom_model(
            atom_model_name='ocr',
            ocr_show_log=False,
            det_db_box_thresh=0.5,
            det_db_unclip_ratio=1.6,
            lang=lang
        )
        table_model = YourCustomTableModel(ocr_engine, model_path)
    else:
        logger.error('table model type not allow')
        exit(1)
```

### Step 3: Update Configuration

Add to your `magic-pdf.json`:
```json
{
    "table-config": {
        "model": "your_custom_table",
        "enable": true,
        "max_time": 400,
        "model_path": "/path/to/your/model/weights"
    }
}
```

## 🔍 Method 3: Modify Existing RapidTable

### Customize RapidTable Parameters

Edit `magic_pdf/model/sub_modules/table/rapidtable/rapid_table.py`:

```python
class RapidTableModel(object):
    def __init__(self, ocr_engine, table_sub_model_name='slanet_plus'):
        # Modify these parameters:
        # - model_type: Change sub-model 
        # - model_path: Use custom model path
        # - confidence thresholds
        # - preprocessing parameters
        
        if table_sub_model_name == "your_custom_slanet":
            # Load your custom trained SLANet model
            custom_model_path = "/path/to/your/custom/model.onnx"
            input_args = RapidTableInput(
                model_type="slanet_plus", 
                model_path=custom_model_path
            )
```

### Modify Table Processing Logic

You can also modify the prediction logic in the `predict()` method:

```python
def predict(self, image):
    # Add custom preprocessing
    image = self.custom_preprocess(image)
    
    # Modify rotation detection logic
    # Change OCR parameters
    # Adjust post-processing
    
    # Your custom logic here...
```

## 📊 Model Performance Comparison

| Model | Speed | Accuracy | Memory Usage | Best Use Case |
|-------|-------|----------|--------------|---------------|
| rapid_table (slanet_plus) | Fast | High | Low | General tables |
| rapid_table (unitable) | Medium | Very High | Medium | Complex tables |
| tablemaster | Medium | High | Medium | Scientific papers |
| struct_eqtable | Slow | Very High | High | Math/equation tables |

## 🧪 Testing Your Changes

### Create Test Script

```python
# test_custom_table.py
from magic_pdf.model.sub_modules.model_init import AtomModelSingleton
from PIL import Image

# Initialize model manager
atom_model_manager = AtomModelSingleton()

# Get your custom table model
table_model = atom_model_manager.get_atom_model(
    atom_model_name='table',
    table_model_name='your_custom_table',  # Your model name
    table_model_path='/path/to/model',
    table_max_time=400,
    device='cpu',
    lang='ch'
)

# Test with sample image
img = Image.open("test_table.jpg")
html_code, cell_bboxes, logic_points, elapse = table_model.predict(img)
print(f"Generated HTML: {html_code}")
```

### Run Tests
```bash
# Test with existing test suite
python -m pytest tests/unittest/test_table/ -v

# Test with your custom model
python test_custom_table.py
```

## 🚀 Advanced Customizations

### 1. Multi-Model Ensemble
Combine multiple table models for better accuracy:

```python
class EnsembleTableModel:
    def __init__(self, ocr_engine):
        self.rapid_model = RapidTableModel(ocr_engine, 'slanet_plus')
        self.unitable_model = RapidTableModel(ocr_engine, 'unitable')
        
    def predict(self, image):
        # Get predictions from both models
        html1, _, _, _ = self.rapid_model.predict(image)
        html2, _, _, _ = self.unitable_model.predict(image)
        
        # Use confidence scoring or voting to choose best result
        return self.select_best_result(html1, html2)
```

### 2. Add Model-Specific Preprocessing

```python
def custom_preprocess(self, image, model_type):
    if model_type == "high_resolution_tables":
        # Apply super-resolution
        image = self.upscale_image(image)
    elif model_type == "handwritten_tables": 
        # Apply denoising
        image = self.denoise_image(image)
    return image
```

### 3. Custom Post-Processing

```python
def enhance_html_output(self, html_code):
    # Add CSS styling
    # Fix common HTML errors  
    # Add table metadata
    # Convert to other formats (CSV, JSON)
    return enhanced_html
```

## 🔧 Configuration File Reference

Complete `magic-pdf.json` with table options:

```json
{
    "models-dir": "/path/to/models",
    "device-mode": "gpu",
    "table-config": {
        "model": "rapid_table",
        "sub_model": "slanet_plus", 
        "enable": true,
        "max_time": 400,
        "custom_model_path": "",
        "confidence_threshold": 0.5,
        "preprocessing": {
            "enable_rotation_correction": true,
            "enhance_resolution": false
        }
    }
}
```

## 🐛 Common Issues & Solutions

### Issue 1: Model Not Loading
- Check model path in configuration
- Verify model file exists and is accessible
- Check device compatibility (CPU/GPU)

### Issue 2: Poor Table Recognition
- Try different sub-models (slanet_plus vs unitable)
- Adjust confidence thresholds
- Enable preprocessing options
- Use appropriate language settings

### Issue 3: Performance Issues
- Reduce max_time parameter
- Use CPU instead of GPU for small tables
- Enable batch processing for multiple tables

## 📚 Further Reading

- [RapidTable Documentation](https://github.com/RapidAI/RapidTable)
- [TableMaster Paper](https://arxiv.org/abs/2105.01848)
- [MinerU Model Configuration](https://github.com/opendatalab/MinerU/blob/master/docs/)

---

This guide provides comprehensive instructions for changing table processing models in MinerU. Choose the method that best fits your needs!