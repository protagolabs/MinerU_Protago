# SuryaTableWrapper - Modified Implementation

## Overview

The `SuryaTableWrapper` has been completely rewritten to properly handle table recognition using the Surya library and generate HTML output from table images. This implementation follows the pattern used in the `TableProcessor` class from `table.py`.

## Key Changes

### 1. **Complete Rewrite**
- Removed the old implementation that was trying to extract HTML from non-existent JSON output
- Implemented a proper table recognition pipeline using Surya's models

### 2. **Model Integration**
- **Detection Model**: `DetectionPredictor` for text detection
- **Recognition Model**: `RecognitionPredictor` for OCR text recognition
- **Table Recognition Model**: `TableRecPredictor` for table structure recognition

### 3. **Processing Pipeline**
The new implementation follows this workflow:

1. **OCR Processing**: Extract text lines from the table image
2. **Table Recognition**: Identify table structure and cell boundaries
3. **Text Assignment**: Map OCR text to table cells based on intersection
4. **Post-processing**: Apply row splitting and dollar column combination
5. **HTML Generation**: Convert the processed table to HTML format

### 4. **HTML Generation**
- Generates proper HTML table structure with `<table>`, `<tr>`, `<td>`, and `<th>` tags
- Handles rowspan and colspan attributes for merged cells
- Properly escapes HTML special characters
- Identifies header cells and uses `<th>` tags accordingly

## Usage

### Basic Usage

```python
from magic_pdf.model.sub_modules.table.surya_table.surya_table_wrapper import SuryaTableWrapper
from PIL import Image

# Initialize wrapper
wrapper = SuryaTableWrapper()

# Load table image
image = Image.open("table_image.png")

# Process table with default language
html_code, table_cell_bboxes, logic_points, elapse = wrapper.predict(image)

# Or process table with specific language (overrides default)
html_code, table_cell_bboxes, logic_points, elapse = wrapper.predict(image, language='zh')

# Use the HTML output
if html_code:
    print(html_code)
```

### Configuration Options

```python
config = {
    'disable_tqdm': True,           # Disable progress bars
    'drop_repeated_text': False,    # Keep repeated text in OCR
    'format_lines': False,          # Don't format text lines
    'language': 'en',               # Language for OCR (en, zh, ja, ko, etc.)
    'detection_batch_size': 4,      # Batch size for detection model
    'recognition_batch_size': 32,   # Batch size for recognition model
    'table_rec_batch_size': 6       # Batch size for table recognition model
}

wrapper = SuryaTableWrapper(config=config)
```

## Return Values

The `predict()` method returns a tuple with the following elements:

1. **html_code** (str): Generated HTML table markup
2. **table_cell_bboxes** (list): Currently empty (for compatibility)
3. **logic_points** (list): Currently empty (for compatibility)
4. **elapse** (float): Processing time in seconds

## Features

### 1. **Text Processing**
- Normalizes various Unicode space characters
- Cleans up text artifacts (dots, repeated spaces)
- Uses `ftfy` for text fixing

### 2. **Row Splitting**
- Automatically splits combined rows when appropriate
- Handles multi-line text within cells
- Maintains proper row structure

### 3. **Dollar Column Combination**
- Combines columns that contain only dollar signs
- Merges dollar signs with adjacent numeric columns
- Improves table readability

### 4. **Cell Intersection**
- Calculates intersection areas between text and cell bounding boxes
- Assigns text to the most overlapping cell
- Handles edge cases and empty cells

### 5. **Language Support**
- Configurable language for OCR processing
- Supports multiple languages (en, zh, ja, ko, etc.)
- Defaults to English if not specified
- **Dynamic language override**: Can specify language per prediction call

## Dependencies

The implementation requires the following dependencies:

```python
from surya.detection import DetectionPredictor
from surya.recognition import RecognitionPredictor, OCRResult
from surya.table_rec import TableRecPredictor
from surya.table_rec.schema import TableResult, TableCell as SuryaTableCell
from ftfy import fix_text
from PIL import Image
import numpy as np
```

## Error Handling

The wrapper includes comprehensive error handling:

- Model initialization errors
- OCR processing failures with fallback methods
- Table recognition failures
- HTML generation errors

All errors are logged and the method returns `(None, None, None, None)` on failure.

## Recent Fixes

### RecognitionPredictor Method Signature
- Fixed the `RecognitionPredictor.__call__()` method signature
- Added required `langs` parameter
- Removed unsupported `drop_repeated_text` and `task_names` parameters
- Added fallback methods for different API versions

## Testing

Use the provided test script to verify the implementation:

```bash
python test_surya_table.py
```

## Compatibility

This implementation maintains compatibility with the existing MinerU table model interface, returning the same tuple format as other table recognition models in the system.

## Performance

The implementation includes configurable batch sizes for optimal performance on different hardware configurations. Default batch sizes are:

- Detection: 4
- Recognition: 32  
- Table Recognition: 6

These can be adjusted based on available GPU memory and processing requirements.

## Language Configuration

The wrapper supports multiple languages for OCR processing:

```python
# English (default)
config = {'language': 'en'}

# Chinese
config = {'language': 'zh'}

# Japanese
config = {'language': 'ja'}

# Korean
config = {'language': 'ko'}

# And many more supported by Surya
```

The language parameter is used for the OCR recognition step to improve accuracy for different scripts and languages.

## Integration with Batch Processing

The wrapper is designed to work seamlessly with batch processing systems like the one in `batch_analyze.py`. The language information from the batch processing can be passed directly to the predict method:

```python
# In batch processing context
for table_res_dict in table_res_list_all_page:
    _lang = table_res_dict['lang']  # Language from batch processing
    table_img = table_res_dict['table_img']
    
    # Pass the language dynamically
    html_code, table_cell_bboxes, logic_points, elapse = wrapper.predict(
        table_img, 
        language=_lang
    )
```

This allows the wrapper to use the most appropriate language for each table image based on the context from the batch processing pipeline. 