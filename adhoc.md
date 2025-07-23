# HTML Table to Row-wise Data Converter

This script processes JSON files containing HTML tables within Form blocks and adds row-wise structured data while preserving the original HTML content.

> **Attention**: This code is tested on the JSON result from [marker](https://github.com/datalab-to/marker) with version 1.6.2

## Features

- Extracts tabular data from HTML tables in Form blocks
- Preserves original HTML content
- Maintains field order in the JSON structure
- Handles both simple and complex table structures
- Supports in-place modification or output to new file

## Installation

Required dependencies:
```bash
pip install marker-pdf[full]==1.6.2
pip install -U "transformers==4.49.0"
pip install beautifulsoup4
```

## Usage

The script can be used in two ways:

1. Create a new output file:
```bash
marker_single /path/to/input.pdf --output_dir "output"
python adhoc.py -i output/input.json -o output/output.json
```


### Command Line Arguments

- `-i, --input`: Path to input JSON file (required)
- `-o, --output`: Path to output JSON file (optional)
- `-h, --help`: Show help message and exit

## How It Works

### 1. JSON Structure

The script expects JSON files with Form blocks containing HTML tables. Example structure:
```json
{
  "block_type": "Form",
  "html": "<table><tr><th>Name</th><td>Value</td></tr></table>",
  "other_fields": "..."
}
```

### 2. Conversion Process

The script:
1. Recursively searches for Form blocks in the JSON structure
2. For each Form block:
   - Preserves the original HTML content
   - Extracts table data into rows
   - Adds a new "rows" field after the "html" field
3. Saves the modified structure

### 3. Output Format

The processed JSON will include a new "rows" field:
```json
{
  "block_type": "Form",
  "html": "<table><tr><th>Name</th><td>Value</td></tr></table>",
  "rows": [
    ["Name", "Value"]
  ],
  "other_fields": "..."
}
```

## Examples

### Simple Table Conversion

Input HTML:
```html
<table>
    <tr>
        <th>Name</th>
        <th>Age</th>
    </tr>
    <tr>
        <td>John</td>
        <td>25</td>
    </tr>
</table>
```

Resulting rows:
```json
"rows": [
    ["Name", "Age"],
    ["John", "25"]
]
```

### Complex Table Conversion

Input HTML:
```html
<table>
    <thead>
        <tr>
            <th colspan="2">Employee Info</th>
            <th>Contact</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>John Smith</td>
            <td>Manager</td>
            <td>123-456-7890</td>
        </tr>
    </tbody>
</table>
```

Resulting rows:
```json
"rows": [
    ["Employee Info", "Contact"],
    ["John Smith", "Manager", "123-456-7890"]
]
```

## Features

- Handles thead and tbody sections
- Preserves table structure
- Strips whitespace from cell contents
- Skips empty rows
- Processes nested tables
- Maintains field order in JSON

## Error Handling

- Gracefully handles missing tables
- Skips malformed HTML
- Preserves original data if parsing fails

## Notes

1. The script uses BeautifulSoup4 for HTML parsing
2. Original HTML content is preserved in the "html" field
3. Row extraction handles both header cells (th) and data cells (td)
4. Empty rows are filtered out
5. Whitespace is stripped from cell contents 