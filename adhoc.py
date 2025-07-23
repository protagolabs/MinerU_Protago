import os
import json
import argparse
from pathlib import Path
from collections import OrderedDict
from bs4 import BeautifulSoup

def load_json(file_path):
    """Load and parse a JSON file.
    
    Args:
        file_path (str): Path to the JSON file
        
    Returns:
        dict: Parsed JSON data
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def save_json(data, file_path):
    """Save data to a JSON file with proper formatting.
    
    Args:
        data (dict): Data to save
        file_path (str): Path where to save the JSON file
    """
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=2, ensure_ascii=False)

def extract_table_rows(html_content):
    """Extract rows from HTML table content.
    
    Args:
        html_content (str): HTML string containing table data
        
    Returns:
        list: List of rows, where each row is a list of cell contents
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    table = soup.find('table')
    if not table:
        return []
    
    rows = []
    # Process header row if exists
    header = table.find('thead')
    if header:
        header_cells = header.find_all(['th', 'td'])
        if header_cells:
            rows.append([cell.get_text(strip=True) for cell in header_cells])
    
    # Process body rows
    body = table.find('tbody') or table
    for tr in body.find_all('tr'):
        cells = tr.find_all(['td', 'th'])
        if cells:  # Only add non-empty rows
            row_data = [cell.get_text(strip=True) for cell in cells]
            if any(row_data):  # Check if row contains any non-empty cells
                rows.append(row_data)
    
    return rows

def add_rows_to_forms(node):
    """Recursively process nodes and add rows section after html in Form blocks.
    
    Args:
        node (dict): Node to process
    """
    if isinstance(node, dict):
        if node.get('block_type') == 'Form':
            # Create a new OrderedDict to maintain field order
            new_node = OrderedDict()
            for key, value in node.items():
                new_node[key] = value
                # After html field, add rows
                if key == 'html':
                    # Extract rows from HTML content
                    html_content = value
                    rows = extract_table_rows(html_content)
                    new_node['rows'] = rows
            # Update the node with ordered fields
            node.clear()
            node.update(new_node)
        
        # Recursively process all child fields that are dicts or lists
        for key, value in node.items():
            if isinstance(value, (dict, list)):
                add_rows_to_forms(value)
    elif isinstance(node, list):
        for item in node:
            add_rows_to_forms(item)

def process_json(input_path, output_path=None):
    """Process JSON file by adding rows section to Form blocks.
    
    Args:
        input_path (str): Path to input JSON file
        output_path (str, optional): Path for output JSON file. If None, will modify input file
    """
    # Load the JSON data
    data = load_json(input_path)
    
    # Process all nodes recursively
    add_rows_to_forms(data)
    
    # Determine output path
    final_output_path = output_path if output_path else input_path
    
    # Save the modified data
    save_json(data, final_output_path)
    print(f"Processed JSON saved to: {final_output_path}")

def parse_args():
    """Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description='Process JSON file to add row-wise data from HTML tables in Form blocks.'
    )
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='Path to input JSON file'
    )
    parser.add_argument(
        '-o', '--output',
        help='Path to output JSON file. If not provided, will modify input file'
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    process_json(args.input, args.output)