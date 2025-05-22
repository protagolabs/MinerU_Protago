#!/usr/bin/env python3
"""
Table Extractor for Orbit Marker Outputs

This script processes Orbit marker output JSON files to extract tables.
It processes all JSON files in the specified input directory structure.
"""

import os
import json
import argparse
from pathlib import Path

def load_json(file_path):
    """Load and parse a JSON file.
    
    Args:
        file_path (str): Path to the JSON file
        
    Returns:
        dict: Parsed JSON data
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def extract_tables(data):
    """Extract tables from Orbit marker output data.
    
    Args:
        data (dict): Orbit marker output data
        
    Returns:
        list: List of dictionaries containing extracted table information
    """
    formatted_output = []
    page_numbers = {}  # Store page numbers for each node ID
    
    # First pass: collect all page numbers
    def collect_page_numbers(node):
        if node.get('block_type') == 'Page':
            try:
                # Extract page number from ID like "/page/1/Page/206" and add 1 to make it 1-based
                page_num = int(node['id'].split('/')[2]) + 1
                # Store page number for all children
                if node.get('children'):
                    for child in node['children']:
                        page_numbers[child['id']] = page_num
                        if child.get('children'):
                            for grandchild in child['children']:
                                page_numbers[grandchild['id']] = page_num
            except (ValueError, IndexError, KeyError):
                pass
        
        # Recursively process children
        if node.get('children'):
            for child in node['children']:
                collect_page_numbers(child)
    
    # Second pass: extract tables
    def process_node(node):
        if node.get('block_type') == 'Table':
            # Get page number from our mapping
            page_num = page_numbers.get(node['id'], 1)
            
            # Extract table information
            table_info = {
                "page": page_num,
                "img_path": "",  # Empty image path as we don't have image information
                "types": "table",
                "sentence": f"<html><body>{node.get('html', '')}</body></html>"
            }
            formatted_output.append(table_info)
        
        # Recursively process children
        if node.get('children'):
            for child in node['children']:
                process_node(child)
    
    # First collect all page numbers
    for child in data.get('children', []):
        collect_page_numbers(child)
    
    # Then process tables
    for child in data.get('children', []):
        process_node(child)
    
    return formatted_output

def process_directory(input_dir, output_dir):
    """Process all JSON files in the input directory structure.
    
    Args:
        input_dir (str): Input directory containing Orbit marker output files
        output_dir (str): Output directory for processed results
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            # Skip meta files
            if file.endswith('_meta.json'):
                continue
                
            if file.endswith('.json'):
                input_path = os.path.join(root, file)
                
                # Create output filename with .tables.json suffix
                output_filename = f"{os.path.splitext(file)[0]}.tables.json"
                output_path = os.path.join(output_dir, output_filename)
                
                try:
                    # Load and process the file
                    data = load_json(input_path)
                    extracted_data = extract_tables(data)
                    
                    # Save to JSON file
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(extracted_data, f, ensure_ascii=False, indent=4)
                    
                    print(f"Processed {input_path} -> {output_path}")
                    
                except Exception as e:
                    print(f"Error processing {input_path}: {str(e)}")
                    continue

def main():
    """Main function to parse arguments and process files."""
    parser = argparse.ArgumentParser(
        description='Extract tables from Orbit marker output files.'
    )
    parser.add_argument('--input', '-i', required=True, help='Input directory containing Orbit marker output files')
    parser.add_argument('--output', '-o', required=True, help='Output directory for processed results')
    
    args = parser.parse_args()
    
    process_directory(args.input, args.output)

if __name__ == '__main__':
    main() 