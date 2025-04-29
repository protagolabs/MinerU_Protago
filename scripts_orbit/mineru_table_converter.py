#!/usr/bin/env python3
"""
Table and Image Extractor

This script processes PDF OCR output files to extract tables and images information.
It takes input and output directories as command line arguments and processes all
JSON files in the input directory structure.
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

def extract_tables_images(base_folder, filename, data):
    """Extract tables and images information from OCR data.
    
    Args:
        base_folder (str): Base directory containing the files
        filename (str): Name of the current file being processed
        data (dict): OCR data containing PDF information
        
    Returns:
        list: List of dictionaries containing extracted table and image information
    """
    formatted_output = []
    current_page = 1
    
    # Handle case where data is a list
    pdf_info = data.get("pdf_info", [])
    
    # Data is a list of pages
    for page_data in pdf_info:
        tables = []
        images = []

        # Each page_data is a dictionary with a 'preproc_blocks' key
        page = page_data.get('preproc_blocks', {})
        
        # Extract tables directly from the 'tables' key
        if 'tables' in page_data:
            tables.extend(page_data['tables'])
            
        # Extract images directly from the 'images' key
        if 'images' in page_data:
            images.extend(page_data['images'])
            
        # Also check discarded_blocks for any relevant information
        for block in page_data.get('discarded_blocks', []):
            if block.get('type') == 'table':
                tables.append(block)
            elif block.get('type') == 'image':
                images.append(block)

        # Format tables and images into content
        for table in tables:
            for block in table.get('blocks', []):
                for line in block.get('lines', []):
                    for span in line.get('spans', []):
                        if span.get('html'):
                            # Create separate entry for each table
                            formatted_output.append({
                                "page": current_page,
                                "img_path": os.path.join(base_folder, filename, 'ocr/images', span['image_path']),
                                "types": span['type'],
                                "sentence": f"<table>{span['html']}</table>"
                            })

        for image in images:
            if image.get('image_path'):
                formatted_output.append({
                    "page": current_page,
                    "img_path": image['image_path'],
                    "types": image['type'],
                    "sentence": ""  # Empty for images
                })
        
        current_page += 1
    
    return formatted_output

def process_folder(base_folder, output_folder):
    """Process all files in the specified folder structure.
    
    Args:
        base_folder (str): Input directory containing OCR files
        output_folder (str): Output directory for processed results
    """
    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # List all subdirectories in the base folder
    for filename in os.listdir(base_folder):
        input_path = os.path.join(base_folder, filename, 'ocr', f'{filename}_middle.json')
        
        if os.path.exists(input_path):
            try:
                # Load and process the file
                data = load_json(input_path)
                extracted_data = extract_tables_images(base_folder, filename, data)
                
                # Create output filename
                output_path = os.path.join(output_folder, f'{filename}.tables.json')
                
                # Save to JSON file
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(extracted_data, f, ensure_ascii=False, indent=4)
                
                print(f"Processed {filename} -> {os.path.basename(output_path)}")
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                continue

def main():
    """Main function to parse arguments and process files."""
    parser = argparse.ArgumentParser(
        description='Extract tables and images from PDF OCR output files.'
    )
    parser.add_argument('--input', '-i', required=True, help='Input directory containing mineru  files')
    parser.add_argument('--output', '-o', required=True, help='Output directory for converted files')
    
    args = parser.parse_args()
    
    process_folder(args.input, args.output)

if __name__ == '__main__':
    main() 