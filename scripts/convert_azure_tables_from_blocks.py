#!/usr/bin/env python3
"""
Table Extractor - Extracts tables from Azure OCR blocks format

This script processes Azure OCR output files in blocks format and extracts
tables into a standardized format for further analysis.
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any


def extract_tables_from_blocks(blocks_file: str) -> List[Dict[str, Any]]:
    """Extract tables from Azure OCR blocks format.
    
    Args:
        blocks_file: Path to the Azure OCR blocks output file
        
    Returns:
        List of extracted tables in the standardized format with coordinates
    """
    extracted_tables = []
    
    with open(blocks_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
        for line in lines:
            try:
                data = json.loads(line.strip())
                
                # Skip empty lines or non-JSON data
                if not data:
                    continue
                
                sentence = data.get('sentence', '')
                page = data.get('page', 0)
                
                # Skip if there's no table content
                if '<table>' not in sentence or '</table>' not in sentence:
                    continue
                
                # Get coordinates from text_location if available
                coordinates = data.get('text_location', {})
                
                # Extract all tables from the sentence
                start_idx = 0
                while True:
                    start = sentence.find('<table>', start_idx)
                    if start == -1:
                        break
                    
                    end = sentence.find('</table>', start)
                    if end == -1:
                        break
                    
                    # Include the closing tag
                    end += 8  # Length of '</table>'
                    table_content = sentence[start:end]
                    
                    # Add HTML body tags to match extracted format
                    # formatted_table = f"<table><html><body>{table_content}</body></html></table>"
                    formatted_table = table_content
                    extracted_table = {
                        "page": page,
                        "img_path": "",
                        "types": "table",
                        "sentence": formatted_table,
                        "id": data.get('id', ''),
                        "seq_no": data.get('seq_no', 0),
                        "coordinates": coordinates  # Add coordinates
                    }
                    extracted_tables.append(extracted_table)
                    
                    # Move to the next potential table
                    start_idx = end
                
            except json.JSONDecodeError:
                continue  # Skip lines that aren't valid JSON
    
    return extracted_tables


def process_folder(input_folder: str, output_folder: str) -> None:
    """Process all files in input folder and save results to output folder.
    
    Args:
        input_folder: Path to the folder containing Azure OCR blocks files
        output_folder: Path to save the extracted table files
    """
    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Get list of files to process
    files = [f for f in os.listdir(input_folder) if f.endswith('.blocks.txt')]
    total_files = len(files)
    
    print(f"Found {total_files} files to process")
    
    # Process each file in the input folder
    for i, filename in enumerate(files, 1):
        input_path = os.path.join(input_folder, filename)
        
        # Create output filename
        output_filename = filename.replace('.blocks.txt', '.blocks.tables.json')
        output_path = os.path.join(output_folder, output_filename)
        
        try:
            # Extract tables
            extracted_tables = extract_tables_from_blocks(input_path)
            
            # Save to JSON file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(extracted_tables, f, ensure_ascii=False, indent=4)
            
            print(f"[{i}/{total_files}] Processed {filename} -> {output_filename} ({len(extracted_tables)} tables extracted)")
            
        except Exception as e:
            print(f"[{i}/{total_files}] Error processing {filename}: {str(e)}")


def main():
    """Main function to parse arguments and run the extraction process."""
    parser = argparse.ArgumentParser(
        description="Extract tables from Azure OCR blocks format"
    )
    parser.add_argument(
        "-i", "--input", 
        required=True,
        help="Input folder containing Azure OCR blocks files"
    )
    parser.add_argument(
        "-o", "--output", 
        required=True,
        help="Output folder to save extracted table files"
    )
    
    args = parser.parse_args()
    
    print(f"Starting table extraction process")
    print(f"Input folder: {args.input}")
    print(f"Output folder: {args.output}")
    
    process_folder(args.input, args.output)
    
    print("Extraction process completed")


if __name__ == "__main__":
    main()