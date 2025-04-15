#!/usr/bin/env python3
"""
Table Converter - Converts Azure table format to extracted format

This script processes Azure OCR output files containing tables and converts them
to a standardized extracted format for further analysis.
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any


def convert_azure_to_extracted_format(azure_file: str) -> List[Dict[str, Any]]:
    """Convert Azure table format to match the extracted format.
    
    Args:
        azure_file: Path to the Azure OCR output file
        
    Returns:
        List of converted tables in the extracted format
    """
    converted_tables = []
    
    with open(azure_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            sentence = data['sentence']
            
            # Find all tables in the sentence
            table_starts = [i for i in range(len(sentence)) if sentence.startswith('<table>', i)]
            
            for start in table_starts:
                end = sentence.find('</table>', start) + 8
                if end > start:
                    table_content = sentence[start:end]
                    
                    # Check if there are multiple tables within the content
                    inner_tables = table_content.split('</table><table>')
                    
                    for inner_table in inner_tables:
                        # Clean up the table content
                        if not inner_table.startswith('<table>'):
                            inner_table = '<table>' + inner_table
                        if not inner_table.endswith('</table>'):
                            inner_table = inner_table + '</table>'
                            
                        # Add HTML body tags to match extracted format
                        # formatted_table = f"<table><html><body>{inner_table}</body></html></table>"
                        formatted_table = inner_table
                        
                        converted_table = {
                            "page": data['page'],
                            "img_path": "",
                            "types": "table",
                            "sentence": formatted_table
                        }
                        converted_tables.append(converted_table)
    
    return converted_tables


def process_folder(input_folder: str, output_folder: str) -> None:
    """Process all files in input folder and save results to output folder.
    
    Args:
        input_folder: Path to the folder containing Azure OCR output files
        output_folder: Path to save the converted table files
    """
    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Get list of files to process
    files = [f for f in os.listdir(input_folder) if f.endswith('.txt')]
    total_files = len(files)
    
    print(f"Found {total_files} files to process")
    
    # Process each file in the input folder
    for i, filename in enumerate(files, 1):
        input_path = os.path.join(input_folder, filename)
        
        # Create output filename by replacing .txt with .json
        output_filename = filename.replace('.txt', '.tables.json')
        output_path = os.path.join(output_folder, output_filename)
        
        try:
            # Convert tables
            converted_tables = convert_azure_to_extracted_format(input_path)
            
            # Save to JSON file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(converted_tables, f, ensure_ascii=False, indent=4)
            
            print(f"[{i}/{total_files}] Processed {filename} -> {output_filename}")
            
        except Exception as e:
            print(f"[{i}/{total_files}] Error processing {filename}: {str(e)}")


def main():
    """Main function to parse arguments and run the conversion process."""
    parser = argparse.ArgumentParser(
        description="Convert Azure OCR table format to extracted format"
    )
    parser.add_argument(
        "-i", "--input", 
        required=True,
        help="Input folder containing Azure OCR output files"
    )
    parser.add_argument(
        "-o", "--output", 
        required=True,
        help="Output folder to save converted table files"
    )
    
    args = parser.parse_args()
    
    print(f"Starting table conversion process")
    print(f"Input folder: {args.input}")
    print(f"Output folder: {args.output}")
    
    process_folder(args.input, args.output)
    
    print("Conversion process completed")


if __name__ == "__main__":
    main()