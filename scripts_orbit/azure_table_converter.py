#!/usr/bin/env python3
"""
Table Format Converter

This script converts Azure table format to extracted format by processing input files
and saving the results to an output directory.
"""

import os
import json
import argparse
from pathlib import Path

def convert_azure_to_extracted_format(azure_file):
    """Convert Azure table format to match the extracted format.
    
    Args:
        azure_file (str): Path to the input Azure format file
        
    Returns:
        list: List of converted tables in the extracted format
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
                        formatted_table = f"<table><html><body>{inner_table}</body></html></table>"
                        
                        converted_table = {
                            "page": data['page'],
                            "img_path": "",
                            "types": "table",
                            "sentence": formatted_table
                        }
                        converted_tables.append(converted_table)
    
    return converted_tables

def process_folder(input_folder, output_folder):
    """Process all files in input folder and save results to output folder.
    
    Args:
        input_folder (str): Path to the input directory containing Azure format files
        output_folder (str): Path to the output directory for converted files
    """
    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Process each file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):
            input_path = os.path.join(input_folder, filename)
            
            # Create output filename by replacing .txt with .json
            output_filename = filename.replace('.txt', '.tables.json')
            output_path = os.path.join(output_folder, output_filename)
            
            # Convert tables
            converted_tables = convert_azure_to_extracted_format(input_path)
            
            # Save to JSON file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(converted_tables, f, ensure_ascii=False, indent=4)
            
            print(f"Processed {filename} -> {output_filename}")

def main():
    parser = argparse.ArgumentParser(description='Convert Azure table format to extracted format')
    parser.add_argument('--input', '-i', required=True, help='Input directory containing Azure format files')
    parser.add_argument('--output', '-o', required=True, help='Output directory for converted files')
    
    args = parser.parse_args()
    
    process_folder(args.input, args.output)

if __name__ == '__main__':
    main() 