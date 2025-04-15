import json
import os
import re
from typing import Dict, List, Any, Tuple
from html.parser import HTMLParser
from tqdm import tqdm
import shutil

class HTMLStructureParser(HTMLParser):
    """HTML parser to extract structure tokens from HTML string"""
    
    def __init__(self):
        super().__init__()
        self.tokens = []
        self.cell_count = 0
        # Tags to exclude from structure tokens
        self.exclude_tags = {'html', 'body', 'table'}
    
    def handle_starttag(self, tag, attrs):
        # Skip excluded tags
        if tag in self.exclude_tags:
            return
            
        # Track cell count for td and th tags
        if tag in ['td', 'th']:
            self.cell_count += 1
            
        # Add the opening tag
        if attrs:
            # If there are attributes, add the tag without closing bracket
            self.tokens.append(f"<{tag}")
            
            # Add each attribute as a separate token
            for attr_name, attr_value in attrs:
                self.tokens.append(f' {attr_name}="{attr_value}"')
            
            # Add the closing bracket as a separate token
            self.tokens.append(">")
        else:
            # If no attributes, add the complete tag
            self.tokens.append(f"<{tag}>")
    
    def handle_endtag(self, tag):
        # Skip excluded tags
        if tag in self.exclude_tags:
            return
            
        self.tokens.append(f"</{tag}>")
    
    def handle_data(self, data):
        # We don't include data in structure tokens
        pass

def process_table_json_to_html(input_json_path: str, output_json_path: str, combine_output: bool = False) -> None:
    """
    Process a JSON file containing table data and convert all tables to HTML format.
    
    Args:
        input_json_path: Path to the input JSON file containing table data
        output_dir: Directory to save the output text files with HTML content
        combine_output: If True, combine all tables into a single output file
    """
    # Load the JSON data
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)


    document_id = data.get("document_id")

    # Find all tables in the data
    tables = []
    if isinstance(data, list):
        tables = [table for table in data if "table_id" in table]
    elif isinstance(data, dict):
        if "table_id" in data:
            tables = [data]
        else:
            # Check if there's a tables list in the data
            for key, value in data.items():
                if isinstance(value, list) and value and isinstance(value[0], dict) and "table_id" in value[0]:
                    tables = value
                    break
    
    if not tables:
        print(f"No tables found in {input_json_path}")
        return 0, 0
    
    print(f"Found {len(tables)} tables in {input_json_path}")

    mismatch_count = 0
    all_outputs = []
    # Process each table
    for table_data in tables:
        table_id = table_data.get("table_id", 0)
        
        
        # Get the image path from the table data
        source_image_path = None
        if "image_paths" in table_data and table_data["image_paths"]:
            source_image_path = table_data["image_paths"][0]
        
        if not source_image_path:
            print(f"No image path found for table {table_id}")
            continue
        else:
            image_name = source_image_path.split("/")[-1]

            # renmae the image name to include the document id
            image_name = document_id + "_" + image_name

            image_path = os.path.join("images", image_name)


            source_image_path = os.path.join(args.input_dir, document_id,source_image_path)

            destination_image_path = os.path.join(args.output_dir, image_path)
            # copy the image to the images directory
            shutil.copy(source_image_path, destination_image_path)


            html_cells = []
            
            # Process cells from the table data
            for cell in table_data.get("cells", []):
                content = cell.get("content", "")
                
                bounding_region = cell.get("bounding_regions", [])[0]

                bbox = []

                if "cell_bbox" in bounding_region:
                    x1, y1, x2, y2 = bounding_region["cell_bbox"]

                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    bbox = [[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]]
                
                # Add cell content to html_cells
                if content:
                    # Split content into tokens (characters)
                    tokens = list(content)
                    cell_data = {
                        "tokens": tokens,
                        "bbox": bbox
                    }
                    html_cells.append(cell_data)
                else:
                    # Add empty cell
                    html_cells.append({"tokens": [], "bbox": bbox})
            

            # Check if HTML content is already available in the table data
            html_string = table_data.get("html", "")
            

            # print("html_string: ", html_string)
            
            # Parse the HTML to extract structure tokens
            parser = HTMLStructureParser()
            parser.feed(html_string)
            html_structure_tokens = parser.tokens
            
            # Verify that the number of cells matches the structure
            num_html_cells = len(html_cells)
            num_structure_cells = parser.cell_count
            
            if num_html_cells != num_structure_cells:
                print(f"Warning: Mismatch between number of cells ({num_html_cells}) and structure tokens ({num_structure_cells}) for table {table_id}")
                mismatch_count += 1  # Increment mismatch counter
                # Skip further processing for mismatched samples
                continue
                
            # Create the final output structure
            output_data = {
                "filename": image_path,
                "html": {
                    "structure": {
                        "tokens": html_structure_tokens
                    },
                    "cells": html_cells
                },
                "gt": html_string
            }



            
            # If combining output, add to the list
            if combine_output:
                all_outputs.append(output_data)
            else:
                # Create output filename
                base_name = os.path.basename(input_json_path).split('.')[0]
                
                # Extract page number if available
                page_number = table_data.get("page_number", 0)
                
                # Create output filename with the requested format
                # output_filename = f"{base_name}_table_{table_id}_page_{page_number}.txt"
                # output_path = os.path.join(output_dir, output_filename)
                output_path = output_json_path

                # Write the output to a text file
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(output_data, f, ensure_ascii=False)
                
                print(f"Processed table {table_id} on page {page_number} -> {output_path}")
    


    # If combining output, write all tables to a single file
    if combine_output and all_outputs:
        base_name = os.path.basename(input_json_path).split('.')[0]
        # print("base_name: ", base_name)
        # output_filename = f"{base_name.replace('_tables', '')}_all_tables.json"

        # output_path = os.path.join(output_dir, output_filename)
        output_path = output_json_path
        
        with open(output_path, 'w', encoding='utf-8') as f:
            # Write each table as a JSON object on a single line
            for table_data in all_outputs:
                # Convert table data to a compact JSON string (no pretty printing)
                json_line = json.dumps(table_data, ensure_ascii=False)
                f.write(json_line + '\n')
        
        # print(f"Combined {len(all_outputs)} tables into {output_path}, with each table on a separate line")

    return len(tables), mismatch_count
# Example usage

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process table JSON files to HTML format')
    parser.add_argument('--input_dir', type=str, required=True, 
                        help='Path to input directory containing subdirectories with JSON files')
    parser.add_argument('--output_dir', type=str, default='output', 
                        help='Directory to save output files')
    parser.add_argument('--combine', action='store_true', 
                        help='Combine all tables into a single output file per input file')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if input_dir is a directory
    if not os.path.isdir(args.input_dir):
        print(f"Error: {args.input_dir} is not a valid directory")
        exit(1)

    # Check if output_dir is a existing directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Created output directory: {args.output_dir}")

    if not os.path.exists(os.path.join(args.output_dir, "images")):
        os.makedirs(os.path.join(args.output_dir, "images"), exist_ok=True)
        print(f"Created images directory: {os.path.join(args.output_dir, 'images')}")

    total_tables = 0
    total_mismatches = 0
    # Process all subdirectories in the input directory
    for subdir_name in tqdm(os.listdir(args.input_dir), desc="Processing directories"):
        subdir_path = os.path.join(args.input_dir, subdir_name)
        
        # Skip if not a directory
        if not os.path.isdir(subdir_path):
            continue
            
        
        # Look for JSON files in the subdirectory
        json_files = [f for f in os.listdir(subdir_path) if f.endswith('.json')]
        for filename in json_files:
            input_file_path = os.path.join(subdir_path, filename)
            # print(f"Processing {input_file_path}...")

            output_path = os.path.join(args.output_dir, filename)

            # print(f"Processing {input_file_path}...")
            # Process the file and get counts
            tables_count, mismatches_count = process_table_json_to_html(input_file_path, output_path, combine_output=args.combine)
            
            # Update the totals
            total_tables += tables_count
            total_mismatches += mismatches_count
    
    print(f"Completed processing all subdirectories in {args.input_dir}")
    print(f"Total tables found: {total_tables}")
    print(f"Total tables skipped due to mismatches: {total_mismatches}")
    print(f"Percentage of tables with mismatches: {(total_mismatches/total_tables)*100:.2f}%" if total_tables > 0 else 0)




    # List to store paths of all generated JSON files

    all_json_files = [f for f in os.listdir(args.output_dir) if f.endswith('.json')]

    # all_json_files = all_json_files[:2]
    consolidated_file = os.path.join(args.output_dir, "final_output.txt")
    print(f"Consolidating all output files into {consolidated_file}...")
    
    with open(consolidated_file, 'w', encoding='utf-8') as outfile:
        for json_file in all_json_files:
            # Create the full path to the JSON file
            json_file_path = os.path.join(args.output_dir, json_file)
            if os.path.exists(json_file_path):
                with open(json_file_path, 'r', encoding='utf-8') as infile:
                    # Read each line and write to the consolidated file
                    for line in infile:
                        outfile.write(line)

