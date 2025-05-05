#!/usr/bin/env python3
"""
Table Image Extractor - Extracts table images from PDFs based on coordinates

This script processes JSON files containing table data with coordinates
and extracts the corresponding table images from PDF files.
"""

import json
import os
import argparse
from pathlib import Path
import fitz  # PyMuPDF
import glob
from pdf2image import convert_from_path
from PIL import Image
from tqdm import tqdm

def extract_tables_from_pdf(pdf_path, json_path, output_dir):
    """
    Extract table images from PDF based on coordinates in JSON file
    
    Args:
        pdf_path: Path to the PDF file
        json_path: Path to the JSON file containing table coordinates
        output_dir: Directory to save extracted images
    """
    # Get the JSON filename without extension for creating subdirectory
    json_basename = os.path.basename(json_path).replace('.blocks.tables.json', '')
    
    # Create specific output directory for this JSON file
    json_output_dir = os.path.join(output_dir, json_basename, "images")
    os.makedirs(json_output_dir, exist_ok=True)
    
    # Load JSON data
    with open(json_path, 'r', encoding='utf-8') as f:
        tables_data = json.load(f)
    
    # Set DPI - this affects the resolution of the output images
    dpi = 72
    
    # Convert PDF to images
    images = convert_from_path(pdf_path, dpi=dpi)
    
    # Create a list to track valid tables
    valid_tables = []
    
    # List to store data for val.txt format
    table_rec_data = []
    
    # Process each table entry
    for i, table in enumerate(tables_data):
        try:
            page_num = table.get('page')
            if page_num is None:
                print(f"Warning: No page number for table {i} in {json_path}. Skipping.")
                continue
                
            # PDF pages are 0-indexed in the list
            page_index = page_num - 1
            
            # Check if page index is valid
            if page_index < 0 or page_index >= len(images):
                print(f"Warning: Page {page_num} does not exist in the PDF {pdf_path}. Skipping table {i}.")
                continue
            
            # Get coordinates
            location = table.get('coordinates', {}).get('location', [])
            if not location or not location[0]:
                # Try alternative key structure
                location = table.get('text_location', {}).get('location', [])
                if not location or not location[0]:
                    print(f"Warning: No valid coordinates for table {i} on page {page_num} in {json_path}. Skipping.")
                    continue
            
            # Get the page image
            page_image = images[page_index]
            actual_width = page_image.width
            actual_height = page_image.height
            
            # Determine coordinate format and extract values
            coords = location[0]

            # Check if we have 4 values in the coordinates
            if len(coords) == 4:
                x1, y1, x2, y2 = coords # azure original coordinate is at the bottom left page
                y1 = actual_height - y1
                y2 = actual_height - y2
            else:
                print(f"Warning: Unexpected coordinate format for table {i}. Skipping.")
                continue
            
            # Calculate scaling factor based on DPI
            # Assuming coordinates are in points (72 points = 1 inch)
            scale_factor = dpi / 72
            
            # Scale the coordinates
            x1 = int(x1 * scale_factor)
            y1 = int(y1 * scale_factor)
            x2 = int(x2 * scale_factor)
            y2 = int(y2 * scale_factor)
            
            # Make sure coordinates are within image bounds
            x1_scaled = max(0, min(x1, actual_width - 1))
            y1_scaled = max(0, min(y1, actual_height - 1))
            x2_scaled = max(0, min(x2, actual_width - 1))
            y2_scaled = max(0, min(y2, actual_height - 1))
            
            # Check if the crop area is valid
            if x2_scaled <= x1_scaled or y2_scaled <= y1_scaled:
                print(f"Warning: Invalid crop area for table {i}: ({x1_scaled}, {y1_scaled}, {x2_scaled}, {y2_scaled}). Skipping.")
                continue
                
            # Crop the image to extract the table
            table_image = page_image.crop((x1_scaled, y1_scaled, x2_scaled, y2_scaled))
            
            # Generate a filename
             # Generate a filename
            table_id = table.get('id', f'table_{i}')
            filename = f"{json_basename}_{table_id}.png"
            output_path = os.path.join(json_output_dir, filename)
            
            # Save the image
            table_image.save(output_path)
            
            # Update the JSON with the image path
            table['img_path'] = output_path
            
            # Extract HTML structure from the table data
            if 'sentence' in table and table['sentence'].startswith('<table>'):
                # Extract HTML structure and cell data from the sentence field
                html_content = table['sentence']
                html_structure, cell_data, original_html = extract_html_from_table_sentence(html_content, table_image.width, table_image.height)
                
                if html_structure:
                    table_rec_data.append({
                        "image_name": filename,
                        "html_structure": html_structure,
                        "cell_data": cell_data,
                        "gt": original_html
                    })
            
            # # Add this table to the valid tables list
            # valid_tables.append(table)
                    valid_tables.append(table)
                    
        except Exception as e:
            print(f"Error processing table {i} in {json_path}: {e}")
            continue
    
    # Save the updated JSON with only valid tables
    updated_json_dir = os.path.dirname(json_output_dir)
    updated_json_path = os.path.join(updated_json_dir, f"{json_basename}.blocks.tables_with_images.json")
    
    with open(updated_json_path, 'w', encoding='utf-8') as f:
        json.dump(valid_tables, f, ensure_ascii=False, indent=4)
    
    # Generate val.txt format file if we have table_rec_data
    if table_rec_data:
        val_txt_path = os.path.join(updated_json_dir, f"{json_basename}.blocks.tables_with_images.txt")
        generate_table_rec_dataset(val_txt_path, table_rec_data)
    
    print(f"Extracted {len(valid_tables)} valid table images from {pdf_path}")
    print(f"Updated JSON saved to {updated_json_path}")
    
    return len(valid_tables)


def extract_html_from_table_sentence(html_content, img_width, img_height):
    """
    Extract HTML structure and cell data from the table sentence
    
    Args:
        html_content: HTML content from the sentence field
        img_width: Width of the table image
        img_height: Height of the table image
        
    Returns:
        tuple: (html_structure, cell_data)
    """
    try:
        
        # Extract the HTML structure tokens
        html_structure = []
        
        # Remove the outer <table> tags
        inner_html = html_content.replace('<table>', '').replace('</table>', '')
        
        # Store the original HTML content for the gt field
        original_html = inner_html 

        # Extract the HTML from <html><body><table>...</table></body></html>
        if '<html>' in inner_html and '</html>' in inner_html:
            inner_html = inner_html.split('<html>')[1].split('</html>')[0]
            inner_html = inner_html.replace('<body>', '').replace('</body>', '')
            inner_html = inner_html.replace('<table>', '').replace('</table>', '')
        
        # Parse the HTML to extract structure and cell content
        html_structure = []
        cell_data = []
        
        # Split the HTML into tokens
        current_token = ""
        in_tag = False
        in_attr = False
        tokens = []
        
        for char in inner_html:
            if char == '<':
                if current_token.strip():
                    tokens.append(current_token)
                current_token = "<"
                in_tag = True
            elif char == '>' and in_tag:
                current_token += ">"
                tokens.append(current_token)
                current_token = ""
                in_tag = False
            elif char == '"' and in_tag:
                current_token += char
                in_attr = not in_attr
            elif char == ' ' and in_tag and not in_attr:
                if current_token.strip():
                    tokens.append(current_token)
                    current_token = " "
                else:
                    current_token += char
            else:
                current_token += char
        
        if current_token.strip():
            tokens.append(current_token)
        
        # Clean up tokens and separate structure from content
        html_structure = []
        cell_contents = []
        current_cell_content = ""
        
        for token in tokens:
            if token.startswith('<'):
                if current_cell_content.strip():
                    cell_contents.append(current_cell_content.strip())
                    current_cell_content = ""
                html_structure.append(token)
            else:
                current_cell_content += token
        
        if current_cell_content.strip():
            cell_contents.append(current_cell_content.strip())
        
        # Count rows and columns to create a grid layout
        rows = inner_html.count('<tr>')
        
        # Calculate approximate cell dimensions
        row_heights = [img_height / rows] * rows
        
        # Extract row positions
        row_positions = []
        start_pos = 0
        while True:
            pos = inner_html.find('<tr>', start_pos)
            if pos == -1:
                break
            row_positions.append(pos)
            start_pos = pos + 4
        
        # Create cell data with proper content and bounding boxes
        cell_index = 0
        y_offset = 0
        
        for row_idx in range(rows):
            # Find all <td> tags in this row
            if row_idx < len(row_positions):
                row_start = row_positions[row_idx]
                row_end = inner_html.find('</tr>', row_start) + 5
                if row_end > row_start:
                    row_html = inner_html[row_start:row_end]
                else:
                    # Fallback if we can't find the end of the row
                    row_html = inner_html[row_start:]
            else:
                # Fallback if we can't find the row
                row_html = ""
            
            # Count columns in this row
            cols_in_row = row_html.count('<td')
            col_width = img_width / cols_in_row if cols_in_row > 0 else img_width
            
            x_offset = 0
            for col_idx in range(cols_in_row):
                # Extract cell content
                if cell_index < len(cell_contents):
                    cell_content = cell_contents[cell_index]
                    # Split content into tokens (characters for non-Latin text)
                    tokens = list(cell_content)
                else:
                    tokens = []
                
                # Create bounding box
                x1 = x_offset
                y1 = y_offset
                x2 = x_offset + col_width
                y2 = y_offset + row_heights[row_idx]
                
                bbox = [[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]]
                
                cell_data.append({
                    "tokens": tokens,
                    "bbox": bbox
                })
                
                x_offset += col_width
                cell_index += 1
            
            y_offset += row_heights[row_idx]
        
        return html_structure, cell_data, original_html
    
    except Exception as e:
        print(f"Error extracting HTML structure: {e}")
        return None, None, None

def generate_table_rec_dataset(output_file, html_data):
    """
    Generate a dataset file in the format of val.txt for table recognition.
    
    Args:
        output_file (str): Path to the output file
        html_data (list): List of dictionaries containing HTML structure and cell data
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in html_data:
            # Create the JSON object with filename and html structure
            data = {
                "filename": os.path.join("images", item["image_name"]),
                "html": {
                    "structure": {"tokens": item["html_structure"]},
                    "cells": item["cell_data"]
                },
                "gt": item["gt"]  # Add the gt field with the original HTML content
            }
            
            # Write the JSON object as a line in the output file
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    print(f"Table recognition dataset generated at {output_file}")

def process_all_files(json_dir, pdf_dir, output_dir):
    """
    Process all JSON files in the specified directory and extract tables from corresponding PDFs
    
    Args:
        json_dir: Directory containing JSON files with table coordinates
        pdf_dir: Directory containing PDF files
        output_dir: Directory to save extracted images
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all JSON files
    json_files = glob.glob(os.path.join(json_dir, "*.blocks.tables.json"))
    
    if not json_files:
        print(f"No JSON files found in {json_dir}")
        return
    
    print(f"Found {len(json_files)} JSON files")
    
    # Process each JSON file
    total_tables = 0
    for json_file in tqdm(json_files, desc="Processing files"):
        # Get the corresponding PDF file
        pdf_basename = os.path.basename(json_file).replace('.blocks.tables.json', '.pdf')
        pdf_file = os.path.join(pdf_dir, pdf_basename)
        
        if not os.path.exists(pdf_file):
            print(f"PDF file {pdf_file} not found. Skipping {json_file}")
            continue
        
        # Extract tables from the PDF
        num_tables = extract_tables_from_pdf(pdf_file, json_file, output_dir)
        total_tables += num_tables
    
    print(f"Total tables extracted: {total_tables}")

def main():
    parser = argparse.ArgumentParser(description='Extract table images from PDFs based on coordinates in JSON files')
    parser.add_argument('--json_dir', required=True, help='Directory containing JSON files with table coordinates')
    parser.add_argument('--pdf_dir', required=True, help='Directory containing PDF files')
    parser.add_argument('--output_dir', required=True, help='Directory to save extracted images')
    
    args = parser.parse_args()
    
    process_all_files(args.json_dir, args.pdf_dir, args.output_dir)

if __name__ == "__main__":
    main()