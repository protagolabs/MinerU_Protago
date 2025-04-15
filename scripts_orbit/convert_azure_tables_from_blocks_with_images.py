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
        
        try:
            # Check if the crop area is valid
            if x2_scaled <= x1_scaled or y2_scaled <= y1_scaled:
                print(f"Warning: Invalid crop area for table {i}: ({x1_scaled}, {y1_scaled}, {x2_scaled}, {y2_scaled}). Skipping.")
                continue
                
            # Crop the image to extract the table
            table_image = page_image.crop((x1_scaled, y1_scaled, x2_scaled, y2_scaled))
            
            # Generate a filename
            table_id = table.get('id', f'table_{i}')
            filename = f"{json_basename}_page_{page_num}_table_{table_id}.png"
            output_path = os.path.join(json_output_dir, filename)
            
            # Save the image
            table_image.save(output_path)
            # print(f"Saved table image to {output_path}")
            
            # Update the JSON with the image path
            table['img_path'] = output_path
            
            # Add this table to the valid tables list
            valid_tables.append(table)

            # Extract HTML structure and cell data if available
            if 'html_structure' in table and 'cells' in table:
                table_rec_data.append({
                    "image_name": filename,
                    "html_structure": table['html_structure'],
                    "cell_data": table['cells']
                })
            else:
                print(f"Warning: No HTML structure or cell data for table {i} in {json_path}. Skipping.")

        except Exception as e:
            print(f"Error processing table {i}: {e}")
            continue
    
    # Save the updated JSON with only valid tables and image paths
    updated_json_dir = os.path.join(output_dir, json_basename)
    updated_json_path = os.path.join(updated_json_dir, os.path.basename(json_path).replace('.json', '_with_images.json'))
    with open(updated_json_path, 'w', encoding='utf-8') as f:
        json.dump(valid_tables, f, ensure_ascii=False, indent=4)
    
    # Generate val.txt format file if we have table_rec_data
    if table_rec_data:
        val_txt_path = os.path.join(updated_json_dir, 'val.txt')
        generate_table_rec_dataset(json_output_dir, val_txt_path, table_rec_data)
    
    print(f"Extracted {len(valid_tables)} valid table images from {pdf_path}")
    print(f"Updated JSON saved to {updated_json_path}")
    
    
    return len(valid_tables)

def generate_table_rec_dataset(images_dir, output_file, html_data):
    """
    Generate a dataset file in the format of val.txt for table recognition.
    
    Args:
        images_dir (str): Directory containing the images
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
                }
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
    # Get all JSON files
    json_files = glob.glob(os.path.join(json_dir, "*.blocks.tables.json"))
    
    if not json_files:
        print(f"No JSON files found in {json_dir}")
        return
    
    total_tables = 0
    processed_files = 0
    
    for json_path in tqdm(json_files, desc="Processing JSON files"):
        # Extract the base filename without extension
        json_basename = os.path.basename(json_path)
        pdf_basename = json_basename.replace('.blocks.tables.json', '.pdf')
        pdf_path = os.path.join(pdf_dir, pdf_basename)
        
        # Check if corresponding PDF exists
        if not os.path.exists(pdf_path):
            print(f"Warning: Corresponding PDF file {pdf_path} not found for {json_path}. Skipping.")
            continue
        
        # print(f"Processing {json_path} with {pdf_path}...")
        tables_count = extract_tables_from_pdf(pdf_path, json_path, output_dir)
        total_tables += tables_count
        processed_files += 1
    
    print(f"Completed processing {processed_files} files, extracted {total_tables} tables in total.")

def main():
    """Main function to parse arguments and run the extraction process."""
    parser = argparse.ArgumentParser(
        description="Extract table images from PDFs based on coordinates"
    )
    parser.add_argument(
        "-j", "--json_dir", 
        default="inputs/export_pdf/tables_from_blocks/",
        help="Directory containing JSON files with table coordinates"
    )
    parser.add_argument(
        "-p", "--pdf_dir", 
        default="inputs/export_pdf/pdf/",
        help="Directory containing PDF files"
    )
    parser.add_argument(
        "-o", "--output", 
        default="outputs/table_images",
        help="Output directory to save extracted images"
    )
    parser.add_argument(
        "--single", 
        action="store_true",
        help="Process a single file instead of all files in the directory"
    )
    parser.add_argument(
        "--pdf", 
        help="Path to a specific PDF file (used with --single)"
    )
    parser.add_argument(
        "--json", 
        help="Path to a specific JSON file (used with --single)"
    )
    
    args = parser.parse_args()
    
    print(f"Starting table image extraction process")
    
    if args.single and args.pdf and args.json:
        print(f"Processing single file:")
        print(f"PDF file: {args.pdf}")
        print(f"JSON file: {args.json}")
        extract_tables_from_pdf(args.pdf, args.json, args.output)
    else:
        print(f"Processing all files:")
        print(f"JSON directory: {args.json_dir}")
        print(f"PDF directory: {args.pdf_dir}")
        print(f"Output directory: {args.output}")
        process_all_files(args.json_dir, args.pdf_dir, args.output)
    
    print("Extraction process completed")

if __name__ == "__main__":
    main()