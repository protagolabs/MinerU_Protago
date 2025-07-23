#!/usr/bin/env python3
"""
Table Format Converter

This script converts Azure table format to extracted format by processing input files
and saving the results to an output directory. It can also extract table images from PDFs.
"""

import os
import json
import argparse
from pathlib import Path
import fitz  # PyMuPDF



def convert_azure_to_extracted_format(azure_file, pdf_folder=None):
    """Convert Azure table format to match the extracted format.
    
    Args:
        azure_file (str): Path to the input Azure format file
        pdf_folder (str, optional): Path to folder containing PDFs for image extraction
        
    Returns:
        list: List of converted tables in the extracted format
    """
    converted_tables = []
    
    # Get base name for PDF file
    base_name = Path(azure_file).stem.split('.')[0]  # Remove .blocks.txt
    pdf_path = None if pdf_folder is None else os.path.join(pdf_folder, f"{base_name}.pdf")
    
    # Create image output directory if needed
    if pdf_folder:
        img_dir = os.path.join(os.path.dirname(azure_file), "table_images")
        os.makedirs(img_dir, exist_ok=True)
    
    with open(azure_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            
            # Only process entries that are tables
            if data.get('type') == 'table':
                table_content = data['sentence']
                img_path = ""
                # Constants
                PADDING_RATIO = 0.0
                # Extract table image if PDF folder is provided
                if pdf_folder and 'text_location' in data:
                    locations = data['text_location']['location']
                    if locations:
                        try:
                            # Get bounding box coordinates
                            # Azure Form Recognizer returns multiple segments, merge them
                            # Each location is [x_min, y_max, x_max, y_min]

                            x_min = min(float(loc[0]) for loc in locations)
                            y_max = max(float(loc[1]) for loc in locations)
                            x_max = max(float(loc[2]) for loc in locations)
                            y_min = min(float(loc[3]) for loc in locations)
                            
                            
                            # Optional: Get page size dynamically
                            import fitz
                            doc = fitz.open(pdf_path)
                            page = doc.load_page(data['page'] - 1)
                            page_width, page_height = page.rect.width, page.rect.height

                            # print(f"page_width: {page_width}, page_height: {page_height}")
                            # Clamp to page size
                            x_max = min(page_width, x_max)
                            y_max = min(page_height, y_max)
                            x_min = max(0, x_min)
                            y_min = max(0, y_min)

                            # Create image filename
                            img_filename = f"{base_name}_page{data['page']}_table{len(converted_tables)}.png"
                            img_path = os.path.join("table_images", img_filename)
                            full_img_path = os.path.join(os.path.dirname(azure_file), img_path)

                            # Extract image

                            # success = extract_table_image(pdf_path, data['page'], [x_min, y_max, x_max, y_min], full_img_path)
                            doc = fitz.open(pdf_path)

                            # Select the page (assuming text is on the first page, index 0)
                            page = doc[data['page']- 1]

                            # print(f"x_min: {x_min}, y_max: {y_max}, x_max: {x_max}, y_min: {y_min}")
                            # Define the rectangle
                            x0, y1, x1, y0 = [x_min, page_height - y_max, x_max, page_height - y_min]
                            
                            # Ensure coordinates are in correct order and form a valid rectangle
                            x0, x1 = min(x0, x1), max(x0, x1)
                            y0, y1 = min(y0, y1), max(y0, y1)
                            
                            # Ensure minimum dimensions (at least 10 pixels)
                            if x1 - x0 < 10:
                                center = (x0 + x1) / 2
                                x0 = max(0, center - 5)
                                x1 = min(page_width, center + 5)
                            if y1 - y0 < 10:
                                center = (y0 + y1) / 2
                                y0 = max(0, center - 5)
                                y1 = min(page_height, center + 5)
                            



                            rect = fitz.Rect(x0, y0, x1, y1)
                            
                            try:
                                # Crop the rectangle and render it to an image
                                # Add matrix for better resolution
                                zoom = 2  # Increase resolution
                                mat = fitz.Matrix(zoom, zoom)
                                pix = page.get_pixmap(matrix=mat, clip=rect)
                                
                                # Save as PNG
                                pix.save(full_img_path)
                                # print(f"Successfully saved image with dimensions: {pix.width}x{pix.height}")
                                img_path = full_img_path
                            except Exception as e:
                                print(f"Error saving image: {e}")
                                print(f"Rectangle dimensions: {rect}")
                                img_path = ""  # Reset image path if saving failed

                        except (ValueError, TypeError) as e:
                            print(f"Error processing coordinates in {azure_file}: {e}")
                            img_path = ""
                
                # Add HTML body tags to match extracted format
                formatted_table = f"<html><body>{table_content}</body></html>"
                
                converted_table = {
                    "page": data['page'],
                    "img_path": img_path,
                    "types": "table",
                    "sentence": formatted_table
                }
                converted_tables.append(converted_table)
    
    return converted_tables

def process_folder(input_folder, output_folder, pdf_folder=None):
    """Process all files in input folder and save results to output folder.
    
    Args:
        input_folder (str): Path to the input directory containing Azure format files
        output_folder (str): Path to the output directory for converted files
        pdf_folder (str, optional): Path to folder containing PDFs for image extraction
    """
    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Process each file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.blocks.txt'):
            input_path = os.path.join(input_folder, filename)
            
            # Create output filename
            output_filename = filename.replace('.blocks.txt', '.tables.json')
            output_path = os.path.join(output_folder, output_filename)
            
            # Convert tables
            converted_tables = convert_azure_to_extracted_format(input_path, pdf_folder)
            
            # Save to JSON file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(converted_tables, f, ensure_ascii=False, indent=4)
            
            print(f"Processed {filename} -> {output_filename}")

def main():
    parser = argparse.ArgumentParser(description='Convert Azure table format to extracted format')
    parser.add_argument('--input', '-i', required=True, help='Input directory containing Azure format files')
    parser.add_argument('--output', '-o', required=True, help='Output directory for converted files')
    parser.add_argument('--pdf', '-p', help='Directory containing PDF files for image extraction')
    
    args = parser.parse_args()
    
    process_folder(args.input, args.output, args.pdf)

if __name__ == '__main__':
    main() 