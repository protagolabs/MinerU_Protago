#!/usr/bin/env python3
"""
Image Extractor

This script processes PDF OCR output files to extract images information only.
It takes input and output directories as command line arguments, processes all
JSON files in the input directory structure, and concatenates all results into
a single JSON file.
"""

import os
import json
import argparse
import shutil
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

def extract_images_only(base_folder, filename, data):
    """Extract only images information from OCR data.
    
    Args:
        base_folder (str): Base directory containing the files
        filename (str): Name of the current file being processed
        data (dict): OCR data containing PDF information
        
    Returns:
        list: List of dictionaries containing extracted image information
    """
    formatted_output = []
    current_page = 1
    
    # Handle case where data is a list
    pdf_info = data.get("pdf_info", [])
    
    # Data is a list of pages
    for page_data in pdf_info:
        images = []
        
        # Extract images directly from the 'images' key
        if 'images' in page_data:
            images.extend(page_data['images'])
            
        # Also check preproc_blocks for image entities
        for block in page_data.get('preproc_blocks', []):
            if block.get('type') == 'image':
                images.append(block)
            
        # Also check discarded_blocks for any relevant information
        for block in page_data.get('discarded_blocks', []):
            if block.get('type') == 'image':
                images.append(block)

        # Format only images into content
        for image in images:
            image_path = None
            
            # Check if image_path is directly in the image object
            if image.get('image_path'):
                image_path = image['image_path']
            
            # If not found, look for it in nested blocks/lines/spans structure
            if not image_path and 'blocks' in image:
                for block in image['blocks']:
                    if 'lines' in block:
                        for line in block['lines']:
                            if 'spans' in line:
                                for span in line['spans']:
                                    if span.get('image_path'):
                                        image_path = span['image_path']
                                        break
                                if image_path:
                                    break
                            if image_path:
                                break
                        if image_path:
                            break
            
            # Add to output if image_path was found
            if image_path:
                formatted_output.append({
                    "page": current_page,
                    "img_path": image_path,
                    "types": image['type'],
                    "sentence": ""  # Empty for images
                })
        
        current_page += 1
    
    return formatted_output

def remove_duplicates(data_list, base_folder, output_folder):
    """Remove duplicate entries from the extracted data and copy unique images.
    
    Args:
        data_list (list): List of dictionaries containing extracted data
        base_folder (str): Base directory containing the original files
        output_folder (str): Output directory for processed results
        
    Returns:
        list: List with duplicates removed
    """
    seen = set()
    unique_data = []
    copied_images = set()
    
    # Create images subdirectory in output folder
    images_output_dir = os.path.join(output_folder, 'images')
    Path(images_output_dir).mkdir(parents=True, exist_ok=True)
    
    for item in data_list:
        # Create a unique identifier based on img_path, page, and source_file
        # This allows the same image to exist on different pages but removes exact duplicates
        identifier = (item.get('img_path', ''), item.get('page', 0), item.get('source_file', ''))
        
        if identifier not in seen:
            seen.add(identifier)
            
            # Copy the image file if it hasn't been copied yet
            img_path = item.get('img_path', '')
            if img_path and img_path not in copied_images:
                try:
                    # Construct the full source path
                    source_file = item.get('source_file', '')
                    source_img_path = os.path.join(base_folder, source_file, 'auto', 'images', os.path.basename(img_path))
                    
                    # Check if source file exists
                    if os.path.exists(source_img_path):
                        # Copy to output images directory
                        dest_path = os.path.join(images_output_dir, os.path.basename(img_path))
                        shutil.copy2(source_img_path, dest_path)
                        copied_images.add(img_path)
                        
                        # Update the img_path to point to the new location
                        item['img_path'] = f"images/{os.path.basename(img_path)}"
                        
                        print(f"Copied image: {os.path.basename(img_path)}")
                    else:
                        print(f"Warning: Image file not found: {source_img_path}")
                        
                except Exception as e:
                    print(f"Error copying image {img_path}: {str(e)}")
            
            unique_data.append(item)
    
    print(f"Total unique images copied: {len(copied_images)}")
    return unique_data

def process_folder(base_folder, output_folder):
    """Process all files in the specified folder structure.
    
    Args:
        base_folder (str): Input directory containing OCR files
        output_folder (str): Output directory for processed results
    """
    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Collect all extracted data
    all_extracted_data = []
    
    # List all subdirectories in the base folder
    for filename in os.listdir(base_folder):
        input_path = os.path.join(base_folder, filename, 'auto', f'{filename}_middle.json')
        
        if os.path.exists(input_path):
            try:
                # Load and process the file
                data = load_json(input_path)
                extracted_data = extract_images_only(base_folder, filename, data)
                
                # Add source filename to each extracted item
                for item in extracted_data:
                    item['source_file'] = filename
                
                # Add to the combined results
                all_extracted_data.extend(extracted_data)
                
                print(f"Processed {filename} -> Found {len(extracted_data)} images")
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                continue
    
    # Remove duplicates and copy images before saving
    original_count = len(all_extracted_data)
    all_extracted_data = remove_duplicates(all_extracted_data, base_folder, output_folder)
    deduplicated_count = len(all_extracted_data)
    
    if original_count != deduplicated_count:
        print(f"Removed {original_count - deduplicated_count} duplicate entries")
    
    # Save all results to a single JSON file
    output_path = os.path.join(output_folder, 'all_images.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_extracted_data, f, ensure_ascii=False, indent=4)
    
    print(f"\nTotal images extracted: {len(all_extracted_data)}")
    print(f"All results saved to: {os.path.basename(output_path)}")

def main():
    """Main function to parse arguments and process files."""
    parser = argparse.ArgumentParser(
        description='Extract images only from PDF OCR output files and concatenate into a single JSON file.'
    )
    parser.add_argument('--input', '-i', required=True, help='Input directory containing mineru files')
    parser.add_argument('--output', '-o', required=True, help='Output directory for the combined results file')
    
    args = parser.parse_args()
    
    process_folder(args.input, args.output)

if __name__ == '__main__':
    main() 