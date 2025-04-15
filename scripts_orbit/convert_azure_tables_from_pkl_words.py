#!/usr/bin/env python3
"""
Azure Document Intelligence Table Extractor and Image Cropper

This script extracts tables from Azure Document Intelligence API results
stored in pickle files, saves them as JSON, and crops the table images
from the original PDF files.

Usage:
    python azure_table_extractor.py --input /path/to/pkl/files --output /path/to/output/dir --pdf-dir /path/to/pdf/files
    python azure_table_extractor.py --file /path/to/single/file.pkl --output /path/to/output.json --pdf-file /path/to/pdf/file.pdf
"""

import os
import sys
import pickle
import json
import argparse
from typing import Dict, List, Any, Optional
import fitz  # PyMuPDF
import numpy as np
from PIL import Image, ImageDraw
import io
from tqdm import tqdm

def load_pkl_file(file_path: str) -> Any:
    """Load data from a pickle file."""
    try:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except Exception as e:
        raise Exception(f"Error loading pickle file: {str(e)}")
    

def convert_table_to_html(table):
    """Convert an Azure table to HTML format with explicit colspan and rowspan attributes"""
    html = ['<table>']
    
    # Get row and column count
    row_count = table.row_count if hasattr(table, 'row_count') else 0
    column_count = table.column_count if hasattr(table, 'column_count') else 0
    
    # Create a 2D grid to place cells correctly
    grid = [[None for _ in range(column_count)] for _ in range(row_count)]
    
    # First pass: Place cells in the grid and mark their spans
    for cell in table.cells:
        row_idx = cell.row_index
        col_idx = cell.column_index
        row_span = cell.row_span
        col_span = cell.column_span
        content = cell.content
        
        # Place the cell in the grid
        grid[row_idx][col_idx] = {
            'content': content,
            'row_span': row_span,
            'col_span': col_span,
            'is_main_cell': True  # This is the top-left cell of a possibly spanning cell
        }
        
        # Mark spanned cells
        for r in range(row_idx, min(row_idx + row_span, row_count)):
            for c in range(col_idx, min(col_idx + col_span, column_count)):
                if r != row_idx or c != col_idx:
                    grid[r][c] = {
                        'is_main_cell': False,  # This cell is covered by another cell's span
                        'main_cell_row': row_idx,
                        'main_cell_col': col_idx
                    }
    
    # Second pass: Generate HTML from the grid
    for row_idx in range(row_count):
        html.append('\t<tr>')
        for col_idx in range(column_count):
            cell = grid[row_idx][col_idx]
            
            # Skip if cell is None or not a main cell
            if cell is None or not cell.get('is_main_cell', False):
                continue
            
            content = cell['content']
            row_span = cell['row_span']
            col_span = cell['col_span']
            
            # Always add colspan and rowspan attributes
            colspan_attr = f' colspan="{col_span}"'
            rowspan_attr = f' rowspan="{row_span}"'
            
            # Add the cell to HTML
            html.append(f'\t\t<td{colspan_attr}{rowspan_attr}>{content}</td>')
        
        html.append('\t</tr>')
    
    html.append('</table>')
    return '\n'.join(html)

def extract_tables_to_json(data: Any, output_dir: str, filename: str, pdf_path: str = None, visualize: bool = False) -> List[Dict]:
    """Extract tables from Azure Document Intelligence data and save as JSON.
    Also crop table images if pdf_path is provided.
    
    Args:
        data: The Azure Document Intelligence result object
        output_dir: Directory where output will be saved
        filename: Base filename without extension
        pdf_path: Path to the original PDF file (optional)
        visualize: Whether to visualize bounding boxes on images
        
    Returns:
        List of extracted tables
    """
    # Create document-specific directory
    doc_dir = os.path.join(output_dir, filename)
    os.makedirs(doc_dir, exist_ok=True)
    
    # Define output paths
    json_path = os.path.join(doc_dir, f"{filename}_tables.json")
    images_dir = os.path.join(doc_dir, "images")
    visualizations_dir = os.path.join(doc_dir, "visualizations")
    
    result = {
        "document_id": filename,
        "model_id": data.model_id if hasattr(data, 'model_id') else "unknown",
        "api_version": data.api_version if hasattr(data, 'api_version') else "unknown",
        "pages": [],
        "tables": []
    }
    
    # Extract page information
    if hasattr(data, 'pages'):
        for page in data.pages:
            page_info = {
                "page_number": page.page_number,
                "width": page.width,
                "height": page.height,
                "unit": page.unit,
                "angle": page.angle if hasattr(page, 'angle') else None
            }
            result["pages"].append(page_info)
    
    # Open PDF if path is provided
    pdf_doc = None
    
    if pdf_path and os.path.exists(pdf_path):
        try:
            pdf_doc = fitz.open(pdf_path)
        except Exception as e:
            print(f"Warning: Could not open PDF file {pdf_path}: {str(e)}")
    
    # Create images directory if PDF is available
    if pdf_doc:
        os.makedirs(images_dir, exist_ok=True)
        if visualize:
            os.makedirs(visualizations_dir, exist_ok=True)
    
    page = pdf_doc[0] if pdf_doc else None
    
    # Extract all words from the document and organize by span offset
    words_by_offset = {}
    if hasattr(data, 'pages'):
        for page in data.pages:
            if hasattr(page, 'words'):
                for word in page.words:
                    if hasattr(word, 'span') and hasattr(word.span, 'offset'):
                        offset = word.span.offset
                        word_data = {
                            "content": word.content,
                            "page_number": page.page_number,
                            "confidence": word.confidence if hasattr(word, 'confidence') else None,
                            "span": {
                                "offset": offset,
                                "length": word.span.length if hasattr(word.span, 'length') else None
                            },
                            "word_bbox": None
                        }
                        
                        # Extract bounding polygon for the word
                        if hasattr(word, 'polygon'):
                            word_data["polygon"] = [{"x": point.x, "y": point.y} for point in word.polygon]
                        elif hasattr(word, 'bounding_polygon'):
                            word_data["polygon"] = [{"x": point.x, "y": point.y} for point in word.bounding_polygon]
                        
                        # Extract bounding box for the word
                        if hasattr(word, 'bounding_box'):
                            word_data["bounding_box"] = {
                                "x": word.bounding_box.x,
                                "y": word.bounding_box.y,
                                "width": word.bounding_box.width,
                                "height": word.bounding_box.height
                            }
                        
                        words_by_offset[offset] = word_data

    # Check if tables exist in the data
    if hasattr(data, 'tables') and data.tables:
        for table_idx, table in enumerate(data.tables):
            # if table_idx == 0:
            #     print("table: ", table)
            # Initialize table data with ordered fields
            table_data = {
                "table_id": table_idx,
                "row_count": table.row_count,
                "column_count": table.column_count,
            }
            
            # Convert table to HTML and add to table data
            html_content = convert_table_to_html(table)
            table_data["html"] = html_content
            
            # Extract page number if available
            table_page_number = None
            if hasattr(table, 'bounding_regions') and table.bounding_regions:
                for region in table.bounding_regions:
                    if hasattr(region, 'page_number'):
                        table_page_number = region.page_number
                        table_data["page_number"] = table_page_number
                        if pdf_doc:
                            page = pdf_doc[table_page_number - 1] 
                            table_data["page_width"] = page.rect.width
                            table_data["page_height"] = page.rect.height
                        break
            
            # Extract table spans if available
            if hasattr(table, 'spans') and table.spans:
                table_data["spans"] = []
                for span in table.spans:
                    span_data = {
                        "offset": span.offset,
                        "length": span.length
                    }
                    table_data["spans"].append(span_data)
            
            # Extract bounding box for the whole table if available
            if hasattr(table, 'bounding_regions') and table.bounding_regions:
                table_data["bbox"] = []
                for region in table.bounding_regions:
                    region_data = {
                        "page_number": region.page_number if hasattr(region, 'page_number') else None,
                        "polygon": [{"x": point.x, "y": point.y} for point in region.polygon]
                    }
                    
                    # Crop table image if PDF is available
                    if pdf_doc and hasattr(region, 'page_number'):
                        page_number = region.page_number - 1  # PDF pages are 0-indexed
                        if 0 <= page_number < len(pdf_doc):
                            try:
                                # Get page dimensions
                                pdf_page = pdf_doc[page_number]
                                page_width = pdf_page.rect.width
                                page_height = pdf_page.rect.height
                                
                                # Find page in Azure data to get dimensions
                                azure_page = None
                                for p in result["pages"]:
                                    if p["page_number"] == region.page_number:
                                        azure_page = p
                                        break
                                
                                if azure_page:
                                    # Calculate scaling factors between Azure coordinates and PDF, should around 72dpi
                                    scale_x = page_width / azure_page["width"]
                                    scale_y = page_height / azure_page["height"]
                                    
                                    # Convert polygon points to PDF coordinates
                                    points = []
                                    pixel_polygon = []
                                    for point in region.polygon:
                                        px = point.x * scale_x
                                        py = point.y * scale_y
                                        points.append((px, py))
                                        pixel_polygon.append({"x": px, "y": py})
                                    
                                    # Calculate bounding box
                                    x_values = [p[0] for p in points]
                                    y_values = [p[1] for p in points]
                                    x0, y0 = min(x_values), min(y_values)
                                    x1, y1 = max(x_values), max(y_values)
                                    
                                    # Add pixel coordinates to the region data
                                    region_data["polygon_pixels"] = pixel_polygon

                                    table_data["bbox"] = [x0, y0, x1, y1]
                                    # Create rectangle for cropping
                                    rect = fitz.Rect(x0, y0, x1, y1)
                                    
                                    # Crop the image
                                    pix = pdf_page.get_pixmap(matrix=fitz.Matrix(1, 1), clip=rect)
                                    
                                    # Save the image
                                    image_filename = f"table_{table_idx}_page_{region.page_number}.png"
                                    image_path = os.path.join(images_dir, image_filename)
                                    pix.save(image_path)
                                    
                                    # Add image path to table data
                                    if "image_paths" not in table_data:
                                        table_data["image_paths"] = []
                                    table_data["image_paths"].append(os.path.join("images", image_filename))
                            except Exception as e:
                                print(f"Warning: Could not crop table {table_idx} from page {region.page_number}: {str(e)}")
            
            # Extract cells data
            if hasattr(table, 'cells') and table.cells:
                table_data["cells"] = []
                
                # Store cell word data for visualization
                cell_words_data = []
                
                for cell in table.cells:
                    cell_data = {
                        "row_index": cell.row_index,
                        "column_index": cell.column_index,
                        "row_span": cell.row_span,
                        "column_span": cell.column_span,
                        "content": cell.content,
                        "kind": cell.kind if hasattr(cell, 'kind') else None
                    }
                    
                    # Extract words in this cell using spans
                    if hasattr(cell, 'spans') and cell.spans:
                        cell_data["words"] = []
                        
                        for span in cell.spans:
                            offset = span.offset
                            length = span.length
                            
                            # Find all words that fall within this span
                            for word_offset, word in words_by_offset.items():
                                word_length = word["span"]["length"] if "span" in word and "length" in word["span"] else 0
                                
                                # Check if word is within the cell's span
                                if offset <= word_offset < offset + length:
                                    # Add word to cell's words list
                                    word_copy = word.copy()  # Create a copy to avoid modifying the original
                                    
                                    # Calculate word_bbox if polygon exists
                                    if "polygon" in word and table_page_number == word["page_number"]:
                                        # Find page in Azure data to get dimensions
                                        azure_page = None
                                        for p in result["pages"]:
                                            if p["page_number"] == word["page_number"]:
                                                azure_page = p
                                                break
                                        
                                        if azure_page and "bbox" in table_data:
                                            # Get polygon points
                                            polygon = word["polygon"]
                                            
                                            # Calculate scaling factors between Azure coordinates and PDF
                                            scale_x = page_width / azure_page["width"]
                                            scale_y = page_height / azure_page["height"]
                                            
                                            # Convert polygon to PDF coordinates and calculate bbox
                                            pdf_points = []
                                            for point in polygon:
                                                pdf_x = point["x"] * scale_x
                                                pdf_y = point["y"] * scale_y
                                                pdf_points.append((pdf_x, pdf_y))
                                            
                                            # Calculate bounding box in PDF coordinates
                                            x_values = [p[0] for p in pdf_points]
                                            y_values = [p[1] for p in pdf_points]
                                            x0, y0 = min(x_values), min(y_values)
                                            x1, y1 = max(x_values), max(y_values)
                                            
                                            # Calculate bounding box relative to table
                                            table_x0, table_y0, table_x1, table_y1 = table_data["bbox"]
                                            rel_x0 = x0 - table_x0
                                            rel_y0 = y0 - table_y0
                                            rel_x1 = x1 - table_x0
                                            rel_y1 = y1 - table_y0
                                            
                                            # Store the bbox in PDF coordinates relative to table
                                            word_copy["word_bbox"] = [rel_x0, rel_y0, rel_x1, rel_y1]
                                    
                                    cell_data["words"].append(word_copy)
                                    
                                    # Store word data for visualization
                                    if visualize and "polygon" in word:
                                        cell_words_data.append({
                                            "content": word["content"],
                                            "polygon": word["polygon"],
                                            "page_number": word["page_number"],
                                            "cell": {
                                                "row": cell.row_index,
                                                "col": cell.column_index
                                            }
                                        })
                    
                    # Extract bounding box for the cell if available
                    if hasattr(cell, 'bounding_regions') and cell.bounding_regions:
                        cell_data["bounding_regions"] = []
                        
                        for region in cell.bounding_regions:
                            region_data = {
                                "page_number": region.page_number if hasattr(region, 'page_number') else None,
                                "polygon": [{"x": point.x, "y": point.y} for point in region.polygon]
                            }
                            cell_data["bounding_regions"].append(region_data)
                            
                            # Add pixel coordinates for cell bounding regions if PDF is available
                            if pdf_doc and hasattr(region, 'page_number'):
                                page_number = region.page_number - 1  # PDF pages are 0-indexed
                                if 0 <= page_number < len(pdf_doc):
                                    try:
                                        # Get page dimensions
                                        pdf_page = pdf_doc[page_number]
                                        page_width = pdf_page.rect.width
                                        page_height = pdf_page.rect.height
                                        
                                        # Find page in Azure data to get dimensions
                                        azure_page = None
                                        for p in result["pages"]:
                                            if p["page_number"] == region.page_number:
                                                azure_page = p
                                                break
                                        
                                        if azure_page:
                                            # Calculate scaling factors between Azure coordinates and PDF
                                            scale_x = page_width / azure_page["width"]
                                            scale_y = page_height / azure_page["height"]
                                            
                                            # Convert polygon points to PDF coordinates
                                            points = []
                                            pixel_polygon = []
                                            for point in region.polygon:
                                                px = point.x * scale_x
                                                py = point.y * scale_y
                                                points.append((px, py))
                                                pixel_polygon.append({"x": px, "y": py})
                                            
                                            # Calculate bounding box
                                            x_values = [p[0] for p in points]
                                            y_values = [p[1] for p in points]
                                            x0, y0 = min(x_values), min(y_values)
                                            x1, y1 = max(x_values), max(y_values)
                                            
                                            if table_data["bbox"] is not None:
                                                x0 = x0 - table_data["bbox"][0]
                                                y0 = y0 - table_data["bbox"][1]
                                                x1 = x1 - table_data["bbox"][0]
                                                y1 = y1 - table_data["bbox"][1]

                                            x0 = max(x0, 0)
                                            y0 = max(y0, 0)
                                            x1 = min(x1, page_width)
                                            y1 = min(y1, page_height)

                                            # Add pixel coordinates to the region data
                                            region_data["polygon_pixels_in_table"] = pixel_polygon
                                            region_data["cell_bbox"] = [x0, y0, x1, y1]
                                    except Exception as e:
                                        print(f"Warning: Could not convert cell coordinates to pixels: {str(e)}")
                    
                    table_data["cells"].append(cell_data)
                
                # Create visualization of the table with word bounding boxes
                if visualize and pdf_doc and "image_paths" in table_data and cell_words_data:
                    try:
                        for img_path in table_data["image_paths"]:
                            # Load the table image
                            full_img_path = os.path.join(doc_dir, img_path)
                            if os.path.exists(full_img_path):
                                img = Image.open(full_img_path)
                                draw = ImageDraw.Draw(img)
                                
                            # Find page in Azure data to get dimensions
                            azure_page = None
                            for p in result["pages"]:
                                if p["page_number"] == table_page_number:
                                    azure_page = p
                                    break
                            
                            if azure_page:
                                # Calculate scaling factors
                                table_width = table_data["bbox"][2] - table_data["bbox"][0]
                                table_height = table_data["bbox"][3] - table_data["bbox"][1]
                                scale_x = img.width / table_width
                                scale_y = img.height / table_height
                                
                                # Draw word bounding boxes
                                for word_data in cell_words_data:
                                    if word_data["page_number"] == table_page_number:
                                        # Get word polygon
                                        polygon = word_data["polygon"]
                                        
                                        # Convert polygon to image coordinates
                                        img_polygon = []
                                        for point in polygon:
                                            # Convert from Azure coordinates to PDF coordinates
                                            pdf_x = point["x"] * (page_width / azure_page["width"])
                                            pdf_y = point["y"] * (page_height / azure_page["height"])
                                            
                                            # Convert from PDF coordinates to image coordinates
                                            img_x = (pdf_x - table_data["bbox"][0]) * scale_x
                                            img_y = (pdf_y - table_data["bbox"][1]) * scale_y
                                            
                                            img_polygon.append((img_x, img_y))
                                        
                                        # Calculate and store word bbox in image coordinates
                                        if len(img_polygon) >= 3:
                                            x_values = [p[0] for p in img_polygon]
                                            y_values = [p[1] for p in img_polygon]
                                            x0, y0 = min(x_values), min(y_values)
                                            x1, y1 = max(x_values), max(y_values)
                                            
                                            # Store the bbox in image coordinates
                                            word_data["word_bbox_in_image"] = [x0, y0, x1, y1]
                                            
                                            # Draw polygon
                                            if len(img_polygon) >= 3:  # Need at least 3 points for a polygon
                                                # Choose color based on cell position
                                                row = word_data["cell"]["row"]
                                                col = word_data["cell"]["col"]
                                                color = (
                                                    (row * 50) % 256,  # R
                                                    (col * 50) % 256,  # G
                                                    ((row + col) * 30) % 256  # B
                                                )
                                                
                                                draw.polygon(img_polygon, outline=color, width=1)
                                                
                                                # Calculate center of polygon for text
                                                # center_x = sum(p[0] for p in img_polygon) / len(img_polygon)
                                                # center_y = sum(p[1] for p in img_polygon) / len(img_polygon)
                                                
                                                # Draw word content
                                                # draw.text((center_x, center_y), word_data["content"], fill=color)
                                
                                # Save visualization
                                vis_filename = os.path.basename(img_path).replace('.png', '_vis.png')
                                vis_path = os.path.join(visualizations_dir, vis_filename)
                                img.save(vis_path)
                                
                                # Add visualization path to table data
                                if "visualization_paths" not in table_data:
                                    table_data["visualization_paths"] = []
                                table_data["visualization_paths"].append(os.path.join("visualizations", vis_filename))
                    except Exception as e:
                        print(f"Warning: Could not create visualization for table {table_idx}: {str(e)}")
            
            result["tables"].append(table_data)
    
    # Save result to JSON file
    with open(json_path, 'w', encoding='utf-8') as json_file:
        json.dump(result, json_file, indent=2, ensure_ascii=False)
    
    return result["tables"]


def process_single_file(input_path: str, output_dir: str, pdf_path: str = None, visualize: bool = False) -> bool:
    """Process a single pickle file and extract tables."""
    try:
        # Get filename without extension
        filename = os.path.basename(input_path).replace('.pkl', '')
        
        # Load the data
        data = load_pkl_file(input_path)
        
        # Extract tables and save as JSON
        tables = extract_tables_to_json(data, output_dir, filename, pdf_path, visualize)
        
        if tables:
            print(f"Found {len(tables)} tables in {filename}")
        else:
            print(f"No tables found in {filename}")
            
        return True
    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")
        return False


def process_directory(input_dir: str, output_dir: str, pdf_dir: str = None, visualize: bool = False) -> Dict[str, int]:
    """Process all pickle files in a directory and extract tables."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all pickle files
    pkl_files = [f for f in os.listdir(input_dir) if f.endswith('.pkl')]
    
    stats = {
        "total_files": len(pkl_files),
        "processed_files": 0,
        "files_with_tables": 0,
        "total_tables": 0,
        "failed_files": 0
    }
    
    for pkl_file in tqdm(pkl_files):
        input_path = os.path.join(input_dir, pkl_file)
        filename = pkl_file.replace('.pkl', '')
        
        # Determine PDF path if pdf_dir is provided
        pdf_path = None
        if pdf_dir:
            pdf_path = os.path.join(pdf_dir, f"{filename}.pdf")
            if not os.path.exists(pdf_path):
                print(f"Warning: PDF file not found: {pdf_path}")
                pdf_path = None
        
        try:
            # Load the data
            data = load_pkl_file(input_path)
            
            # Extract tables and save as JSON
            tables = extract_tables_to_json(data, output_dir, filename, pdf_path, visualize)
            
            stats["processed_files"] += 1
            
            if tables:
                stats["total_tables"] += len(tables)
                stats["files_with_tables"] += 1
                print(f"Found {len(tables)} tables in {filename}")
            else:
                print(f"No tables found in {filename}")
                
        except Exception as e:
            stats["failed_files"] += 1
            print(f"Error processing {pkl_file}: {str(e)}")
    
    return stats


def main():
    """Main function to parse arguments and process files."""
    parser = argparse.ArgumentParser(
        description="Extract tables from Azure Document Intelligence API results and crop table images"
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--pkl-file", 
        help="Path to a single pickle file containing Azure Document Intelligence results"
    )
    group.add_argument(
        "--input", 
        help="Directory containing pickle files with Azure Document Intelligence results"
    )
    
    parser.add_argument(
        "--output", 
        required=True,
        help="Output directory where document-specific folders will be created"
    )
    
    parser.add_argument(
        "--pdf-file",
        help="Path to the original PDF file (for single file mode)"
    )
    
    parser.add_argument(
        "--pdf-dir",
        help="Directory containing original PDF files (for batch mode)"
    )
    
    parser.add_argument(
        "--visualize", 
        action="store_true",
        help="Create visualizations of word bounding boxes on table images"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    if args.pkl_file:   
        # Process a single file
        if not os.path.isfile(args.pkl_file):
            print(f"Error: Input file not found: {args.pkl_file}")
            return 1
            
        print(f"Processing file: {args.pkl_file}")
        success = process_single_file(args.pkl_file, args.output, args.pdf_file, args.visualize)
        
        if success:
            filename = os.path.basename(args.pkl_file).replace('.pkl', '')
            print(f"Table extraction complete. Results saved to: {os.path.join(args.output, filename)}")
            return 0
        else:
            print("Table extraction failed.")
            return 1
    else:
        # Process a directory
        if not os.path.isdir(args.input):
            print(f"Error: Input directory not found: {args.input}")
            return 1
            
        print(f"Processing all pickle files in: {args.input}")
        stats = process_directory(args.input, args.output, args.pdf_dir, args.visualize)
        
        print("\nProcessing complete:")
        print(f"Total files: {stats['total_files']}")
        print(f"Successfully processed: {stats['processed_files']}")
        print(f"Files with tables: {stats['files_with_tables']}")
        print(f"Total tables extracted: {stats['total_tables']}")
        print(f"Failed files: {stats['failed_files']}")
        
        if stats['failed_files'] > 0:
            return 1
        return 0


if __name__ == "__main__":
    sys.exit(main())