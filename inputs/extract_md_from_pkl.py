import pickle
import os
import re
from pathlib import Path
import argparse

def is_table_line(line_content):
    """Check if a line is likely part of a table by looking for patterns."""
    # Check for pipe character which is common in markdown tables
    if "|" in line_content:
        return True
    
    # Check for multiple spaces or tabs between "cells"
    if re.search(r'\w+\s{2,}\w+\s{2,}\w+', line_content):
        return True
    
    return False

def extract_table_from_lines(lines, start_idx):
    """Extract table from consecutive lines that appear to be table rows."""
    table_lines = []
    i = start_idx
    
    while i < len(lines) and is_table_line(lines[i].content):
        table_lines.append(lines[i].content)
        i += 1
    
    return table_lines, i

def extract_page_content_as_markdown(page):
    """Convert a page object to markdown format."""
    md_content = []
    
    # Extract and format lines
    if hasattr(page, 'lines') and page.lines:
        i = 0
        while i < len(page.lines):
            line = page.lines[i]
            
            # Check if this line might be the start of a table
            if is_table_line(line.content):
                table_lines, new_i = extract_table_from_lines(page.lines, i)
                if len(table_lines) > 1:  # Only consider it a table if multiple rows
                    for table_line in table_lines:
                        md_content.append(table_line)
                    md_content.append("")
                    i = new_i
                    continue
            
            # Regular text line
            md_content.append(line.content)
            i += 1
        
        md_content.append("")
    
    # Add any selection marks if present
    if hasattr(page, 'selection_marks') and page.selection_marks:
        for mark in page.selection_marks:
            state = getattr(mark, 'state', 'unknown')
            md_content.append(f"- {state}")
        md_content.append("")
    
    # Add page end marker
    if hasattr(page, 'words') and page.words:
        md_content.append(f"*Page {page.page_number} ends*")
        md_content.append("")
    
    return "\n".join(md_content)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Extract page content from pickle files and save them as markdown files.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        required=True,
        help='Directory containing input pickle files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Directory where markdown files will be saved'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode to output more information'
    )

    # Parse arguments
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Process all pickle files in the input directory
    for pkl_file in Path(args.input_dir).glob('*.pkl'):
        try:
            # Load pickle file
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            
            # Check if data has pages attribute
            if not hasattr(data, 'pages'):
                print(f"Warning: {pkl_file.name} does not have 'pages' attribute, skipping...")
                continue
            
            # Create single output file for all pages
            output_file = Path(args.output_dir) / f"{pkl_file.stem}.md"
            
            # Process all pages and combine content
            all_pages_content = []
            
            for page in data.pages:
                # Extract raw content from the page
                page_content = extract_page_content_as_markdown(page)
                all_pages_content.append(page_content)
            
            # Save all content to a single markdown file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("\n".join(all_pages_content))
            
            print(f"Processed {pkl_file.name} -> {output_file.name} ({len(data.pages)} pages)")
            
        except Exception as e:
            print(f"Error processing {pkl_file.name}: {str(e)}")
            if args.debug:
                import traceback
                traceback.print_exc()

    print("Processing complete!")

if __name__ == '__main__':
    main()