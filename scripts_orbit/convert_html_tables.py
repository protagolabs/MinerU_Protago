import os
import re
import glob
import argparse
from htmltabletomd import convert_table

def process_markdown_file(input_file, output_file):
    """Process a markdown file and convert HTML tables to markdown."""
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all HTML tables in the content
    html_tables = re.findall(r'<html>.*?</html>', content, re.DOTALL)
    
    # Convert each HTML table to markdown
    for html_table in html_tables:
        try:
            markdown_table = convert_table(html_table)
            content = content.replace(html_table, markdown_table)
        except Exception as e:
            print(f"Error converting table in {input_file}: {str(e)}")
            continue
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Write the modified content to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Convert HTML tables in markdown files to markdown format')
    parser.add_argument('--input-dir', required=True, help='Input directory containing markdown files')
    parser.add_argument('--output-dir', required=True, help='Output directory for converted files')
    args = parser.parse_args()
    
    # Get all .md files in the input directory
    md_files = glob.glob(os.path.join(args.input_dir, '*.md'))
    
    # Process each file
    for input_file in md_files:
        try:
            # Create output file path
            rel_path = os.path.relpath(input_file, args.input_dir)
            output_file = os.path.join(args.output_dir, rel_path)
            
            print(f"Processing {input_file}...")
            process_markdown_file(input_file, output_file)
            print(f"Successfully processed {input_file}")
        except Exception as e:
            print(f"Error processing {input_file}: {str(e)}")

if __name__ == '__main__':
    main() 