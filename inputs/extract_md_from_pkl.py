import pickle
import os
import re
from pathlib import Path
import argparse

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Extract markdown content from pickle files and save them as markdown files.',
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
            
            # Create output markdown file path
            output_file = Path(args.output_dir) / f"{pkl_file.stem}.md"
            
            # Replace page numbers with double newlines
            content = data.content
            # Replace X/XX format page numbers with double newlines
            # content = re.sub(r'\d+/\d+', '\n\n', content)
            
            # Save content to markdown file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(content)
                
            print(f"Processed {pkl_file.name} -> {output_file.name}")
            
        except Exception as e:
            print(f"Error processing {pkl_file.name}: {str(e)}")

    print("Processing complete!")

if __name__ == '__main__':
    main()



    
# print("Type:", type(data))
# print("\nAvailable attributes:", dir(data))
# print("\nDocuments:", len(data.documents))
# if data.documents:
#     print("\nFirst document fields:", data.documents[0].fields.keys())
