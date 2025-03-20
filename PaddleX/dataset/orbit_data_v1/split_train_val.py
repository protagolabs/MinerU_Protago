import os
import random
import json
import re

def split_data(input_file, train_ratio=0.8, output_dir="split_data", random_seed=42):
    """
    Split the data from final_output.txt into train and validation sets,
    ensuring no overlap of filenames between the two sets.
    
    Args:
        input_file (str): Path to the final_output.txt file
        train_ratio (float): Ratio of data to use for training (default: 0.8)
        output_dir (str): Directory to save the split files
        random_seed (int): Random seed for reproducibility
    """
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Read all lines from the input file and group by filename
    filename_to_lines = {}
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line.strip())
                full_filename = data.get('filename', '')
                
                # Extract the base filename without path, table number, and page number
                if full_filename:
                    # Extract the part like "f_zaUn2eJQ" from "images/f_zaUn2eJQ_table_0_page_1.png"
                    match = re.search(r'images/([^_]+_[^_]+)', full_filename)
                    if match:
                        base_filename = match.group(1)
                        
                        if base_filename not in filename_to_lines:
                            filename_to_lines[base_filename] = []
                        
                        filename_to_lines[base_filename].append(line)
                    else:
                        print(f"Warning: Could not extract base filename from {full_filename}")
            except json.JSONDecodeError:
                print(f"Warning: Line {i+1} is not valid JSON and will be skipped")
    
    # Get unique base filenames and shuffle them
    unique_filenames = list(filename_to_lines.keys())
    random.shuffle(unique_filenames)
    
    # Calculate split point for filenames
    split_idx = int(len(unique_filenames) * train_ratio)
    
    # Split the filenames
    train_filenames = unique_filenames[:split_idx]
    val_filenames = unique_filenames[split_idx:]
    
    # Collect all lines for train and val sets
    train_lines = []
    for filename in train_filenames:
        train_lines.extend(filename_to_lines[filename])
    
    val_lines = []
    for filename in val_filenames:
        val_lines.extend(filename_to_lines[filename])
    
    # Write train data


    with open(os.path.join(output_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        f.writelines(train_lines)
    
    # Write validation data
    with open(os.path.join(output_dir, 'val.txt'), 'w', encoding='utf-8') as f:
        f.writelines(val_lines)
    
    # Create a summary file with statistics
    with open(os.path.join(output_dir, 'split_summary.txt'), 'w', encoding='utf-8') as f:
        f.write(f"Total unique documents: {len(unique_filenames)}\n")
        f.write(f"Total samples: {sum(len(lines) for lines in filename_to_lines.values())}\n")
        f.write(f"Training documents: {len(train_filenames)} ({len(train_filenames)/len(unique_filenames)*100:.1f}%)\n")
        f.write(f"Training samples: {len(train_lines)}\n")
        f.write(f"Validation documents: {len(val_filenames)} ({len(val_filenames)/len(unique_filenames)*100:.1f}%)\n")
        f.write(f"Validation samples: {len(val_lines)}\n")
        f.write(f"Random seed: {random_seed}\n")
    
    print(f"Split complete:")
    print(f"  - {len(train_filenames)} documents with {len(train_lines)} samples in training set")
    print(f"  - {len(val_filenames)} documents with {len(val_lines)} samples in validation set")
    print(f"Files saved in {output_dir}")

if __name__ == "__main__":
    # Path to your final_output.txt file

    input_file = "dataset/orbit_data_v1/final_output.txt"
    output_dir = "dataset/orbit_data_v1"

    # Split the data (80% train, 20% validation by default)
    split_data(input_file, train_ratio=0.8, output_dir=output_dir, random_seed=42)