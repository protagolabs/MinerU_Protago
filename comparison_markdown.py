import os
import sys
import json
import re
from datetime import datetime
import concurrent.futures
from typing import Dict, Any, List, Tuple
import argparse

def find_matching_files(dir1, dir2):
    """Find files that exist in both directories."""
    files1 = set(os.listdir(dir1))
    files2 = set(os.listdir(dir2))
    return files1.intersection(files2)

def split_by_page_numbers(content: str) -> List[str]:
    """Split content by page endings marked with "*Page X ends*" format."""
    # Pattern to match page endings in format "*Page X ends*"
    page_pattern = r'\n\*Page \d+ ends\*\n'
    
    # Split content by page endings
    blocks = re.split(page_pattern, content)
    
    # Filter out empty blocks and strip whitespace
    blocks = [block.strip() for block in blocks if block.strip()]
    
    return blocks
    

def compare_single_file(gt_file: str, method_file: str) -> Dict[str, Any]:
    """Compare a single pair of markdown files."""
    try:
        # Read the files with explicit encoding
        with open(gt_file, 'r', encoding='utf-8') as f:
            gt_content = f.read()
        with open(method_file, 'r', encoding='utf-8') as f:
            method_content = f.read()
        
        # Compare using the existing function
        from scorers.heuristic import HeuristicScorer
        scorer = HeuristicScorer()
        
        # Split content by page numbers
        gt_blocks = split_by_page_numbers(gt_content)
        # print(len(gt_blocks))
        
        # method_blocks = split_by_page_numbers(method_content)
        # print(len(method_blocks))

        # Skip if only one block is found
        if len(gt_blocks) == 1:
            print(f"Skipping {os.path.basename(gt_file)} - only one block found")
            return {'skipped': True, 'reason': 'Only one block found'}
        
        result = scorer(None, gt_blocks, method_content)
        general_result = result['score']
        order_result = result['specific_scores']['order']
        print(f"\n Overall Score: {general_result:.2f}")
        print(f"\n Order Score: {order_result:.2f}")        

        # general_result = 0
        # order_result = 0
        # if len(method_blocks) == 1:
        #     result = scorer(None, gt_blocks, method_content)
        #     general_result = result['score']
        #     order_result = result['specific_scores']['order']
        # else:
        #     for block in method_blocks:
        #         result = scorer(None, gt_blocks, block)
        #         general_result += result['score']
        #         order_result += result['specific_scores']['order']
        #     general_result /= len(method_blocks)
        #     order_result /= len(method_blocks)

        # Print results
        # print(f"Overall Score: {general_result:.2f}")
        # print(f"Order Score: {order_result:.2f}")

        return result
        
    except Exception as e:
        print(f"Error processing files: {str(e)}")
        return {'error': str(e)}

def process_file_pair(args: Tuple[str, str, str]) -> Tuple[str, Dict[str, Any]]:
    """Process a single file pair for parallel execution."""
    filename, gt_file, method_file = args
    print(f"\nProcessing for {os.path.basename(filename)}:")
    result = compare_single_file(gt_file, method_file)
    return filename, result

def batch_compare_directories(gt_dir: str, method_dir: str, output_file: str = None, max_workers: int = None):
    """Compare all matching markdown files between two directories using parallel processing."""
    matching_files = find_matching_files(gt_dir, method_dir)
    results = {}
    
    print(f"Found {len(matching_files)} matching files to compare")
    
    # Prepare arguments for parallel processing
    file_pairs = [
        (filename, 
         os.path.join(gt_dir, filename), 
         os.path.join(method_dir, filename))
        for filename in matching_files
    ]
    
    # Initialize progress counter
    processed_files = 0
    total_files = len(matching_files)
    
    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks and get futures
        future_to_file = {
            executor.submit(process_file_pair, file_pair): file_pair[0]
            for file_pair in file_pairs
        }
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_file):
            filename = future_to_file[future]
            try:
                _, result = future.result()
                results[filename] = result
                # Update and display progress
                processed_files += 1
                print(f"\rProgress: {processed_files}/{total_files} files processed ({(processed_files/total_files)*100:.1f}%)", end="")
            except Exception as e:
                print(f"\nError processing {filename}: {str(e)}")
                results[filename] = {'error': str(e)}
                processed_files += 1
                print(f"\rProgress: {processed_files}/{total_files} files processed ({(processed_files/total_files)*100:.1f}%)", end="")
    
    print("\n")  # Add newline after progress counter
    
    # Calculate and print summary statistics
    valid_scores = [r['score'] for r in results.values() if 'score' in r]
    valid_order_scores = [r['specific_scores']['order'] for r in results.values() if 'specific_scores' in r and 'order' in r['specific_scores']]
    if valid_scores:
        avg_score = sum(valid_scores) / len(valid_scores)
        print(f"\nSummary:")
        print(f"Total files compared: {len(matching_files)}")
        print(f"Average overall score: {avg_score:.2f}")
        
    if valid_order_scores:
        avg_order_score = sum(valid_order_scores) / len(valid_order_scores)
        print(f"Average order score: {avg_order_score:.2f}")
    # Save results if output file specified
    if output_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"{output_file}_{timestamp}.json"
        
        # Add summary statistics to the results
        output_data = {
            'results': results,
            'summary': {
                'total_files': len(matching_files),
                'average_score': avg_score if valid_scores else 0,
                'average_order_score': avg_order_score if valid_order_scores else 0,
                'timestamp': timestamp
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare markdown files or directories.")
    parser.add_argument('--gt_path', type=str, help="Path to the ground truth file or directory.")
    parser.add_argument('--method_path', type=str, help="Path to the method file or directory.")
    parser.add_argument('--output_file', type=str, help="Optional output file for results.")
    parser.add_argument('--max_workers', type=int, help="Optional number of workers for parallel processing.")
    
    args = parser.parse_args()
    
    # Check if paths are files or directories
    if not os.path.exists(args.gt_path):
        print(f"Error: Ground truth path '{args.gt_path}' does not exist.")
        sys.exit(1)
    if not os.path.exists(args.method_path):
        print(f"Error: Method path '{args.method_path}' does not exist.")
        sys.exit(1)

    if os.path.isfile(args.gt_path) and os.path.isfile(args.method_path):
        compare_single_file(args.gt_path, args.method_path)
    elif os.path.isdir(args.gt_path) and os.path.isdir(args.method_path):
        batch_compare_directories(args.gt_path, args.method_path, args.output_file, args.max_workers)
    else:
        print("Error: Both paths must be either files or directories")
        sys.exit(1)