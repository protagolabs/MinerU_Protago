import os
import sys
import json
import re
from datetime import datetime
import concurrent.futures
from typing import Dict, Any, List, Tuple

def find_matching_files(dir1, dir2):
    """Find files that exist in both directories."""
    files1 = set(os.listdir(dir1))
    files2 = set(os.listdir(dir2))
    return files1.intersection(files2)

def split_by_page_numbers(content: str) -> List[str]:
    """Split content by page numbers in formats like '1', '2' or '1/40', '2/40'."""
    # Pattern to match page numbers in formats like "1", "2" or "1/40", "2/40"
    page_pattern = r'(?:^|\n)(?:\d+(?:/\d+)?)\s*\n'
    
    # Split content by page numbers
    blocks = re.split(page_pattern, content)
    
    # Filter out empty blocks and strip whitespace
    blocks = [block.strip() for block in blocks if block.strip()]
    
    return blocks

def compare_single_file(gt_file: str, method_file: str) -> Dict[str, Any]:
    """Compare a single pair of markdown files."""
    try:
        # Read the files
        with open(gt_file, 'r') as f:
            gt_content = f.read()
        with open(method_file, 'r') as f:
            method_content = f.read()
        
        # Compare using the existing function
        from scorers.heuristic import HeuristicScorer
        scorer = HeuristicScorer()
        
        # Split content by page numbers
        gt_blocks = split_by_page_numbers(gt_content)
        # print(len(gt_blocks))
        
        # Skip if only one block is found
        if len(gt_blocks) == 1:
            print(f"Skipping {os.path.basename(gt_file)} - only one block found")
            return {'skipped': True, 'reason': 'Only one block found'}
        
        result = scorer(None, gt_blocks, method_content)
        
        # Print results
        print(f"\nResults for {os.path.basename(gt_file)}:")
        print(f"Overall Score: {result['score']:.2f}")
        print(f"Order Score: {result['specific_scores']['order']:.2f}")

        
        return result
        
    except Exception as e:
        print(f"Error processing files: {str(e)}")
        return {'error': str(e)}

def process_file_pair(args: Tuple[str, str, str]) -> Tuple[str, Dict[str, Any]]:
    """Process a single file pair for parallel execution."""
    filename, gt_file, method_file = args
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
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                results[filename] = {'error': str(e)}
    
    # Calculate and print summary statistics
    valid_scores = [r['score'] for r in results.values() if 'score' in r]
    if valid_scores:
        avg_score = sum(valid_scores) / len(valid_scores)
        print(f"\nSummary:")
        print(f"Total files compared: {len(matching_files)}")
        print(f"Average overall score: {avg_score:.2f}")
    
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
                'timestamp': timestamp
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 3 or len(sys.argv) > 5:
        print("Usage:")
        print("For single file comparison:")
        print("  python batch_compare_markdown.py <ground_truth_file> <method_file>")
        print("For directory comparison:")
        print("  python batch_compare_markdown.py <ground_truth_dir> <method_dir> [output_file] [max_workers]")
        sys.exit(1)
    
    gt_path = sys.argv[1]
    method_path = sys.argv[2]
    output_file = sys.argv[3] if len(sys.argv) > 3 else None
    max_workers = int(sys.argv[4]) if len(sys.argv) > 4 else None
    
    # Check if paths are files or directories
    if os.path.isfile(gt_path) and os.path.isfile(method_path):
        compare_single_file(gt_path, method_path)
    elif os.path.isdir(gt_path) and os.path.isdir(method_path):
        batch_compare_directories(gt_path, method_path, output_file, max_workers)
    else:
        print("Error: Both paths must be either files or directories")
        sys.exit(1) 