#!/usr/bin/env python3
"""
Table Comparison Tool

This script compares table structures between MinerU and Azure JSON files,
calculating detailed similarity scores and generating comprehensive reports.

Author: Xing
Date: 2024-01-20
"""

import os
import json
import re
import sys
import argparse
from pathlib import Path
from datetime import datetime
from difflib import SequenceMatcher
from bs4 import BeautifulSoup
from typing import Dict, Any, Optional, Set, List, Tuple
from tqdm import tqdm
import numpy as np
from utils.metric import TEDS
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import multiprocessing
from itertools import product

# Compile regex patterns once at module level
WHITESPACE_PATTERN = re.compile(r'\s+')
DEFAULT_ATTR_PATTERN = re.compile(r'\s+colspan="1"|\s+rowspan="1"')
HTML_SPACE_PATTERN = re.compile(r'>\s+<|<\s+|\s+>')

class Logger:
    """Custom logger that writes to both console and file."""
    def __init__(self, filename: str):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')
        self.last_msg = ""

    def write(self, message: str) -> None:
        # Only log non-progress bar updates to file
        if '\r' not in message:  # Progress bars typically use carriage returns
            self.terminal.write(message)
            self.log.write(message)
            self.last_msg = message
        else:
            # For progress bars, only write to terminal
            self.terminal.write(message)
            # If it's the final progress bar update (ends with newline)
            if '\n' in message:
                self.log.write(f"Progress complete\n")

    def flush(self) -> None:
        self.terminal.flush()
        self.log.flush()

    def close(self) -> None:
        self.log.flush()
        self.log.close()


def normalize_and_format_table_html(html_str: str) -> str:
    """Normalize and format table HTML string for comparison."""
    # Use pre-compiled patterns
    html_str = WHITESPACE_PATTERN.sub(' ', html_str)
    html_str = DEFAULT_ATTR_PATTERN.sub('', html_str)
    html_str = HTML_SPACE_PATTERN.sub(lambda m: '><' if '><' in m.group() else m.group().strip(), html_str)
    
    # Format HTML structure if needed
    if not html_str.strip().startswith('<html>'):
        html_str = f"""<html><head><meta charset="UTF-8"><style>table,th,td{{border:1px solid black;font-size:10px}}</style></head><body>{html_str}</body></html>"""
    
    return html_str

# Create TEDS instances once
TEDS_INSTANCE = TEDS(structure_only=False, n_jobs=1)
TEDS_STRUCT_INSTANCE = TEDS(structure_only=True, n_jobs=1)

def calculate_std_similarity(text1: str, text2: str) -> Tuple[float, float]:
    """Calculate TED similarity and structural similarity using TEDS."""
    return TEDS_INSTANCE.evaluate(text1, text2), TEDS_STRUCT_INSTANCE.evaluate(text1, text2)

def process_table_batch(args: Tuple[List[Dict], List[Dict]]) -> List[Dict]:
    """Process a batch of table comparisons."""
    gt_batch, extracted_batch = args
    results = []
    
    # Create all possible pairs for comparison
    all_pairs = list(product(gt_batch, extracted_batch))
    
    # Process each pair
    for gt, ext in all_pairs:
        sim, struct_sim = calculate_std_similarity(ext['normalized'], gt['normalized'])
        
        if struct_sim > 0:  # Only include if there's a match
            results.append({
                "azure_table_index": gt['index'],
                "mineru_table_index": ext['index'] if 'index' in ext else None,
                "mineru_table_page": ext['original'].get('page', 'N/A'),
                "azure_table_page": gt['page'],
                "similarity_score": sim,
                "structure_similarity_score": struct_sim,
                "mineru_text": ext['original']['sentence'],
                "azure_text": gt['original']['sentence'],
                "mineru_normalized_html": ext['normalized'],
                "azure_normalized_html": gt['normalized']
            })
    
    # Group results by gt table and find best matches
    gt_results = {}
    for result in results:
        gt_idx = result['azure_table_index']
        if gt_idx not in gt_results or result['structure_similarity_score'] > gt_results[gt_idx]['structure_similarity_score']:
            gt_results[gt_idx] = result
    
    return list(gt_results.values())

def compare_tables(gt_file: str, extracted_file: str) -> Dict[str, Any]:
    """Compare tables between Azure and extracted files."""
    # Load files
    with open(gt_file, 'r', encoding='utf-8') as f:
        gt_tables = json.load(f)
    with open(extracted_file, 'r', encoding='utf-8') as f:
        extracted_tables = json.load(f)
    
    # Normalize tables
    gt_tables_norm = [{'index': i+1, 'original': t, 'normalized': normalize_and_format_table_html(t['sentence']), 'page': t['page']} 
                     for i, t in enumerate(gt_tables)]
    extracted_tables_norm = [{'index': i+1, 'original': t, 'normalized': normalize_and_format_table_html(t['sentence']), 'page': t['page']} 
                           for i, t in enumerate(extracted_tables)]
    
    # Group by page for more efficient comparison
    extracted_by_page = {}
    for idx, table in enumerate(extracted_tables_norm):
        extracted_by_page.setdefault(table['page'], []).append(table)
    
    # Prepare batches for parallel processing
    batches = []
    for gt in gt_tables_norm:
        tables_to_compare = extracted_by_page.get(gt['page'], []) or extracted_tables_norm
        batches.append(([gt], tables_to_compare))
    
    # Process batches in parallel
    num_workers = max(1, multiprocessing.cpu_count() - 1)
    total_sim = total_struct_sim = matched = 0
    detailed_matches = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_table_batch, batch) for batch in batches]
        for future in as_completed(futures):
            batch_results = future.result()
            for result in batch_results:
                total_sim += result['similarity_score']
                total_struct_sim += result['structure_similarity_score']
                matched += 1
                detailed_matches.append(result)
    
    return {
        "total_matched_tables": matched,
        "average_similarity": total_sim / matched if matched > 0 else 0,
        "average_structure_similarity": total_struct_sim / matched if matched > 0 else 0,
        "total_similarity": total_sim,
        "total_structure_similarity": total_struct_sim,
        "mineru_table_count": len(extracted_tables),
        "azure_table_count": len(gt_tables),
        "detailed_matches": detailed_matches,
        "file_stats": {
            "mineru_file": extracted_file,
            "azure_file": gt_file,
        }
    }

def process_single_file(azure_file, mineru_folder, azure_folder):
    """Process a single file comparison."""
    try:
        base_name = azure_file.replace('.pages.tables.json', '')
        mineru_file = base_name + '.tables.json'
        mineru_path = os.path.join(mineru_folder, mineru_file)
        azure_path = os.path.join(azure_folder, azure_file)
        
        if os.path.exists(mineru_path):
            try:
                results = compare_tables(azure_path, mineru_path)
                return base_name, results
            except Exception as e:
                print(f"Error comparing tables in {azure_file}: {str(e)}")
                # Return a placeholder result instead of None to avoid breaking the pipeline
                return base_name, {
                    "error": str(e),
                    "total_matched_tables": 0,
                    "average_similarity": 0,
                    "average_structure_similarity": 0,
                    "total_similarity": 0,
                    "total_structure_similarity": 0,
                    "mineru_table_count": 0,
                    "azure_table_count": 0,
                    "detailed_matches": [],
                    "file_stats": {
                        "mineru_file": mineru_path,
                        "azure_file": azure_path,
                    }
                }
        return None
    except Exception as e:
        print(f"Error processing {azure_file}: {str(e)}")
        return None
    
def process_folders(mineru_folder: str, 
                   azure_folder: str, 
                   output_dir: str) -> Dict[str, Any]:
    """Process and compare all corresponding files in both folders."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set up logging
    log_file = os.path.join(output_dir, f"comparison_log_{timestamp}.txt")
    sys.stdout = Logger(log_file)
    
    # Get all JSON files and create mapping
    azure_files = [f for f in os.listdir(azure_folder) if f.endswith('.json')]
    mineru_files = {f.replace('.tables.json', '.pages.tables.json'): f 
                   for f in os.listdir(mineru_folder) if f.endswith('.tables.json')}
    
    # Filter azure files to only those with corresponding mineru files
    files_to_process = [f for f in azure_files if f in mineru_files]
    
    if not files_to_process:
        print("No matching files found between MinerU and Azure folders!")
        return {}
    
    print(f"\nComparison started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"MinerU folder: {mineru_folder}")
    print(f"Azure folder: {azure_folder}")
    print(f"Found {len(files_to_process)} matching files to process")
    
    # Process files in parallel
    all_results = {}
    total_avg_similarity = 0
    total_files = 0
    total_avg_structure_similarity = 0
    files_with_tables = 0
    total_similarity_with_tables = 0
    total_structure_similarity_with_tables = 0
    
    num_workers = max(1, multiprocessing.cpu_count() - 1)
    process_func = partial(process_single_file, mineru_folder=mineru_folder, azure_folder=azure_folder)
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_file = {executor.submit(process_func, azure_file): azure_file 
                         for azure_file in files_to_process}
        
        for future in tqdm(as_completed(future_to_file), total=len(files_to_process), desc="Processing files"):
            result = future.result()
            if result:
                base_name, file_results = result
                all_results[base_name] = file_results
                total_avg_similarity += file_results['average_similarity']
                total_avg_structure_similarity += file_results['average_structure_similarity']
                total_files += 1
                
                if file_results['azure_table_count'] > 0 and file_results['mineru_table_count'] > 0:
                    files_with_tables += 1
                    total_similarity_with_tables += file_results['average_similarity']
                    total_structure_similarity_with_tables += file_results['average_structure_similarity']
    
    # Calculate overall statistics
    overall_stats = {
        "total_files_processed": total_files,
        "overall_average_similarity": total_avg_similarity / total_files if total_files > 0 else 0,
        "overall_average_structure_similarity": total_avg_structure_similarity / total_files if total_files > 0 else 0,
        "files_with_tables": files_with_tables,
        "overall_average_similarity_tables_only": total_similarity_with_tables / files_with_tables if files_with_tables > 0 else 0,
        "overall_average_structure_similarity_tables_only": total_structure_similarity_with_tables / files_with_tables if files_with_tables > 0 else 0,
        "timestamp": timestamp
    }
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Total files processed: {total_files}")
    print(f"Files with tables in both Azure and MinerU: {files_with_tables}")
    print(f"Overall average similarity (all files): {overall_stats['overall_average_similarity']:.4f}")
    print(f"Overall average structure similarity (all files): {overall_stats['overall_average_structure_similarity']:.4f}")
    print(f"Overall average similarity (files with tables only): {overall_stats['overall_average_similarity_tables_only']:.4f}")
    print(f"Overall average structure similarity (files with tables only): {overall_stats['overall_average_structure_similarity_tables_only']:.4f}")
    
    # Save detailed results
    save_results(all_results, overall_stats, output_dir, timestamp)
    
    # Restore original stdout and close log file
    if isinstance(sys.stdout, Logger):
        sys.stdout.close()
        sys.stdout = sys.stdout.terminal
    
    return all_results

def save_results(results: Dict[str, Any], 
                overall_stats: Dict[str, Any], 
                output_dir: str, 
                timestamp: str) -> None:
    """Save comparison results to JSON file.
    
    Args:
        results: Dictionary containing comparison results
        overall_stats: Dictionary containing overall statistics
        output_dir: Output directory path
        timestamp: Timestamp string for filename
    """
    output_file = os.path.join(output_dir, f"comparison_results_{timestamp}.json")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "summary": overall_stats,
            "detailed_results": results
        }, f, indent=4, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")

def parse_args() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Compare table structures between MinerU and Azure JSON files."
    )
    parser.add_argument(
        "--mineru-tables",
        type=str,
        default="inputs/export_pdf/azure_tables",
        help="Path to folder containing MinerU table JSON files"
    )
    parser.add_argument(
        "--azure-tables",
        type=str,
        default="azure_outputs_tables",
        help="Path to folder containing Azure table JSON files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="comparison_results",
        help="Path to output directory for results"
    )
    return parser.parse_args()

def main() -> None:
    try:
        args = parse_args()
        
        # Validate input folders
        for folder in [args.mineru_tables, args.azure_tables]:
            if not os.path.isdir(folder):
                raise ValueError(f"Directory not found: {folder}")
        
        # Run comparison
        results = process_folders(
            args.mineru_tables,
            args.azure_tables,
            args.output_dir
        )
        
        sys.exit(0)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()