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
from typing import Dict, Any, Optional, Set, List
from tqdm import tqdm
import numpy as np
import multiprocessing
from functools import partial
from utils.metric import TEDS
import concurrent.futures

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
    # Normalize HTML
    html_str = re.sub(r'\s+', ' ', html_str)  # Remove extra whitespace
    html_str = re.sub(r'\s+colspan="1"|\s+rowspan="1"', '', html_str)  # Remove default attributes
    html_str = re.sub(r'>\s+<|<\s+|\s+>', lambda m: '><' if '><' in m.group() else m.group().strip(), html_str)
    
    # Format HTML structure if needed
    if not html_str.strip().startswith('<html>'):
        html_str = f"""<html><head><meta charset="UTF-8"><style>table,th,td{{border:1px solid black;font-size:10px}}</style></head><body>{html_str}</body></html>"""
    
    return html_str

def calculate_std_similarity(text1, text2):
    """Calculate TED similarity and structural similarity using TEDS."""
    teds = TEDS(structure_only=False, n_jobs=1)
    teds_struct = TEDS(structure_only=True, n_jobs=1)
    return teds.evaluate(text1, text2), teds_struct.evaluate(text1, text2)

def compare_single_table(gt_table, extracted_tables_norm, extracted_by_page):
    """Compare a single ground truth table against all extracted tables."""
    best_match = None
    best_sim = best_struct_sim = 0
    best_idx = None
    
    # Get tables to compare
    tables_to_compare = extracted_by_page.get(gt_table['page'], []) or [(idx, t) for idx, t in enumerate(extracted_tables_norm)]
    
    # Find best match
    for j, (orig_idx, ext) in enumerate(tables_to_compare, 1):
        sim, struct_sim = calculate_std_similarity(gt_table['normalized'], ext['normalized'])
        
        if struct_sim > best_struct_sim:
            best_struct_sim = struct_sim
            best_sim = sim
            best_match = ext
            best_idx = orig_idx + 1
    
    if best_match:
        return {
            "azure_table_index": None,  # Will be set by the caller
            "mineru_table_index": best_idx,
            "mineru_table_page": best_match['original'].get('page', 'N/A'),
            "azure_table_page": gt_table['page'],
            "similarity_score": best_sim,
            "structure_similarity_score": best_struct_sim,
            "mineru_text": best_match['original']['sentence'],
            "azure_text": gt_table['original']['sentence'],
            "mineru_normalized_html": best_match['normalized'],
            "azure_normalized_html": gt_table['normalized']
        }
    return None

def compare_tables(gt_file, extracted_file, max_table_threads=32):
    """Compare tables between Azure and extracted files using concurrent.futures."""
    # Load files
    with open(gt_file, 'r', encoding='utf-8') as f:
        gt_tables = json.load(f)
    with open(extracted_file, 'r', encoding='utf-8') as f:
        extracted_tables = json.load(f)
    
    # Normalize tables
    gt_tables_norm = [{'original': t, 'normalized': normalize_and_format_table_html(t['sentence']), 'page': t['page']} 
                     for t in gt_tables]
    extracted_tables_norm = [{'original': t, 'normalized': normalize_and_format_table_html(t['sentence']), 'page': t['page']} 
                           for t in extracted_tables]
    
    # Group by page
    extracted_by_page = {}
    for idx, table in enumerate(extracted_tables_norm):
        extracted_by_page.setdefault(table['page'], []).append((idx, table))
    
    # Process tables in parallel with ThreadPoolExecutor
    total_sim = total_struct_sim = matched = 0
    detailed_matches = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(max_table_threads, len(gt_tables_norm))) as executor:
        # Create futures for each table comparison
        future_to_table = {
            executor.submit(compare_single_table, gt, extracted_tables_norm, extracted_by_page): i 
            for i, gt in enumerate(gt_tables_norm, 1)
        }
        
        # Process results as they complete
        with tqdm(total=len(gt_tables_norm), 
                 desc="Comparing tables", 
                 position=1, 
                 leave=False,  # Don't leave the table progress bar
                 file=sys.stdout) as pbar:
            for future in concurrent.futures.as_completed(future_to_table):
                i = future_to_table[future]
                try:
                    result = future.result()
                    if result:
                        result["azure_table_index"] = i
                        total_sim += result["similarity_score"]
                        total_struct_sim += result["structure_similarity_score"]
                        matched += 1
                        detailed_matches.append(result)
                except Exception as e:
                    print(f"Error comparing table {i}: {str(e)}")
                pbar.update(1)
    
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

def process_single_file(azure_file, mineru_folder, azure_folder, max_table_threads):
    """Process a single file comparison."""
    try:
        base_name = azure_file.replace('.pages.tables.json', '')
        mineru_file = base_name + '.tables.json'
        mineru_path = os.path.join(mineru_folder, mineru_file)
        azure_path = os.path.join(azure_folder, azure_file)
        
        if os.path.exists(mineru_path):
            try:
                results = compare_tables(azure_path, mineru_path, max_table_threads)
                return base_name, results
            except Exception as e:
                print(f"Error comparing tables in {azure_file}: {str(e)}")
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
                   output_dir: str,
                   num_processes: int = None,
                   max_table_threads: int = 32) -> Dict[str, Any]:
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
    print(f"Using {num_processes if num_processes else multiprocessing.cpu_count()} processes for file-level parallelization")
    print(f"Using {max_table_threads} threads for table-level parallelization")
    print()  # Add extra newline to separate progress bars from header
    
    # Set up multiprocessing
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_processes)
    
    # Create partial function with fixed arguments
    process_file = partial(process_single_file, 
                         mineru_folder=mineru_folder, 
                         azure_folder=azure_folder,
                         max_table_threads=max_table_threads)
    
    # Process files in parallel with progress bar
    all_results = {}
    total_avg_similarity = 0
    total_files = 0
    total_avg_structure_similarity = 0
    
    # Track files with tables specifically
    files_with_tables = 0
    total_similarity_with_tables = 0
    total_structure_similarity_with_tables = 0
    
    # Create file-level progress bar
    file_pbar = tqdm(total=len(files_to_process), 
                    desc="Processing files", 
                    position=0, 
                    leave=True,
                    file=sys.stdout)
    
    for result in pool.imap_unordered(process_file, files_to_process):
        if result:
            base_name, file_results = result
            all_results[base_name] = file_results
            total_avg_similarity += file_results['average_similarity']
            total_avg_structure_similarity += file_results['average_structure_similarity']
            total_files += 1
            
            # Count only files that have tables in both Azure and MinerU
            if file_results['azure_table_count'] > 0 and file_results['mineru_table_count'] > 0:
                files_with_tables += 1
                total_similarity_with_tables += file_results['average_similarity']
                total_structure_similarity_with_tables += file_results['average_structure_similarity']
            
        file_pbar.update(1)
    
    file_pbar.close()
    pool.close()
    pool.join()
    
    # Calculate overall statistics
    overall_stats = {
        "total_files_processed": total_files,
        "overall_average_similarity": total_avg_similarity / total_files if total_files > 0 else 0,
        "overall_average_structure_similarity": total_avg_structure_similarity / total_files if total_files > 0 else 0,
        "files_with_tables": files_with_tables,
        "overall_average_similarity_tables_only": total_similarity_with_tables / files_with_tables if files_with_tables > 0 else 0,
        "overall_average_structure_similarity_tables_only": total_structure_similarity_with_tables / files_with_tables if files_with_tables > 0 else 0,
        "timestamp": timestamp,
        "parallelization_config": {
            "num_processes": num_processes,
            "max_table_threads": max_table_threads
        }
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
    parser.add_argument(
        "--processes",
        type=int,
        default=None,
        help="Number of processes for file-level parallelization (default: number of CPU cores)"
    )
    parser.add_argument(
        "--table-threads",
        type=int,
        default=32,
        help="Maximum number of threads for table-level parallelization (default: 32)"
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
            args.output_dir,
            args.processes,
            args.table_threads
        )
        
        sys.exit(0)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()