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
    
    # Remove any existing HTML wrapper if present
    html_str = html_str.strip()
    if html_str.startswith('<html>'):
        html_str = re.sub(r'<html>.*?<body>(.*?)</body>.*?</html>', r'\1', html_str, flags=re.DOTALL)
    
    # Handle multiple table tags
    if '<table>' in html_str:
        # Count opening and closing table tags
        open_count = html_str.count('<table>')
        close_count = html_str.count('</table>')
        
        if open_count > close_count:
            # Remove extra opening table tags
            html_str = re.sub(r'<table>(?=<table>)', '', html_str)
            # If no closing tag, add one at the end
            if close_count == 0:
                html_str = html_str + '</table>'
        elif open_count > 1 and close_count > 1:
            # Extract all table content
            tables = re.findall(r'<table>(.*?)</table>', html_str, re.DOTALL)
            # Combine all tables into one
            combined_table = '<table>' + ''.join(tables) + '</table>'
            html_str = combined_table
    
    # Ensure proper table structure
    if not html_str.strip().startswith('<table>'):
        html_str = f'<table>{html_str}</table>'
    
    # Add HTML wrapper with proper styling
    html_str = f"""<html><body>{html_str}</body></html>"""
    
    return html_str

def calculate_std_similarity(text1, text2):
    """Calculate TED similarity and structural similarity using TEDS.
    
    Args:
        text1 (str): First HTML table text
        text2 (str): Second HTML table text
        
    Returns:
        tuple: (similarity_score, structure_similarity_score)
    """
    # Initialize TEDS with different configurations
    teds = TEDS(structure_only=False, n_jobs=1)
    teds_struct = TEDS(structure_only=True, n_jobs=1)
    
    # Normalize HTML strings - this will handle the HTML wrapping if needed
    text1 = normalize_and_format_table_html(text1)
    text2 = normalize_and_format_table_html(text2)
    
    # Calculate both regular and structure-only similarity scores

    # print(text1)
    # print(text2)
    similarity = teds.evaluate(text1, text2)
    structure_similarity = teds_struct.evaluate(text1, text2)
    
    return similarity, structure_similarity

def compare_tables(gt_file, extracted_file):
    """Compare tables between Azure and extracted files."""
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
    
    # Compare tables
    total_sim = total_struct_sim = matched = 0
    detailed_matches = []
    
    for i, gt in enumerate(gt_tables_norm, 1):
        best_match = None
        best_sim = best_struct_sim = 0
        best_idx = None
        
        # Get tables to compare
        tables_to_compare = extracted_by_page.get(gt['page'], []) or [(idx, t) for idx, t in enumerate(extracted_tables_norm)]
        
        # Find best match
        for j, (orig_idx, ext) in enumerate(tables_to_compare, 1):
            # sim, struct_sim = calculate_std_similarity(gt['normalized'], ext['normalized'])
            sim, struct_sim = calculate_std_similarity(gt['original']['sentence'], ext['original']['sentence'])

            # if struct_sim > best_struct_sim:
            if sim > best_sim:
                best_struct_sim = struct_sim
                best_sim = sim
                best_match = ext
                best_idx = orig_idx + 1
        
        if best_match:
            total_sim += best_sim
            total_struct_sim += best_struct_sim
            matched += 1
            
            detailed_matches.append({
                "azure_table_index": i,
                "mineru_table_index": best_idx,
                "mineru_table_page": best_match['original'].get('page', 'N/A'),
                "azure_table_page": gt['page'],
                "similarity_score": best_sim,
                "structure_similarity_score": best_struct_sim,
                # Full original text
                "mineru_text": best_match['original']['sentence'],
                "azure_text": gt['original']['sentence'],
                # Full normalized HTML
                "mineru_normalized_html": best_match['normalized'],
                "azure_normalized_html": gt['normalized']
            })
    
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
                   output_dir: str,
                   num_processes: int = None) -> Dict[str, Any]:
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
    
    # Set up multiprocessing
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_processes)
    
    # Create partial function with fixed arguments
    process_file = partial(process_single_file, 
                         mineru_folder=mineru_folder, 
                         azure_folder=azure_folder)
    
    # Process files in parallel with progress bar
    all_results = {}
    total_avg_similarity = 0
    total_files = 0
    total_avg_structure_similarity = 0
    
    # Track files with tables specifically
    files_with_tables = 0
    total_similarity_with_tables = 0
    total_structure_similarity_with_tables = 0
    
    with tqdm(total=len(files_to_process), desc="Processing files", file=sys.stdout) as pbar:
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
                
            pbar.update(1)
    
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
    parser.add_argument(
        "--processes",
        type=int,
        default=None,
        help="Number of parallel processes to use (default: number of CPU cores)"
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
            args.processes
        )
        
        sys.exit(0)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()