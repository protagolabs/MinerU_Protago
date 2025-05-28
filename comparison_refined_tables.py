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
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Set, List
from tqdm import tqdm
import multiprocessing
from functools import partial
from utils.metric import TEDS
from lxml import html, etree

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

def preprocess_html_for_ted(html_string):
    """Normalize HTML using lxml, preserving structural compatibility with TEDS."""
    parser = html.HTMLParser(remove_comments=True, encoding='utf-8')
    root = html.fromstring(html_string, parser=parser)

    # 1. Remove unwanted tags
    etree.strip_elements(root, 'script', 'style', 'meta', 'link', 'head', 'noscript', with_tail=False)

    # 2. Normalize <th> to <td>
    for th in root.xpath('//th'):
        th.tag = 'td'

    # 3. Flatten <br> by replacing it with spaces
    for br in root.xpath('//br'):
        parent = br.getparent()
        if parent is not None:
            index = parent.index(br)
            # Insert a space text node before removing <br>
            if index > 0 and parent[index - 1].tail:
                parent[index - 1].tail += ' '
            else:
                br.tail = ' ' + (br.tail or '')
            parent.remove(br)

    # 4. Remove <tbody> (auto-inserted in some renderers)
    for tbody in root.xpath('//tbody'):
        tbody.drop_tag()

    # 5. Strip all attributes from all tags
    for elem in root.iter():
        elem.attrib.clear()

    # 6. Normalize text and tail spacing
    def normalize_text(s):
        if s:
            return ' '.join(s.split())
        return ''

    for elem in root.iter():
        elem.text = normalize_text(elem.text)
        elem.tail = normalize_text(elem.tail)

    # 7. Return cleaned HTML string
    return html.tostring(root, encoding='unicode', method='html')


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
    morlized_text1 = preprocess_html_for_ted(text1)
    morlized_text2 = preprocess_html_for_ted(text2)
    
    # Calculate both regular and structure-only similarity scores

    # print(text1)
    # print(text2)
    similarity = teds.evaluate(morlized_text1, morlized_text2)
    structure_similarity = teds_struct.evaluate(morlized_text1, morlized_text2)
    
    return similarity, structure_similarity, morlized_text1, morlized_text2

def compare_tables(gt_file, extracted_file):
    """Compare tables between Azure and extracted files."""
    # Load files
    with open(gt_file, 'r', encoding='utf-8') as f:
        gt_tables = json.load(f)
    with open(extracted_file, 'r', encoding='utf-8') as f:
        extracted_tables = json.load(f)
    
    # Normalize tables
    gt_tables = [{'original': t, 'page': t['page']} for t in gt_tables]
    extracted_tables = [{'original': t, 'page': t['page']} for t in extracted_tables]
    
    
    # Compare tables
    total_sim = total_struct_sim = matched = 0
    detailed_matches = []
    matched_mineru_indices = set()  # Track which MinerU tables have been matched
    
    for i, gt in enumerate(gt_tables, 1):
        best_match = None
        best_sim = best_struct_sim = 0
        best_idx = None
        best_gt_text = None
        best_ext_text = None
        
        # Get tables to compare
        tables_to_compare = [(idx, t) for idx, t in enumerate(extracted_tables) if t['page'] == gt['page']]
        
        # Find best match
        for j, (orig_idx, ext) in enumerate(tables_to_compare, 1):

            if not ext['original']['sentence'].startswith("<html>"):
                sim, struct_sim, normalized_gt_text, normalized_ext_text = calculate_std_similarity(gt['original']['sentence'], ext['original']['original_sentence'])
            else:
                sim, struct_sim, normalized_gt_text, normalized_ext_text = calculate_std_similarity(gt['original']['sentence'], ext['original']['sentence'])

            # if struct_sim > best_struct_sim:
            if sim > best_sim:
                best_struct_sim = struct_sim
                best_sim = sim
                best_match = ext
                best_idx = orig_idx + 1
                best_gt_text = normalized_gt_text
                best_ext_text = normalized_ext_text
        
        if best_match:
            total_sim += best_sim
            total_struct_sim += best_struct_sim
            matched += 1
            
            # Track the actual index in the original extracted_tables list
            actual_mineru_idx = None
            for idx, table in enumerate(extracted_tables):
                if table is best_match:
                    actual_mineru_idx = idx
                    break
            
            if actual_mineru_idx is not None:
                matched_mineru_indices.add(actual_mineru_idx)
            
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
                "mineru_text_normalized": best_ext_text,
                "azure_text_normalized": best_gt_text,
            })
    
    return {
        "total_matched_tables": matched,  # Number of Azure tables that found a match
        "unique_mineru_tables_matched": len(matched_mineru_indices),  # Number of unique MinerU tables matched
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
                    "unique_mineru_tables_matched": 0,
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
    
    # Track table counts across all files
    total_azure_tables = 0
    total_mineru_tables = 0
    total_compared_tables = 0
    total_unique_mineru_matched = 0
    
    with tqdm(total=len(files_to_process), desc="Processing files", file=sys.stdout) as pbar:
        for result in pool.imap_unordered(process_file, files_to_process):
            if result:
                base_name, file_results = result
                all_results[base_name] = file_results
                total_avg_similarity += file_results['average_similarity']
                total_avg_structure_similarity += file_results['average_structure_similarity']
                total_files += 1
                
                # Add table counts
                total_azure_tables += file_results['azure_table_count']
                total_mineru_tables += file_results['mineru_table_count']
                total_compared_tables += file_results['total_matched_tables']
                total_unique_mineru_matched += file_results['unique_mineru_tables_matched']
                
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
        "total_azure_tables": total_azure_tables,
        "total_mineru_tables": total_mineru_tables,
        "total_compared_tables": total_compared_tables,
        "total_unique_mineru_matched": total_unique_mineru_matched,
        "comparison_coverage_azure": (total_compared_tables / total_azure_tables * 100) if total_azure_tables > 0 else 0,
        "comparison_coverage_mineru": (total_unique_mineru_matched / total_mineru_tables * 100) if total_mineru_tables > 0 else 0,
        "timestamp": timestamp
    }
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Total files processed: {total_files}")
    print(f"Files with tables in both Azure and MinerU: {files_with_tables}")
    print(f"\nTable Statistics:")
    print(f"Total Azure tables: {total_azure_tables}")
    print(f"Total MinerU tables: {total_mineru_tables}")
    print(f"Azure tables that found matches: {total_compared_tables}")
    print(f"Unique MinerU tables matched: {total_unique_mineru_matched}")
    print(f"Comparison coverage (Azure): {overall_stats['comparison_coverage_azure']:.2f}%")
    print(f"Comparison coverage (MinerU): {overall_stats['comparison_coverage_mineru']:.2f}%")
    print(f"\nSimilarity Scores:")
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