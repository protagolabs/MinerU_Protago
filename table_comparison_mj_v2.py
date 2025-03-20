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

class TreeNode:
    """Represents a node in the HTML tree."""
    def __init__(self, tag, children=None, text=""):
        self.tag = tag
        self.children = children if children else []
        self.text = text.strip()

    def __repr__(self):
        return f"TreeNode({self.tag}, children={len(self.children)}, text='{self.text}')"

def parse_html_to_tree(soup, ignore_text=False):
    """Recursively converts a BeautifulSoup object into a tree structure.
    
    Args:
        soup: BeautifulSoup object
        ignore_text: If True, ignores text content (for structural comparison)
    """
    if not soup.name:  # Text node
        if ignore_text:
            return None
        return TreeNode(tag="text", text=soup.strip()) if soup.strip() else None

    children = [parse_html_to_tree(child, ignore_text) for child in soup.children if parse_html_to_tree(child, ignore_text)]
    return TreeNode(tag=soup.name, 
                   children=children, 
                   text="" if ignore_text else soup.get_text().strip())

def tree_edit_distance(tree1, tree2):
    """Computes the Tree Edit Distance (TED) using dynamic programming."""
    if tree1 is None:
        return sum(count_nodes(child) for child in tree2.children) + 1 if tree2 else 0
    if tree2 is None:
        return sum(count_nodes(child) for child in tree1.children) + 1 if tree1 else 0

    if tree1.tag == tree2.tag:  # Same node type
        cost = 0 if tree1.text == tree2.text else 1  # Text difference penalty
    else:
        cost = 1  # Different node type penalty

    dp = np.zeros((len(tree1.children) + 1, len(tree2.children) + 1))

    for i in range(len(tree1.children) + 1):
        for j in range(len(tree2.children) + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            else:
                dp[i][j] = min(
                    dp[i - 1][j] + count_nodes(tree1.children[i - 1]),  # Deletion
                    dp[i][j - 1] + count_nodes(tree2.children[j - 1]),  # Insertion
                    dp[i - 1][j - 1] + tree_edit_distance(tree1.children[i - 1], tree2.children[j - 1])  # Substitution
                )

    return dp[len(tree1.children)][len(tree2.children)] + cost

def count_nodes(tree):
    """Counts the total nodes in a tree (used for normalization)."""
    if tree is None:
        return 0
    return 1 + sum(count_nodes(child) for child in tree.children)

def ted_similarity(html1, html2):
    """Computes the TED similarity between two HTML tables."""
    soup1, soup2 = BeautifulSoup(html1, "html.parser"), BeautifulSoup(html2, "html.parser")
    
    tree1, tree2 = parse_html_to_tree(soup1, ignore_text=False), parse_html_to_tree(soup2, ignore_text=False)
    
    ted_distance = tree_edit_distance(tree1, tree2)
    
    max_nodes = max(count_nodes(tree1), count_nodes(tree2))
    
    return 1 - (ted_distance / max_nodes) if max_nodes > 0 else 1  # Normalized similarity score

def structural_ted_similarity(html1, html2):
    """Computes the TED similarity focusing only on structure (ignoring text content)."""
    soup1, soup2 = BeautifulSoup(html1, "html.parser"), BeautifulSoup(html2, "html.parser")
    
    # Pass ignore_text=True to ignore text content
    tree1, tree2 = parse_html_to_tree(soup1, ignore_text=True), parse_html_to_tree(soup2, ignore_text=True)
    
    ted_distance = tree_edit_distance(tree1, tree2)
    
    max_nodes = max(count_nodes(tree1), count_nodes(tree2))
    
    return 1 - (ted_distance / max_nodes) if max_nodes > 0 else 1  # Normalized score

def clean_table_text(text):
    """Clean table text by removing HTML tags, extra spaces and standardizing whitespace"""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Remove extra spaces and standardize whitespace
    text = ' '.join(text.split())
    return text

def normalize_table_html(html_str: str) -> str:
    """Normalize table HTML string by removing extra whitespace and standardizing attributes.
    
    Args:
        html_str: Input HTML table string
        
    Returns:
        Normalized HTML table string
    """
    # Remove newlines and extra whitespace
    html_str = re.sub(r'\s+', ' ', html_str)
    
    # Normalize colspan="1" rowspan="1" by removing them (they're default values)
    html_str = re.sub(r'\s+colspan="1"', '', html_str)
    html_str = re.sub(r'\s+rowspan="1"', '', html_str)
    
    # Remove extra spaces between tags
    html_str = re.sub(r'>\s+<', '><', html_str)
    
    # Remove space before closing tags
    html_str = re.sub(r'\s+>', '>', html_str)
    
    # Remove space after opening tags
    html_str = re.sub(r'<\s+', '<', html_str)
    
    return html_str

def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate similarity ratio between two texts using SequenceMatcher.
    
    Args:
        text1: First text for comparison
        text2: Second text for comparison
        
    Returns:
        Similarity ratio between 0 and 1
    """
    return SequenceMatcher(None, text1, text2).ratio()

def calculate_std_similarity(text1, text2):
    """Calculate both TED similarity and structural similarity efficiently."""
    # Cache the BeautifulSoup parsing and tree creation
    soup1, soup2 = BeautifulSoup(text1, "html.parser"), BeautifulSoup(text2, "html.parser")
    
    # Create trees once for both calculations - fix the unpacking error
    tree1 = parse_html_to_tree(soup1, ignore_text=False)
    tree2 = parse_html_to_tree(soup2, ignore_text=False)
    tree1_struct = parse_html_to_tree(soup1, ignore_text=True)
    tree2_struct = parse_html_to_tree(soup2, ignore_text=True)
    
    # Calculate distances
    ted_distance = tree_edit_distance(tree1, tree2)
    struct_distance = tree_edit_distance(tree1_struct, tree2_struct)
    
    # Calculate node counts once
    max_nodes = max(count_nodes(tree1), count_nodes(tree2))
    max_struct_nodes = max(count_nodes(tree1_struct), count_nodes(tree2_struct))
    
    # Return normalized scores
    return (1 - (ted_distance / max_nodes) if max_nodes > 0 else 1,
            1 - (struct_distance / max_struct_nodes) if max_struct_nodes > 0 else 1)


def compare_tables(gt_file, extracted_file):
    """Compare tables between Azure and extracted files with performance optimizations."""
    # Load both files
    with open(gt_file, 'r', encoding='utf-8') as f:
        gt_tables = json.load(f)
    
    with open(extracted_file, 'r', encoding='utf-8') as f:
        extracted_tables = json.load(f)
    
    # Pre-normalize all tables once to avoid repeated normalization
    gt_tables_normalized = []
    for table in gt_tables:
        gt_tables_normalized.append({
            'original': table,
            'normalized': normalize_table_html(table['sentence']),
            'page': table['page']
        })
    
    extracted_tables_normalized = []
    for table in extracted_tables:
        extracted_tables_normalized.append({
            'original': table,
            'normalized': normalize_table_html(table['sentence']),
            'page': table['page']
        })
    
    # Group extracted tables by page for faster lookup
    extracted_by_page = {}
    for idx, table in enumerate(extracted_tables_normalized):
        page = table['page']
        if page not in extracted_by_page:
            extracted_by_page[page] = []
        extracted_by_page[page].append((idx, table))
    
    # Track total and scores
    total_similarity = 0
    total_structure_similarity = 0
    matched_tables = 0
    detailed_matches = []
    
    # Compare each GT table with extracted tables
    for i, gt_table_norm in enumerate(gt_tables_normalized, 1):
        gt_text = gt_table_norm['original']['sentence']
        gt_page = gt_table_norm['page']
        gt_text_norm = gt_table_norm['normalized']
        
        # Find best matching extracted table
        best_match = None
        best_similarity = 0
        best_structure_similarity = 0
        best_idx = None
        best_match_original = None

        # Only compare with tables on the same page if available
        tables_to_compare = extracted_by_page.get(gt_page, [])
        
        # If no tables on this page, compare with all tables (fallback)
        if not tables_to_compare:
            tables_to_compare = [(idx, table) for idx, table in enumerate(extracted_tables_normalized)]
        
        for j, (orig_idx, extracted_table_norm) in enumerate(tables_to_compare, 1):
            extracted_text_norm = extracted_table_norm['normalized']
            
            # Calculate similarity metrics
            similarity, structure_similarity = calculate_std_similarity(gt_text_norm, extracted_text_norm)
            
            if structure_similarity > best_structure_similarity:
                best_structure_similarity = structure_similarity
                best_similarity = similarity
                best_match = extracted_table_norm
                best_idx = orig_idx + 1  # +1 because enumeration starts at 1
                best_match_original = extracted_table_norm['original']

        if best_match:
            # Add to totals
            total_similarity += best_similarity
            total_structure_similarity += best_structure_similarity
            matched_tables += 1

            detailed_matches.append({
                "azure_table_index": i,
                "mineru_table_index": best_idx,
                "mineru_table_page": best_match_original.get('page', 'N/A'),
                "azure_table_page": gt_page,
                "similarity_score": best_similarity,
                "structure_similarity_score": best_structure_similarity,
                "mineru_text_preview": best_match_original['sentence'][:200] + "..." if len(best_match_original['sentence']) > 200 else best_match_original['sentence'],
                "azure_text_preview": gt_text[:200] + "..." if gt_text and len(gt_text) > 200 else gt_text,
                "normalized_mineru_text": best_match['normalized'][:200] + "..." if len(best_match['normalized']) > 200 else best_match['normalized'],
                "normalized_azure_text": gt_text_norm[:200] + "..." if gt_text_norm and len(gt_text_norm) > 200 else gt_text_norm,
            })
        else:
            print(f"No matching table found for GT table #{i} on page {gt_page}!")

    average_similarity = total_similarity / matched_tables if matched_tables > 0 else 0
    average_structure_similarity = total_structure_similarity / matched_tables if matched_tables > 0 else 0
            
    return {
        "total_matched_tables": matched_tables,
        "average_similarity": average_similarity,
        "average_structure_similarity": average_structure_similarity,
        "total_similarity": total_similarity,
        "total_structure_similarity": total_structure_similarity,
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
    
    # Get all JSON files
    azure_files = [f for f in os.listdir(azure_folder) if f.endswith('.json')]
    
    print(f"\nComparison started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"MinerU folder: {mineru_folder}")
    print(f"Azure folder: {azure_folder}")
    print(f"\nProcessing {len(azure_files)} files in parallel...")

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
    
    with tqdm(total=len(azure_files), desc="Processing files", file=sys.stdout) as pbar:
        for result in pool.imap_unordered(process_file, azure_files):
            if result:
                base_name, file_results = result
                all_results[base_name] = file_results
                total_avg_similarity += file_results['average_similarity']
                total_avg_structure_similarity += file_results['average_structure_similarity']
                total_files += 1
            pbar.update(1)
    
    pool.close()
    pool.join()
    
    # Calculate overall statistics
    overall_stats = {
        "total_files_processed": total_files,
        "overall_average_similarity": total_avg_similarity / total_files if total_files > 0 else 0,
        "overall_average_structure_similarity": total_avg_structure_similarity / total_files if total_files > 0 else 0,
        "timestamp": timestamp
    }
    
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