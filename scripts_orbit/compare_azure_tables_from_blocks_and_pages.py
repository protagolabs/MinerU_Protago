#!/usr/bin/env python3
"""
Table Comparison Tool - Compares tables extracted using different methods

This script compares tables extracted from two different methods to identify
differences and validate extraction quality.
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
import difflib
from bs4 import BeautifulSoup
import pandas as pd
from tabulate import tabulate


def load_tables(file_path: str) -> List[Dict[str, Any]]:
    """Load tables from a JSON file.
    
    Args:
        file_path: Path to the JSON file containing tables
        
    Returns:
        List of tables
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_table_content(table_html: str) -> str:
    """Extract clean table content from HTML for comparison.
    
    Args:
        table_html: HTML string containing table
        
    Returns:
        Cleaned table content as string
    """
    soup = BeautifulSoup(table_html, 'html.parser')
    table = soup.find('table')
    if not table:
        return ""
    
    # Extract rows and cells
    rows = []
    for tr in table.find_all('tr'):
        row = []
        for td in tr.find_all(['td', 'th']):
            # Get text and normalize whitespace
            text = ' '.join(td.get_text().split())
            row.append(text)
        rows.append('|'.join(row))
    
    return '\n'.join(rows)


def compare_tables(tables1: List[Dict[str, Any]], tables2: List[Dict[str, Any]]) -> Tuple[int, int, List[Dict]]:
    """Compare tables from two different sources.
    
    Args:
        tables1: Tables from first source
        tables2: Tables from second source
        
    Returns:
        Tuple containing count of matching tables, total tables, and detailed comparison
    """
    comparison_results = []
    
    # Extract content from all tables for comparison
    contents1 = [extract_table_content(table["sentence"]) for table in tables1]
    contents2 = [extract_table_content(table["sentence"]) for table in tables2]
    
    # Track matches
    matched_indices1 = set()
    matched_indices2 = set()
    
    # Compare each table from source 1 with each from source 2
    for i, content1 in enumerate(contents1):
        best_match_idx = -1
        best_match_ratio = 0
        
        for j, content2 in enumerate(contents2):
            if j in matched_indices2:
                continue  # Skip already matched tables
                
            # Calculate similarity ratio
            ratio = difflib.SequenceMatcher(None, content1, content2).ratio()
            
            if ratio > best_match_ratio and ratio > 0.7:  # 70% similarity threshold
                best_match_ratio = ratio
                best_match_idx = j
        
        # Record the comparison result
        result = {
            "table1_idx": i,
            "table1_page": tables1[i]["page"],
            "table2_idx": best_match_idx,
            "table2_page": tables2[best_match_idx]["page"] if best_match_idx >= 0 else None,
            "similarity": best_match_ratio if best_match_idx >= 0 else 0,
            "status": "matched" if best_match_idx >= 0 else "unmatched"
        }
        comparison_results.append(result)
        
        # Mark as matched
        if best_match_idx >= 0:
            matched_indices1.add(i)
            matched_indices2.add(best_match_idx)
    
    # Add unmatched tables from source 2
    for j in range(len(contents2)):
        if j not in matched_indices2:
            result = {
                "table1_idx": None,
                "table1_page": None,
                "table2_idx": j,
                "table2_page": tables2[j]["page"],
                "similarity": 0,
                "status": "only_in_source2"
            }
            comparison_results.append(result)
    
    # Count matches
    match_count = sum(1 for r in comparison_results if r["status"] == "matched")
    total_count = len(tables1) + len(tables2) - match_count
    
    return match_count, total_count, comparison_results


def compare_files(file1: str, file2: str) -> Dict[str, Any]:
    """Compare tables from two files.
    
    Args:
        file1: Path to first table file
        file2: Path to second table file
        
    Returns:
        Dictionary with comparison results
    """
    tables1 = load_tables(file1)
    tables2 = load_tables(file2)
    
    match_count, total_count, comparison_results = compare_tables(tables1, tables2)
    
    return {
        "file1": file1,
        "file2": file2,
        "tables_in_file1": len(tables1),
        "tables_in_file2": len(tables2),
        "matched_tables": match_count,
        "match_percentage": round(match_count / total_count * 100, 2) if total_count > 0 else 0,
        "comparison_details": comparison_results
    }


def compare_folders(folder1: str, folder2: str, output_folder: str) -> None:
    """Compare all matching files in two folders.
    
    Args:
        folder1: Path to first folder
        folder2: Path to second folder
        output_folder: Path to save comparison results
    """
    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Get list of files in both folders
    files1 = {f.stem.split('.')[0]: f for f in Path(folder1).glob('*.json')}
    files2 = {f.stem.split('.')[0]: f for f in Path(folder2).glob('*.json')}
    
    # Find common file bases
    common_files = set(files1.keys()) & set(files2.keys())
    
    print(f"Found {len(common_files)} common files to compare")
    
    # Summary data for all files
    summary_data = []
    
    # Process each common file
    for i, base_name in enumerate(sorted(common_files), 1):
        file1 = str(files1[base_name])
        file2 = str(files2[base_name])
        
        print(f"[{i}/{len(common_files)}] Comparing {base_name}...")
        
        try:
            # Compare the files
            result = compare_files(file1, file2)
            
            # Add to summary
            summary_data.append({
                "file_name": base_name,
                "tables_in_method1": result["tables_in_file1"],
                "tables_in_method2": result["tables_in_file2"],
                "matched_tables": result["matched_tables"],
                "match_percentage": result["match_percentage"]
            })
            
            # Save detailed comparison
            output_file = os.path.join(output_folder, f"{base_name}_comparison.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=4)
                
        except Exception as e:
            print(f"  Error comparing {base_name}: {str(e)}")
    
    # Create summary report
    if summary_data:
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(summary_data)
        
        # Calculate overall statistics
        total_tables_method1 = df["tables_in_method1"].sum()
        total_tables_method2 = df["tables_in_method2"].sum()
        total_matched = df["matched_tables"].sum()
        overall_match_pct = round(total_matched / (total_tables_method1 + total_tables_method2 - total_matched) * 100, 2)
        
        # Add summary row
        summary_row = {
            "file_name": "TOTAL",
            "tables_in_method1": total_tables_method1,
            "tables_in_method2": total_tables_method2,
            "matched_tables": total_matched,
            "match_percentage": overall_match_pct
        }
        summary_data.append(summary_row)
        
        # Save summary to CSV
        summary_file = os.path.join(output_folder, "comparison_summary.csv")
        pd.DataFrame(summary_data).to_csv(summary_file, index=False)
        
        # Print summary table
        print("\nComparison Summary:")
        print(tabulate(summary_data, headers="keys", tablefmt="grid"))
        print(f"\nDetailed results saved to {output_folder}")


def main():
    """Main function to parse arguments and run the comparison."""
    parser = argparse.ArgumentParser(
        description="Compare tables extracted using different methods"
    )
    parser.add_argument(
        "-m1", "--method1", 
        required=True,
        help="Folder containing tables extracted with first method"
    )
    parser.add_argument(
        "-m2", "--method2", 
        required=True,
        help="Folder containing tables extracted with second method"
    )
    parser.add_argument(
        "-o", "--output", 
        required=True,
        help="Output folder to save comparison results"
    )
    
    args = parser.parse_args()
    
    print(f"Starting table comparison process")
    print(f"Method 1 folder: {args.method1}")
    print(f"Method 2 folder: {args.method2}")
    print(f"Output folder: {args.output}")
    
    compare_folders(args.method1, args.method2, args.output)
    
    print("Comparison process completed")


if __name__ == "__main__":
    main()