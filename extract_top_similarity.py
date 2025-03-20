#!/usr/bin/env python3
"""
Extract top structure similarities from table comparison results.
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
import sys
import shutil
from tqdm import tqdm

def extract_top_similarities(results_file: str, top_n: int = 20) -> List[Dict[str, Any]]:
    """
    Extract the top N entries with highest structure similarity scores.
    
    Args:
        results_file: Path to the comparison results JSON file
        top_n: Number of top entries to extract (default: 20)
        
    Returns:
        List of dictionaries containing the top entries
    """
    # Load the results file
    with open(results_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract all detailed matches from all files
    all_matches = []
    for file_id, file_results in data["detailed_results"].items():
        for match in file_results.get("detailed_matches", []):
            # Add file identifier to each match
            match["file_id"] = file_id
            
            all_matches.append(match)
    
    # Sort by structure similarity score in descending order
    sorted_matches = sorted(
        all_matches, 
        key=lambda x: x.get("structure_similarity_score", 0), 
        reverse=True
    )
    
    # Return the top N entries
    return sorted_matches[:top_n]

def copy_top_similarities_layout_files(top_matches: List[Dict[str, Any]], output_dir: str) -> None:
    """
    Print the top matches in a readable format.
    
    Args:
        top_matches: List of top match dictionaries
    """
    # print(f"\nTop {len(top_matches)} Matches by Structure Similarity:\n")
    # print(f"{'Rank':<5} {'File ID':<25} {'Azure Table':<12} {'MinerU Table':<12} {'Structure Sim':<15} {'Content Sim':<15}")
    # print("-" * 90)
    
    # for i, match in enumerate(top_matches, 1):
    #     print(f"{i:<5} {match['file_id'][:23]:<25} {match['azure_table_index']:<12} "
    #           f"{match['mineru_table_index']:<12} {match['structure_similarity_score']:<15.4f} "
    #           f"{match['similarity_score']:<15.4f}")
    

    print(output_dir)
    for match in tqdm(top_matches):
        src = f"/home/xing/MinerU_Protago/azure_outputs_ocr/{match['file_id']}/ocr/{match['file_id']}_layout.pdf"
        shutil.copy(src, output_dir)
    
    # print("\nDetailed information for top match:")
    # top = top_matches[0]
    # print(f"File ID: {top['file_id']}")
    # print(f"Azure Table: {top['azure_table_index']} (Page {top['azure_table_page']})")
    # print(f"MinerU Table: {top['mineru_table_index']} (Page {top['mineru_table_page']})")
    # print(f"Structure Similarity: {top['structure_similarity_score']:.4f}")
    # print(f"Content Similarity: {top['similarity_score']:.4f}")
    # print("\nAzure Text Preview:")
    # print(top['azure_text_preview'])
    # print("\nMinerU Text Preview:")
    # print(top['mineru_text_preview'])

def main() -> None:
    """Main function to parse arguments and run the extraction."""
    parser = argparse.ArgumentParser(
        description="Extract top structure similarities from table comparison results."
    )
    parser.add_argument(
        "results_file",
        type=str,
        help="Path to the comparison results JSON file"
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of top entries to extract (default: 20)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file to save results (default: print to console)"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not Path(args.results_file).exists():
        print(f"Error: Results file not found: {args.results_file}", file=sys.stderr)
        sys.exit(1)
    
    # Extract top similarities
    top_matches = extract_top_similarities(args.results_file, args.top)
    
    # Output results

    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(top_matches, f, indent=2)
    print(f"Top {args.top} matches saved to: {args.output}")

    output_path = Path(args.output)



    print(f"Copying layout files to: {output_path.parent}")
    copy_top_similarities_layout_files(top_matches, output_path.parent)

if __name__ == "__main__":
    main()