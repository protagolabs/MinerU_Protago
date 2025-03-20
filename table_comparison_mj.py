# #!/usr/bin/env python3
# """
# Table Comparison Tool

# This script compares table structures between MinerU and Azure JSON files,
# calculating detailed similarity scores and generating comprehensive reports.

# Author: Xing
# Date: 2024-01-20
# """

# import os
# import json
# import re
# import sys
# import argparse
# from pathlib import Path
# from datetime import datetime
# from difflib import SequenceMatcher
# from bs4 import BeautifulSoup
# from typing import Dict, Any, Optional, Set, List
# from tqdm import tqdm
# import numpy as np
# import multiprocessing
# from functools import partial

# class Logger:
#     """Custom logger that writes to both console and file."""
#     def __init__(self, filename: str):
#         self.terminal = sys.stdout
#         self.log = open(filename, 'w', encoding='utf-8')
#         self.last_msg = ""

#     def write(self, message: str) -> None:
#         # Only log non-progress bar updates to file
#         if '\r' not in message:  # Progress bars typically use carriage returns
#             self.terminal.write(message)
#             self.log.write(message)
#             self.last_msg = message
#         else:
#             # For progress bars, only write to terminal
#             self.terminal.write(message)
#             # If it's the final progress bar update (ends with newline)
#             if '\n' in message:
#                 self.log.write(f"Progress complete\n")

#     def flush(self) -> None:
#         self.terminal.flush()
#         self.log.flush()

#     def close(self) -> None:
#         self.log.flush()
#         self.log.close()

# class TreeNode:
#     """Represents a node in the HTML tree."""
#     def __init__(self, tag, children=None, text=""):
#         self.tag = tag
#         self.children = children if children else []
#         self.text = text.strip()

#     def __repr__(self):
#         return f"TreeNode({self.tag}, children={len(self.children)}, text='{self.text}')"

# def parse_html_to_tree(soup, ignore_text=False):
#     """Recursively converts a BeautifulSoup object into a tree structure.
    
#     Args:
#         soup: BeautifulSoup object
#         ignore_text: If True, ignores text content (for structural comparison)
#     """
#     if not soup.name:  # Text node
#         if ignore_text:
#             return None
#         return TreeNode(tag="text", text=soup.strip()) if soup.strip() else None

#     children = [parse_html_to_tree(child, ignore_text) for child in soup.children if parse_html_to_tree(child, ignore_text)]
#     return TreeNode(tag=soup.name, 
#                    children=children, 
#                    text="" if ignore_text else soup.get_text().strip())

# def tree_edit_distance(tree1, tree2):
#     """Computes the Tree Edit Distance (TED) using dynamic programming."""
#     if tree1 is None:
#         return sum(count_nodes(child) for child in tree2.children) + 1 if tree2 else 0
#     if tree2 is None:
#         return sum(count_nodes(child) for child in tree1.children) + 1 if tree1 else 0

#     if tree1.tag == tree2.tag:  # Same node type
#         cost = 0 if tree1.text == tree2.text else 1  # Text difference penalty
#     else:
#         cost = 1  # Different node type penalty

#     dp = np.zeros((len(tree1.children) + 1, len(tree2.children) + 1))

#     for i in range(len(tree1.children) + 1):
#         for j in range(len(tree2.children) + 1):
#             if i == 0:
#                 dp[i][j] = j
#             elif j == 0:
#                 dp[i][j] = i
#             else:
#                 dp[i][j] = min(
#                     dp[i - 1][j] + count_nodes(tree1.children[i - 1]),  # Deletion
#                     dp[i][j - 1] + count_nodes(tree2.children[j - 1]),  # Insertion
#                     dp[i - 1][j - 1] + tree_edit_distance(tree1.children[i - 1], tree2.children[j - 1])  # Substitution
#                 )

#     return dp[len(tree1.children)][len(tree2.children)] + cost

# def count_nodes(tree):
#     """Counts the total nodes in a tree (used for normalization)."""
#     if tree is None:
#         return 0
#     return 1 + sum(count_nodes(child) for child in tree.children)

# def ted_similarity(html1, html2):
#     """Computes the TED similarity between two HTML tables."""
#     soup1, soup2 = BeautifulSoup(html1, "html.parser"), BeautifulSoup(html2, "html.parser")
    
#     tree1, tree2 = parse_html_to_tree(soup1, ignore_text=False), parse_html_to_tree(soup2, ignore_text=False)
    
#     ted_distance = tree_edit_distance(tree1, tree2)
    
#     max_nodes = max(count_nodes(tree1), count_nodes(tree2))
    
#     return 1 - (ted_distance / max_nodes) if max_nodes > 0 else 1  # Normalized similarity score

# def structural_ted_similarity(html1, html2):
#     """Computes the TED similarity focusing only on structure (ignoring text content)."""
#     soup1, soup2 = BeautifulSoup(html1, "html.parser"), BeautifulSoup(html2, "html.parser")
    
#     # Pass ignore_text=True to ignore text content
#     tree1, tree2 = parse_html_to_tree(soup1, ignore_text=True), parse_html_to_tree(soup2, ignore_text=True)
    
#     ted_distance = tree_edit_distance(tree1, tree2)
    
#     max_nodes = max(count_nodes(tree1), count_nodes(tree2))
    
#     return 1 - (ted_distance / max_nodes) if max_nodes > 0 else 1  # Normalized score

# def clean_table_text(text):
#     """Clean table text by removing HTML tags, extra spaces and standardizing whitespace"""
#     # Remove HTML tags
#     text = re.sub(r'<[^>]+>', ' ', text)
#     # Remove extra spaces and standardize whitespace
#     text = ' '.join(text.split())
#     return text

# def normalize_table_html(html_str: str) -> str:
#     """Normalize table HTML string by removing extra whitespace and standardizing attributes.
    
#     Args:
#         html_str: Input HTML table string
        
#     Returns:
#         Normalized HTML table string
#     """
#     # Remove newlines and extra whitespace
#     html_str = re.sub(r'\s+', ' ', html_str)
    
#     # Normalize colspan="1" rowspan="1" by removing them (they're default values)
#     html_str = re.sub(r'\s+colspan="1"', '', html_str)
#     html_str = re.sub(r'\s+rowspan="1"', '', html_str)
    
#     # Remove extra spaces between tags
#     html_str = re.sub(r'>\s+<', '><', html_str)
    
#     # Remove space before closing tags
#     html_str = re.sub(r'\s+>', '>', html_str)
    
#     # Remove space after opening tags
#     html_str = re.sub(r'<\s+', '<', html_str)
    
#     return html_str

# def calculate_similarity(text1: str, text2: str) -> float:
#     """Calculate similarity ratio between two texts using SequenceMatcher.
    
#     Args:
#         text1: First text for comparison
#         text2: Second text for comparison
        
#     Returns:
#         Similarity ratio between 0 and 1
#     """
#     return SequenceMatcher(None, text1, text2).ratio()

# def calculate_std_similarity(text1, text2):

#     return ted_similarity(text1, text2), structural_ted_similarity(text1, text2)

# # def compare_table_files(mineru_file: str, azure_file: str) -> Optional[Dict[str, Any]]:
# #     """Compare tables between MinerU and Azure files with detailed matching.
    
# #     Args:
# #         mineru_file: Path to MinerU JSON file
# #         azure_file: Path to Azure JSON file
        
# #     Returns:
# #         Dictionary containing detailed comparison results or None if error occurs
# #     """
# #     try:
# #         with open(mineru_file, 'r', encoding='utf-8') as f:
# #             mineru_tables = json.load(f)
# #         with open(azure_file, 'r', encoding='utf-8') as f:
# #             azure_tables = json.load(f)
        
# #         total_similarity = 0
# #         matched_tables = 0
# #         detailed_matches: List[Dict[str, Any]] = []
        
# #         # Compare each MinerU table with Azure tables
# #         for i, mineru_table in enumerate(mineru_tables, 1):
# #             mineru_text = clean_table_text(mineru_table['sentence'])
# #             best_similarity = 0
# #             best_match_idx = None
# #             best_match_text = None
            
# #             # Find best matching Azure table
# #             for j, azure_table in enumerate(azure_tables, 1):
# #                 azure_text = clean_table_text(azure_table['sentence'])
                
# #                 similarity = calculate_similarity(mineru_text, azure_text)
                
# #                 if similarity > best_similarity:
# #                     best_similarity = similarity
# #                     best_match_idx = j
# #                     best_match_text = azure_text
            
# #             if best_similarity > 0:
# #                 total_similarity += best_similarity
# #                 matched_tables += 1
                
# #                 # Store detailed match information
# #                 detailed_matches.append({
# #                     "mineru_table_index": i,
# #                     "mineru_table_page": mineru_table.get('page', 'N/A'),
# #                     "azure_table_index": best_match_idx,
# #                     "azure_table_page": azure_tables[best_match_idx-1].get('page', 'N/A') if best_match_idx else 'N/A',
# #                     "similarity_score": best_similarity,
# #                     "mineru_text_preview": mineru_text[:200] + "..." if len(mineru_text) > 200 else mineru_text,
# #                     "azure_text_preview": best_match_text[:200] + "..." if best_match_text and len(best_match_text) > 200 else best_match_text
# #                 })
        
# #         average_similarity = total_similarity / matched_tables if matched_tables > 0 else 0
        
# #         return {
# #             "total_tables": matched_tables,
# #             "average_similarity": average_similarity,
# #             "total_similarity": total_similarity,
# #             "mineru_table_count": len(mineru_tables),
# #             "azure_table_count": len(azure_tables),
# #             "detailed_matches": detailed_matches,
# #             "file_stats": {
# #                 "mineru_file": mineru_file,
# #                 "azure_file": azure_file,
# #                 "unmatched_tables": len(mineru_tables) - matched_tables
# #             }
# #         }
    
# #     except Exception as e:
# #         print(f"Error processing files {mineru_file} and {azure_file}: {str(e)}")
# #         return None

# def compare_tables(gt_file, extracted_file):
#     """Compare tables between Azure and extracted files"""
#     # Load both files
#     with open(gt_file, 'r', encoding='utf-8') as f:
#         gt_tables = json.load(f)
    
#     with open(extracted_file, 'r', encoding='utf-8') as f:
#         extracted_tables = json.load(f)
    
    
#     # Track total and scores
#     total_similarity = 0
#     total_structure_similarity = 0
#     matched_tables = 0
#     detailed_matches: List[Dict[str, Any]] = []
    
#     # Compare each Azure table with each extracted table
#     for i, table in enumerate(gt_tables, 1):
#         # azure_text = clean_table_text(azure_table['sentence'])
#         gt_text = table['sentence']
#         gt_page = table['page']
        
#         # print(f"\nAzure Table #{i} (Page {gt_page}):")
#         # print("First few words:", ' '.join(gt_text.split()[:10]), "...")
        
#         gt_text_norm = normalize_table_html(gt_text)
#         # Find best matching extracted table
#         best_match = None
#         best_similarity = 0
#         best_structure_similarity = 0
#         best_idx = None

#         # print(gt_page)
#         extracted_tables_on_page = [table for table in extracted_tables if table['page'] == gt_page]


#         for j, extracted_table in enumerate(extracted_tables, 1):
#         # for j, extracted_table in enumerate(extracted_tables_on_page, 1):
#             # extracted_text = clean_table_text(extracted_table['sentence'])
#             extracted_text = extracted_table['sentence']
#             extracted_text_norm = normalize_table_html(extracted_text)

#             # print(gt_text_norm)
#             # print(extracted_text_norm)
#             # similarity = calculate_similarity(azure_text, extracted_text)
#             similarity, structure_similarity = calculate_std_similarity(gt_text_norm, extracted_text_norm)
            

#             # print(similarity)
#             # print(structure_similarity)
#             # print(f"Similarity: {similarity:.2%}, Structure Similarity: {structure_similarity:.2%}")
            
#             # if similarity > best_similarity:
#             if structure_similarity >= best_structure_similarity:
#                 best_structure_similarity = structure_similarity
#                 best_similarity = similarity
#                 best_match = extracted_table
                
#                 best_idx = j
#             # else:

#             #     print(best_match)
#             #     print(structure_similarity)
#             #     print(best_structure_similarity)

#         # print(best_match)
#         if best_match:
#             # print(f"Best match: Extracted Table #{best_idx} (Page {best_match['page']})")
#             # print(f"Similarity score: {best_similarity:.2%}")
#             # print(f"Structure Similarity: {best_structure_similarity:.2%}")
            
#             # Add to totals
#             total_similarity += best_similarity
#             total_structure_similarity += best_structure_similarity
#             matched_tables += 1
#             best_match_text_norm = normalize_table_html(best_match['sentence'])

#             detailed_matches.append({
#                 "azure_table_index": i,
#                 "mineru_table_index": best_idx,
#                 "mineru_table_page": best_match.get('page', 'N/A'),
#                 "azure_table_page": gt_page,
#                 "similarity_score": best_similarity,
#                 "structure_similarity_score": best_structure_similarity,
#                 "mineru_text_preview": best_match['sentence'][:200] + "..." if len(best_match['sentence']) > 200 else best_match['sentence'],
#                 "azure_text_preview": gt_text[:200] + "..." if gt_text and len(gt_text) > 200 else gt_text,
#                 "normalized_mineru_text": best_match_text_norm[:200] + "..." if len(best_match_text_norm) > 200 else best_match_text_norm,
#                 "normalized_azure_text": gt_text_norm[:200] + "..." if gt_text_norm and len(gt_text_norm) > 200 else gt_text_norm,
#             })

            
#         else:
#             # print(best_match)
#             print("No matching table found!")


#     if matched_tables > 0:

#         average_similarity = total_similarity / matched_tables if matched_tables > 0 else 0
#         average_structure_similarity = total_structure_similarity / matched_tables if matched_tables > 0 else 0
            

#     return {
#     "total_matched_tables": matched_tables,
#     "average_similarity": average_similarity,
#     "average_structure_similarity": average_structure_similarity,
#     "total_similarity": total_similarity,
#     "total_structure_similarity": total_structure_similarity,
#     "mineru_table_count": len(extracted_tables),
#     "azure_table_count": len(gt_tables),
#     "detailed_matches": detailed_matches,
#     "file_stats": {
#         "mineru_file": extracted_file,
#         "azure_file": gt_file,
#         }
#     }

# def process_single_file(azure_file, mineru_folder, azure_folder):
#     """Process a single file comparison."""
#     try:
#         base_name = azure_file.replace('.pages.tables.json', '')
#         mineru_file = base_name + '.tables.json'
#         mineru_path = os.path.join(mineru_folder, mineru_file)
#         azure_path = os.path.join(azure_folder, azure_file)
        
#         if os.path.exists(mineru_path):
#             results = compare_tables(azure_path, mineru_path)
#             return base_name, results
#         return None
#     except Exception as e:
#         print(f"Error processing {azure_file}: {str(e)}")
#         return None
    
# def process_folders(mineru_folder: str, 
#                    azure_folder: str, 
#                    output_dir: str,
#                    num_processes: int = None) -> Dict[str, Any]:
#     """Process and compare all corresponding files in both folders."""
#     Path(output_dir).mkdir(parents=True, exist_ok=True)
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
#     # Set up logging
#     log_file = os.path.join(output_dir, f"comparison_log_{timestamp}.txt")
#     sys.stdout = Logger(log_file)
    
#     # Get all JSON files
#     azure_files = [f for f in os.listdir(azure_folder) if f.endswith('.json')]
    
#     print(f"\nComparison started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
#     print(f"MinerU folder: {mineru_folder}")
#     print(f"Azure folder: {azure_folder}")
#     print(f"\nProcessing {len(azure_files)} files in parallel...")

#     # Set up multiprocessing
#     if num_processes is None:
#         num_processes = multiprocessing.cpu_count()
#     pool = multiprocessing.Pool(processes=num_processes)
    
#     # Create partial function with fixed arguments
#     process_file = partial(process_single_file, 
#                          mineru_folder=mineru_folder, 
#                          azure_folder=azure_folder)
    
#     # Process files in parallel with progress bar
#     all_results = {}
#     total_avg_similarity = 0
#     total_files = 0
#     total_avg_structure_similarity = 0
    
#     with tqdm(total=len(azure_files), desc="Processing files", file=sys.stdout) as pbar:
#         for result in pool.imap_unordered(process_file, azure_files):
#             if result:
#                 base_name, file_results = result
#                 all_results[base_name] = file_results
#                 total_avg_similarity += file_results['average_similarity']
#                 total_avg_structure_similarity += file_results['average_structure_similarity']
#                 total_files += 1
#             pbar.update(1)
    
#     pool.close()
#     pool.join()
    
#     # Calculate overall statistics
#     overall_stats = {
#         "total_files_processed": total_files,
#         "overall_average_similarity": total_avg_similarity / total_files if total_files > 0 else 0,
#         "overall_average_structure_similarity": total_avg_structure_similarity / total_files if total_files > 0 else 0,
#         "timestamp": timestamp
#     }
    
#     # Save detailed results
#     save_results(all_results, overall_stats, output_dir, timestamp)
    
#     # Restore original stdout and close log file
#     if isinstance(sys.stdout, Logger):
#         sys.stdout.close()
#         sys.stdout = sys.stdout.terminal
    
#     return all_results

# def save_results(results: Dict[str, Any], 
#                 overall_stats: Dict[str, Any], 
#                 output_dir: str, 
#                 timestamp: str) -> None:
#     """Save comparison results to JSON file.
    
#     Args:
#         results: Dictionary containing comparison results
#         overall_stats: Dictionary containing overall statistics
#         output_dir: Output directory path
#         timestamp: Timestamp string for filename
#     """
#     output_file = os.path.join(output_dir, f"comparison_results_{timestamp}.json")
    
#     with open(output_file, 'w', encoding='utf-8') as f:
#         json.dump({
#             "summary": overall_stats,
#             "detailed_results": results
#         }, f, indent=4, ensure_ascii=False)
    
#     print(f"\nResults saved to: {output_file}")

# def parse_args() -> argparse.Namespace:
#     """Parse command line arguments.
    
#     Returns:
#         Parsed command line arguments
#     """
#     parser = argparse.ArgumentParser(
#         description="Compare table structures between MinerU and Azure JSON files."
#     )
#     parser.add_argument(
#         "--mineru-tables",
#         type=str,
#         default="inputs/export_pdf/azure_tables",
#         help="Path to folder containing MinerU table JSON files"
#     )
#     parser.add_argument(
#         "--azure-tables",
#         type=str,
#         default="azure_outputs_tables",
#         help="Path to folder containing Azure table JSON files"
#     )
#     parser.add_argument(
#         "--output-dir",
#         type=str,
#         default="comparison_results",
#         help="Path to output directory for results"
#     )
#     parser.add_argument(
#         "--processes",
#         type=int,
#         default=None,
#         help="Number of parallel processes to use (default: number of CPU cores)"
#     )
#     return parser.parse_args()

# def main() -> None:
#     try:
#         args = parse_args()
        
#         # Validate input folders
#         for folder in [args.mineru_tables, args.azure_tables]:
#             if not os.path.isdir(folder):
#                 raise ValueError(f"Directory not found: {folder}")
        
#         # Run comparison
#         results = process_folders(
#             args.mineru_tables,
#             args.azure_tables,
#             args.output_dir,
#             args.processes
#         )
        
#         sys.exit(0)
#     except Exception as e:
#         print(f"Error: {str(e)}", file=sys.stderr)
#         sys.exit(1)

# if __name__ == "__main__":
#     main()