import os
import pickle
import argparse
from pdf2image import convert_from_path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from multiprocessing import Pool, cpu_count
from functools import partial
import tqdm

def visualize_page_with_pdf(pdf_path, page_data, page_number, output_path=None, images=None):
    """
    Visualize a single page of a PDF with Azure Document Intelligence annotations.
    
    Args:
        pdf_path (str): Path to the PDF file
        page_data: Page data from Azure Document Intelligence
        page_number (int): Page number to process (1-based)
        output_path (str, optional): Path to save the visualization. If None, displays instead.
        images (list, optional): Pre-loaded PDF images to avoid redundant conversion
    """
    # Use provided images or convert PDF to images
    if images is None:
        images = convert_from_path(pdf_path)
    
    # Get the specific page image (0-based index)
    page_image = images[page_number - 1]
    
    # Create figure and axis with the same aspect ratio as the document
    fig, ax = plt.subplots(figsize=(page_data.width, page_data.height))
    
    # Display the PDF page as background
    ax.imshow(page_image, extent=[0, page_data.width, page_data.height, 0])
    
    # For each line in the page
    for line in page_data.lines:
        # Get the polygon points
        points = line.polygon
        
        # Create a polygon patch
        polygon = patches.Polygon(
            [(p.x, p.y) for p in points],
            fill=False,
            edgecolor='blue',
            alpha=0.3
        )
        ax.add_patch(polygon)
    
    ax.set_title(f'Page {page_data.page_number}')
    plt.axis('equal')
    
    # If output path is provided, save the figure
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()

def process_single_file(args):
    """
    Process a single PDF file and its corresponding pickle file.
    
    Args:
        args (tuple): Tuple containing (pdf_file, pdf_dir, pkl_dir, output_dir)
    """
    pdf_file, pdf_dir, pkl_dir, output_dir = args
    
    # Get corresponding pkl file name
    pkl_file = pdf_file.replace('.pdf', '.pkl')
    
    # Full paths
    pdf_path = os.path.join(pdf_dir, pdf_file)
    pkl_path = os.path.join(pkl_dir, pkl_file)
    
    try:
        # Load pickle data
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        # Create output subfolder for this document
        doc_output_dir = os.path.join(output_dir, pdf_file.replace('.pdf', ''))
        os.makedirs(doc_output_dir, exist_ok=True)
        
        # Check which pages need processing
        pages_to_process = []
        for i, page in enumerate(data.pages):
            output_path = os.path.join(doc_output_dir, f'page_{i+1}.png')
            if not os.path.exists(output_path):
                pages_to_process.append((i, page, output_path))
        
        if not pages_to_process:
            return f"Skipping {pdf_file} - all pages already processed"
            
        # Convert PDF to images once for all pages
        images = convert_from_path(pdf_path)
        
        # Process each page that needs processing
        for i, page, output_path in pages_to_process:
            visualize_page_with_pdf(
                pdf_path=pdf_path,
                page_data=page,
                page_number=i+1,
                output_path=output_path,
                images=images
            )
        
        return f"Successfully processed {pdf_file} ({len(pages_to_process)} pages)"
        
    except Exception as e:
        return f"Error processing {pdf_file}: {str(e)}"

def process_all_files(pdf_dir, pkl_dir, output_dir, num_processes=None):
    """
    Process all PDF files and their corresponding pickle files in parallel.
    
    Args:
        pdf_dir (str): Directory containing PDF files
        pkl_dir (str): Directory containing pickle files
        output_dir (str): Directory to save visualizations
        num_processes (int, optional): Number of processes to use. Defaults to CPU count.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of PDF files
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    
    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        return
    
    # Prepare arguments for parallel processing - skip full validation to improve startup time
    args_list = [(pdf_file, pdf_dir, pkl_dir, output_dir) for pdf_file in pdf_files 
                if os.path.exists(os.path.join(pkl_dir, pdf_file.replace('.pdf', '.pkl')))]
    
    if len(args_list) < len(pdf_files):
        print(f"Warning: {len(pdf_files) - len(args_list)} PDF files don't have corresponding PKL files")
    
    # Use maximum available CPUs if num_processes is not specified
    if num_processes is None:
        num_processes = min(cpu_count(), len(args_list))
    
    # Process files in parallel with progress bar
    with Pool(processes=num_processes) as pool:
        results = list(tqdm.tqdm(
            pool.imap(process_single_file, args_list),
            total=len(args_list),
            desc="Processing files"
        ))
    
    # Print results
    for result in results:
        print(result)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Visualize Azure Document Intelligence results on PDF files'
    )
    parser.add_argument(
        '--pdf_dir',
        type=str,
        default="/home/xing/MinerU_Protago/inputs/export_pdf/pdf",
        help='Directory containing PDF files'
    )
    parser.add_argument(
        '--pkl_dir',
        type=str,
        default="/home/xing/MinerU_Protago/inputs/export_pdf/azure_pkl",
        help='Directory containing pickle files from Azure Document Intelligence'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default="/home/xing/MinerU_Protago/inputs/export_pdf/visualizations",
        help='Directory to save visualizations'
    )
    parser.add_argument(
        '--num_processes',
        type=int,
        default=None,
        help='Number of parallel processes to use (default: number of CPU cores)'
    )
    return parser.parse_args()

def main():
    """Main function."""
    # Parse command line arguments
    args = parse_args()
    
    # Validate directories exist
    for dir_path in [args.pdf_dir, args.pkl_dir]:
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"Directory not found: {dir_path}")
    
    # Process all files
    process_all_files(
        args.pdf_dir, 
        args.pkl_dir, 
        args.output_dir, 
        args.num_processes
    )

if __name__ == "__main__":
    main()