import os
import time
import threading
import uuid
import queue
from pathlib import Path
from PIL import Image
import numpy as np
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from marker.converters.table import TableConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from marker.config.parser import ConfigParser

class MarkerTableWrapper:
    """
    Wrapper class for Marker's TableConverter that provides a predict interface
    compatible with MinerU's table model interface.
    """
    
    def __init__(self, config=None, max_workers=1):
        """
        Initialize MarkerTableWrapper
        
        Args:
            config: Configuration dictionary for TableConverter
                   If None, uses default config
            max_workers: Number of worker threads and converter instances to create
                        More workers = more GPU memory usage but better parallelization
        """
        if config is None:
            config = {
                "output_format": "json",
                "force_layout_block": "Table"
            }
        config_parser = ConfigParser(config)    
        self.config = config
        self.max_workers = max_workers
        
        # Create a pool of converters for parallel processing
        # Number of converters = number of workers for 1:1 mapping
        self.converter_pool = queue.Queue()
        
        try:
            logger.info(f"Initializing {max_workers} MarkerTable converter instances...")
            for i in range(max_workers):
                converter = TableConverter(
                    config=config_parser.generate_config_dict(), 
                    artifact_dict=create_model_dict(),
                    renderer=config_parser.get_renderer()
                )
                self.converter_pool.put(converter)
                logger.debug(f"Converter {i+1}/{max_workers} initialized")
            
            logger.info(f"All {max_workers} MarkerTable converters initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize TableConverter pool: {e}")
            raise
    
    def _get_converter(self):
        """Get a converter from the pool (blocking if none available)"""
        return self.converter_pool.get()
    
    def _return_converter(self, converter):
        """Return a converter to the pool"""
        self.converter_pool.put(converter)
        
    def predict(self, image):
        """
        Predict table structure from image
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            tuple: (html_code, table_cell_bboxes, logic_points, elapse)
                  Following the same interface as RapidTable
        """
        start_time = time.time()
        
        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # image = image.resize((2048, 2048))
        w, h = image.size


        if w < 2048 or h < 2048:
            # Calculate scale factor to make the larger dimension 2048
            scale_factor = 2048 / max(w, h)
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
            image = image.resize((new_w, new_h))
        



        # Get current working directory for temp file
        current_dir = os.getcwd()
        # Create thread-safe temp file name with thread ID and random UUID
        thread_id = threading.get_ident()
        unique_id = str(uuid.uuid4())[:8]
        temp_path = os.path.join(current_dir, f"temp_marker_table_{os.getpid()}_{thread_id}_{unique_id}_{int(time.time())}.png")
        
        # Get a converter from the pool
        converter = self._get_converter()
        
        try:
            # Save image to temp file
            image.save(temp_path)
            
            # Process with Marker's TableConverter (no lock needed since each thread has its own converter)
            rendered = converter(temp_path)
            
            # print(rendered)
            # Extract text and convert to HTML
            # text, _, _ = text_from_rendered(rendered)
            # if self.config['output_format'] == 'html':
            #     html_code = self._convert_to_html(text)
            # else:
            #     html_code = text
            # html_code = text
            html_code = self.extract_table_html_from_json_output(rendered)[0]
            # Calculate elapsed time
            elapse = time.time() - start_time
            
            return html_code, [], [], elapse
            
        except Exception as e:
            logger.error(f"MarkerTableWrapper prediction failed: {e}")
            return None, None, None, None
            
        finally:
            # Return converter to pool
            self._return_converter(converter)
            
            # Cleanup temp file
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError as e:
                    logger.warning(f"Failed to remove temp file {temp_path}: {e}")
                
    def extract_table_html_from_json_output(self,json_output):
        """
        Extract HTML content from blocks with block_type "Table" from a JSONOutput object
        
        Args:
            json_output: JSONOutput object containing the document structure
            
        Returns:
            list: List of HTML contents from Table blocks
        """
        table_htmls = []
        
        def traverse_blocks(block):
            """Recursively traverse blocks to find Table blocks"""
            # Skip if None
            if block is None:
                return
                
            # Handle both Pydantic models and dicts
            if hasattr(block, '__dict__'):
                # For Pydantic models, access attributes directly
                block_type = getattr(block, 'block_type', None)
                html_content = getattr(block, 'html', None)
                children = getattr(block, 'children', [])
                
                # Check if current block is a Table
                if block_type == 'Table' and html_content:
                    table_htmls.append(html_content)
                    
                # Traverse children if they exist
                if children:
                    for child in children:
                        traverse_blocks(child)
            elif isinstance(block, dict):
                # For regular dictionaries
                block_type = block.get('block_type')
                html_content = block.get('html')
                children = block.get('children', [])
                
                if block_type == 'Table' and html_content:
                    table_htmls.append(html_content)
                    
                if children:
                    for child in children:
                        traverse_blocks(child)
            elif isinstance(block, list):
                # If block is a list, traverse each item
                for item in block:
                    traverse_blocks(item)
        
        # Start traversing from the root
        traverse_blocks(json_output)
        return table_htmls

    def predict_batch(self, images, show_progress=True):
        """
        Predict table structure from a list of images using multithreading
        
        Args:
            images: List of PIL Images or numpy arrays
            show_progress: Whether to show progress bar (default: True)
            
        Returns:
            list: List of tuples (html_code, table_cell_bboxes, logic_points, elapse)
                  Following the same interface as other table models
        """
        if not images:
            return []
        
        total_images = len(images)
        logger.info(f"Starting batch processing of {total_images} table images")
        
        # Use all available workers, but don't exceed number of images
        workers_to_use = min(self.max_workers, total_images)
        
        logger.info(f"Using {workers_to_use} worker threads with {self.max_workers} converter instances")
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=workers_to_use) as executor:
            # Submit all prediction tasks
            future_to_index = {
                executor.submit(self.predict, image): idx 
                for idx, image in enumerate(images)
            }
            
            # Initialize results list with None values to maintain order
            batch_results = [None] * len(images)
            
            # Setup progress tracking
            processed_count = 0
            failed_count = 0
            
            # Create progress bar if requested
            if show_progress:
                pbar = tqdm(total=total_images, desc="Table Predict", unit="img")
            
            # Collect results as they complete
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result()
                    batch_results[index] = result
                    if result[0] is None:  # Check if prediction failed
                        failed_count += 1
                except Exception as e:
                    logger.error(f"Failed to process image at index {index}: {e}")
                    batch_results[index] = (None, None, None, None)
                    failed_count += 1
                
                processed_count += 1
                
                # Update progress bar
                if show_progress:
                    pbar.update(1)
                    # Update description with success/failure stats
                    pbar.set_postfix({
                        'success': processed_count - failed_count,
                        'failed': failed_count
                    })
            
            # Close progress bar
            if show_progress:
                pbar.close()
        
        # Final summary
        final_success = processed_count - failed_count
        logger.info(f"Batch processing completed: {final_success}/{total_images} images processed successfully")
        if failed_count > 0:
            logger.warning(f"{failed_count} images failed to process")
        
        return batch_results