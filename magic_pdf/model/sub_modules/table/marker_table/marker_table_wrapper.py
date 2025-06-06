import os
import time
from pathlib import Path
from PIL import Image
import numpy as np
from loguru import logger

from marker.converters.table import TableConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from marker.config.parser import ConfigParser

class MarkerTableWrapper:
    """
    Wrapper class for Marker's TableConverter that provides a predict interface
    compatible with MinerU's table model interface.
    """
    
    def __init__(self, config=None):
        """
        Initialize MarkerTableWrapper
        
        Args:
            config: Configuration dictionary for TableConverter
                   If None, uses default config
        """
        if config is None:
            config = {
                "output_format": "json",
                "force_layout_block": "Table"
            }
        config_parser = ConfigParser(config)    
        self.config = config
        try:
            self.converter = TableConverter(config=config_parser.generate_config_dict(), artifact_dict=create_model_dict(),renderer=config_parser.get_renderer())
            # logger.debug("MarkerTableWrapper initialized")
        except Exception as e:
            logger.error(f"Failed to initialize TableConverter: {e}")
            raise
        
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
        
        # Get current working directory for temp file
        current_dir = os.getcwd()
        temp_path = os.path.join(current_dir, f"temp_marker_table_{os.getpid()}_{int(time.time())}.png")
        
        try:
            # Save image to temp file
            image.save(temp_path)
            
            # Process with Marker's TableConverter
            rendered = self.converter(temp_path)
            
            # Extract text and convert to HTML
            text, _, _ = text_from_rendered(rendered)
            # if self.config['output_format'] == 'html':
            #     html_code = self._convert_to_html(text)
            # else:
            #     html_code = text
            html_code = text
            
            # Calculate elapsed time
            elapse = time.time() - start_time
            
            return html_code, [], [], elapse
            
        except Exception as e:
            logger.error(f"MarkerTableWrapper prediction failed: {e}")
            return None, None, None, None
            
        finally:
            # Cleanup temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    def _convert_to_html(self, text):
        """Convert markdown table text to HTML"""
        if not text:
            return None
            
        lines = text.strip().split('\n')
        if len(lines) < 2:  # Need at least header and separator
            return None
            
        html = ['<table border="1" style="border-collapse: collapse;">']
        
        # Process each line
        is_header = True
        for line in lines:
            line = line.strip()
            if not line or line.startswith('|---'):  # Skip separator lines
                is_header = False
                continue
                
            # Split and clean cells
            cells = [cell.strip() for cell in line.split('|')]
            cells = [cell for cell in cells if cell]  # Remove empty cells
            
            if not cells:
                continue
                
            # Create row
            row = []
            tag = 'th' if is_header else 'td'
            for cell in cells:
                row.append(f'<{tag} style="border: 1px solid black; padding: 8px;">{cell}</{tag}>')
            
            html.append(f"<tr>{''.join(row)}</tr>")
            
        html.append('</table>')
        return '\n'.join(html) 