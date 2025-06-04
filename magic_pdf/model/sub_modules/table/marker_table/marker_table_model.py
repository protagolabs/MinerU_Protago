import os
import time
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
import torch
from loguru import logger
from typing import Optional, Tuple, Any

from magic_pdf.libs.config_reader import get_device

# Import Marker's table converter
try:
    from marker.converters.table import TableConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered
    MARKER_AVAILABLE = True
except ImportError:
    logger.error("Marker not available. Please install marker-pdf: pip install marker-pdf")
    MARKER_AVAILABLE = False


class MarkerTableModel(object):
    def __init__(self, ocr_engine, model_path=None):
        """
        Initialize MarkerTableModel with Marker table converter integration
        
        Args:
            ocr_engine: OCR engine from MinerU
            model_path: Path to model weights (optional, for interface compatibility)
        """
        self.ocr_engine = ocr_engine
        self.model_path = model_path
        
        # Initialize Marker TableConverter
        if not MARKER_AVAILABLE:
            raise ImportError("Marker is required but not available. Please install: pip install marker-pdf")
        
        self.table_converter = self._init_marker_converter()
        if self.table_converter is None:
            raise RuntimeError("Failed to initialize Marker TableConverter")
        
        logger.info("MarkerTableModel initialized with Marker TableConverter")

    def _init_marker_converter(self) -> Optional[TableConverter]:
        """Initialize Marker's TableConverter"""
        try:
            model_dict = create_model_dict()
            converter = TableConverter(artifact_dict=model_dict)
            logger.info("Marker TableConverter loaded successfully")
            return converter
        except Exception as e:
            logger.error(f"Failed to load Marker TableConverter: {e}")
            return None

    def predict(self, image):
        """
        Predict table structure and extract content using Marker
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            Tuple of (html_code, table_cell_bboxes, logic_points, elapse_time)
        """
        start_time = time.time()
        
        # Convert input to PIL Image if needed
        if isinstance(image, np.ndarray):
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            pil_image = Image.fromarray(image)
        else:
            pil_image = image
        
        # Apply image preprocessing
        processed_image = self._preprocess_image(pil_image)
        
        # Extract table using Marker
        html_code, table_cell_bboxes, logic_points = self._extract_with_marker(processed_image)
        
        elapse_time = time.time() - start_time
        
        return html_code, table_cell_bboxes, logic_points, elapse_time

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image for better table extraction
        Includes rotation detection and correction for portrait tables
        """
        # Convert to numpy array for processing
        img_array = np.array(image)
        bgr_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Check image aspect ratio
        img_height, img_width = bgr_image.shape[:2]
        img_aspect_ratio = img_height / img_width if img_width > 0 else 1.0
        img_is_portrait = img_aspect_ratio > 1.2

        # Detect rotation for portrait images
        if img_is_portrait:
            try:
                det_res = self.ocr_engine.ocr(bgr_image, rec=False)[0]
                is_rotated = False
                
                if det_res:
                    vertical_count = 0
                    for box_ocr_res in det_res:
                        p1, p2, p3, p4 = box_ocr_res
                        
                        # Calculate box dimensions
                        width = abs(p3[0] - p1[0])
                        height = abs(p3[1] - p1[1])
                        aspect_ratio = width / height if height > 0 else 1.0
                        
                        # Count vertical text boxes
                        if aspect_ratio < 0.8:  # Taller than wide
                            vertical_count += 1
                    
                    # Rotate if significant vertical text detected
                    if vertical_count >= len(det_res) * 0.3:
                        is_rotated = True
                        logger.debug("Detected rotated table, applying 90° clockwise rotation")
                
                # Apply rotation if needed
                if is_rotated:
                    rotated_array = cv2.rotate(img_array, cv2.ROTATE_90_CLOCKWISE)
                    return Image.fromarray(rotated_array)
                    
            except Exception as e:
                logger.warning(f"Image preprocessing failed: {e}")
        
        return image

    def _extract_with_marker(self, image: Image.Image) -> Tuple[str, list, list]:
        """Extract table using Marker's TableConverter"""
        temp_path = f"temp_marker_table_{os.getpid()}_{int(time.time())}.png"
        
        try:
            # Save image temporarily for Marker processing
            image.save(temp_path)
            
            # Process with Marker TableConverter
            rendered = self.table_converter(temp_path)
            text, _, _ = text_from_rendered(rendered)
            
            # Convert markdown to HTML
            html_code = self._markdown_to_html(text)
            
            # Marker doesn't provide detailed cell/row/column info in this interface
            # Return empty lists for compatibility
            table_cell_bboxes = []
            logic_points = []
            
            logger.debug(f"Marker extraction successful, HTML length: {len(html_code)}")
            return html_code, table_cell_bboxes, logic_points
            
        except Exception as e:
            logger.error(f"Marker extraction failed: {e}")
            return None, None, None
        finally:
            # Cleanup temporary file
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception as e:
                    logger.warning(f"Failed to cleanup temp file {temp_path}: {e}")

    def _markdown_to_html(self, markdown_text: str) -> str:
        """
        Convert markdown table to HTML format
        Enhanced conversion for better table formatting
        """
        if not markdown_text or not markdown_text.strip():
            return "<table><tr><td>No table data extracted</td></tr></table>"
        
        try:
            lines = [line.strip() for line in markdown_text.split('\n') if line.strip()]
            html_lines = ['<table border="1" style="border-collapse: collapse;">']
            
            header_processed = False
            
            for line in lines:
                if '|' not in line:
                    continue
                
                # Split cells and clean up
                cells = [cell.strip() for cell in line.split('|')]
                # Remove empty cells at start/end
                while cells and not cells[0]:
                    cells.pop(0)
                while cells and not cells[-1]:
                    cells.pop()
                
                if not cells:
                    continue
                
                # Skip separator lines (containing only -, |, and spaces)
                if all(set(cell.replace('-', '').replace(' ', '')) == set() for cell in cells):
                    continue
                
                # Determine if this is a header row
                is_header = not header_processed and any(cell.strip() for cell in cells)
                tag = 'th' if is_header else 'td'
                
                # Build HTML row
                cell_html = ''.join(
                    f'<{tag} style="border: 1px solid black; padding: 8px;">{self._escape_html(cell)}</{tag}>' 
                    for cell in cells if cell.strip()
                )
                
                if cell_html:
                    html_lines.append(f'<tr>{cell_html}</tr>')
                    if is_header:
                        header_processed = True
            
            html_lines.append('</table>')
            
            # Return formatted result
            if len(html_lines) > 2:
                return '\n'.join(html_lines)
            else:
                # If no proper table found, return original text in a table
                return f'<table border="1"><tr><td style="border: 1px solid black; padding: 8px;">{self._escape_html(markdown_text)}</td></tr></table>'
                
        except Exception as e:
            logger.warning(f"HTML conversion failed: {e}")
            return f'<table border="1"><tr><td style="border: 1px solid black; padding: 8px;">{self._escape_html(markdown_text)}</td></tr></table>'

    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters"""
        if not text:
            return ""
        return (str(text).replace('&', '&amp;')
                        .replace('<', '&lt;')
                        .replace('>', '&gt;')
                        .replace('"', '&quot;')
                        .replace("'", '&#x27;'))
