"""
Custom Table Model following Surya's Table Recognition patterns
Integrates Marker's high-performance table extraction with MinerU's pipeline
"""

import os
import time
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger

# Import Marker's table converter
try:
    from marker.converters.table import TableConverter
    from marker.models import create_model_dict
    from marker.output import text_from_rendered
    MARKER_AVAILABLE = True
except ImportError:
    logger.warning("Marker not available, falling back to basic OCR")
    MARKER_AVAILABLE = False


class CustomTableModel:
    """
    Custom table model using Marker's TableConverter for table extraction.
    Follows Surya's TableRecPredictor patterns for consistent API.
    """
    
    def __init__(self, ocr_engine, model_path=None, **kwargs):
        """
        Initialize table model following Surya's predictor pattern
        
        Args:
            ocr_engine: OCR engine from MinerU
            model_path: Path to model weights (optional, for interface compatibility)
            **kwargs: Configuration options
        """
        # Store configuration
        self.ocr_engine = ocr_engine
        self.model_path = model_path
        self.confidence_threshold = kwargs.get('confidence_threshold', 0.5)
        self.use_llm = kwargs.get('use_llm', False)
        self.output_format = kwargs.get('output_format', 'html')
        
        # Initialize table converter
        self.table_converter = self._load_table_converter()
        
        # Model metadata
        self.model_name = "CustomTableModel"
        self.backend = "Marker (Surya OCR)" if MARKER_AVAILABLE else "Fallback OCR"
        
        logger.info(f"{self.model_name} initialized with backend: {self.backend}")

    def _load_table_converter(self):
        """Load table converter with proper error handling"""
        if not MARKER_AVAILABLE:
            logger.info("Marker not available, using fallback mode")
            return None
            
        try:
            model_dict = create_model_dict()
            converter = TableConverter(artifact_dict=model_dict)
            logger.info("Marker TableConverter loaded successfully")
            return converter
        except Exception as e:
            logger.error(f"Failed to load Marker TableConverter: {e}")
            return None

    def __call__(self, images, **kwargs):
        """
        Main prediction method following Surya's callable pattern
        
        Args:
            images: Single image or list of images (PIL Images or numpy arrays)
            **kwargs: Additional parameters
            
        Returns:
            List of prediction results, one per image
        """
        # Ensure images is a list
        if not isinstance(images, list):
            images = [images]
        
        results = []
        for image in images:
            result = self.predict_single(image, **kwargs)
            results.append(result)
        
        return results

    def predict(self, image, **kwargs) -> Tuple[str, List, List, float]:
        """
        Legacy predict method for backward compatibility
        Matches rapid_table.py output format exactly
        
        Args:
            image: PIL Image or numpy array
            **kwargs: Additional parameters
            
        Returns:
            tuple: (html_code, table_cell_bboxes, logic_points, elapse)
        """
        result = self.predict_single(image, **kwargs)
        
        if result.get('success', False):
            html_code = result.get('html_code', '')
            table_cell_bboxes = result.get('cells', [])
            logic_points = result.get('rows', [])  # Use rows as logic_points for structure info
            elapse = result.get('processing_time', 0.0)
            return html_code, table_cell_bboxes, logic_points, elapse
        else:
            # Match rapid_table.py format when no results
            return None, None, None, None

    def predict_single(self, image, **kwargs) -> Dict[str, Any]:
        """
        Predict table structure for a single image
        
        Args:
            image: PIL Image or numpy array
            **kwargs: Additional parameters
            
        Returns:
            Dict with prediction results following Surya's result format
        """
        start_time = time.time()
        
        # Convert and validate image
        pil_image = self._convert_image(image)
        if pil_image is None:
            return self._create_error_result("Invalid image format", start_time)
        
        # Extract table
        if self.table_converter is not None:
            result = self._extract_with_marker(pil_image)
        else:
            result = self._extract_with_fallback(pil_image)
        
        # Add timing information
        processing_time = time.time() - start_time
        result['processing_time'] = processing_time
        
        logger.debug(f"Table extraction completed in {processing_time:.3f}s")
        return result

    def _convert_image(self, image) -> Optional[Image.Image]:
        """Convert input to PIL Image with validation"""
        try:
            if isinstance(image, np.ndarray):
                return Image.fromarray(image)
            elif isinstance(image, Image.Image):
                return image
            else:
                logger.error(f"Unsupported image type: {type(image)}")
                return None
        except Exception as e:
            logger.error(f"Image conversion failed: {e}")
            return None

    def _extract_with_marker(self, image: Image.Image) -> Dict[str, Any]:
        """Extract table using Marker's TableConverter"""
        temp_path = self._get_temp_path()
        
        try:
            # Save image for Marker processing
            image.save(temp_path)
            
            # Process with Marker
            rendered = self.table_converter(temp_path)
            text, _, _ = text_from_rendered(rendered)
            
            # Convert to desired format
            html_code = self._convert_to_html(text)
            
            return {
                'html_code': html_code,
                'markdown_text': text,
                'cells': [],  # Marker doesn't provide detailed cell info in this interface
                'rows': [],
                'cols': [],
                'extraction_method': 'marker',
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Marker extraction failed: {e}")
            return self._extract_with_fallback(image)
        finally:
            self._cleanup_temp_file(temp_path)

    def _extract_with_fallback(self, image: Image.Image) -> Dict[str, Any]:
        """Fallback extraction using OCR"""
        try:
            logger.debug("Using fallback OCR extraction")
            
            # Convert to numpy array for OCR
            image_array = np.array(image)
            
            # Use OCR engine
            ocr_result = self.ocr_engine.ocr(image_array)
            
            # Process OCR results
            texts = self._extract_texts_from_ocr(ocr_result)
            html_code = self._create_table_from_texts(texts)
            
            return {
                'html_code': html_code,
                'markdown_text': '\n'.join(texts) if texts else 'No text detected',
                'cells': self._create_cell_data(texts),
                'rows': [],
                'cols': [],
                'extraction_method': 'ocr_fallback',
                'success': len(texts) > 0
            }
            
        except Exception as e:
            logger.error(f"Fallback extraction failed: {e}")
            return self._create_error_result("All extraction methods failed", time.time())

    def _extract_texts_from_ocr(self, ocr_result) -> List[str]:
        """Extract text strings from OCR result structure"""
        texts = []
        if ocr_result and len(ocr_result) > 0:
            for line in ocr_result[0]:
                if len(line) >= 2 and isinstance(line[1], tuple):
                    text = line[1][0].strip()
                    if text:  # Only add non-empty text
                        texts.append(text)
        return texts

    def _create_table_from_texts(self, texts: List[str]) -> str:
        """Create HTML table from text list"""
        if not texts:
            return '<table border="1"><tr><td>No text detected</td></tr></table>'
        
        # Simple table creation - one text per row
        rows = []
        for text in texts:
            rows.append(f'<tr><td>{self._escape_html(text)}</td></tr>')
        
        return f'<table border="1">\n{"".join(rows)}\n</table>'

    def _create_cell_data(self, texts: List[str]) -> List[Dict]:
        """Create cell data structure following Surya's format"""
        cells = []
        for i, text in enumerate(texts):
            cells.append({
                'text': text,
                'bbox': [0, i*20, 100, (i+1)*20],  # Dummy bbox
                'row_id': i,
                'col_id': 0,
                'colspan': 1,
                'rowspan': 1,
                'is_header': i == 0  # First row as header
            })
        return cells

    def _convert_to_html(self, markdown_text: str) -> str:
        """Convert markdown table to HTML format"""
        try:
            lines = [line.strip() for line in markdown_text.split('\n') if line.strip()]
            html_lines = ['<table border="1">']
            
            for i, line in enumerate(lines):
                if '|' not in line:
                    continue
                    
                # Process table row
                cells = [cell.strip() for cell in line.split('|') if cell.strip()]
                
                # Skip separator lines
                if all(set(cell) <= {'-', ' '} for cell in cells):
                    continue
                
                # Determine cell type
                tag = 'th' if i == 0 else 'td'
                
                # Build HTML row
                cell_html = ''.join(f'<{tag}>{self._escape_html(cell)}</{tag}>' for cell in cells)
                html_lines.append(f'<tr>{cell_html}</tr>')
            
            html_lines.append('</table>')
            
            # Return original if no table found
            if len(html_lines) <= 2:
                return f'<pre>{self._escape_html(markdown_text)}</pre>'
            
            return '\n'.join(html_lines)
            
        except Exception as e:
            logger.warning(f"HTML conversion failed: {e}")
            return f'<pre>{self._escape_html(markdown_text)}</pre>'

    def _escape_html(self, text: str) -> str:
        """Escape HTML special characters"""
        return (text.replace('&', '&amp;')
                   .replace('<', '&lt;')
                   .replace('>', '&gt;')
                   .replace('"', '&quot;')
                   .replace("'", '&#x27;'))

    def _get_temp_path(self) -> str:
        """Get temporary file path"""
        return f"temp_table_image_{os.getpid()}.png"

    def _cleanup_temp_file(self, temp_path: str):
        """Clean up temporary file"""
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception as e:
            logger.warning(f"Failed to cleanup temp file {temp_path}: {e}")

    def _create_error_result(self, error_message: str, start_time: float) -> Dict[str, Any]:
        """Create standardized error result"""
        return {
            'html_code': f'<table border="1"><tr><td>Error: {error_message}</td></tr></table>',
            'markdown_text': f'Error: {error_message}',
            'cells': [],
            'rows': [],
            'cols': [],
            'processing_time': time.time() - start_time,
            'extraction_method': 'error',
            'success': False,
            'error': error_message
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        return {
            "name": self.model_name,
            "backend": self.backend,
            "version": "1.1.0",
            "marker_available": MARKER_AVAILABLE,
            "confidence_threshold": self.confidence_threshold,
            "use_llm": self.use_llm,
            "output_format": self.output_format,
            "capabilities": [
                "Multi-language table extraction",
                "Complex table structure detection",
                "Markdown and HTML output",
                "OCR fallback support",
                "High accuracy table recognition"
            ],
            "supported_formats": ["PNG", "JPG", "JPEG", "TIFF", "BMP"],
            "max_image_size": "2048x2048 recommended"
        } 