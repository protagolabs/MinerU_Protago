import time
import cv2
import numpy as np
from PIL import Image
from loguru import logger


class CustomTableModel:
    """
    Example custom table model implementation
    This is a template you can modify for your own table processing needs
    """
    
    def __init__(self, ocr_engine, model_path=None, confidence_threshold=0.5, **kwargs):
        """
        Initialize your custom table model
        Args:
            ocr_engine: OCR engine for text recognition
            model_path: Path to your model weights (if needed)
            confidence_threshold: Confidence threshold for filtering results
            **kwargs: Additional configuration
        """
        self.ocr_engine = ocr_engine
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        
        # Initialize your custom model here
        logger.info(f"Initializing CustomTableModel with confidence_threshold={confidence_threshold}")
        
        # Example: Load your custom model weights
        if model_path:
            logger.info(f"Loading custom model from: {model_path}")
            # self.model = load_your_custom_model(model_path)
        
    def predict(self, image):
        """
        Process table image and return HTML
        Args:
            image: PIL Image or numpy array
            
        Returns:
            tuple: (html_code, table_cell_bboxes, logic_points, elapse_time)
        """
        start_time = time.time()
        
        try:
            # Convert image to proper format
            if isinstance(image, Image.Image):
                image_array = np.array(image)
            else:
                image_array = image
                
            # Process the table
            html_code = self.process_table(image_array)
            
            # Extract additional information (optional)
            table_cell_bboxes = self.extract_cell_bboxes(image_array)
            logic_points = self.extract_logic_points(image_array)
            
            elapse_time = time.time() - start_time
            
            logger.info(f"CustomTableModel processing completed in {elapse_time:.3f}s")
            
            return html_code, table_cell_bboxes, logic_points, elapse_time
            
        except Exception as e:
            logger.error(f"Error in CustomTableModel.predict: {e}")
            return None, None, None, time.time() - start_time
    
    def process_table(self, image_array):
        """
        Main table processing logic - customize this for your needs
        """
        # Method 1: OCR-based approach
        return self._ocr_based_table_extraction(image_array)
        
        # Method 2: Vision-based approach (uncomment to use)
        # return self._vision_based_table_extraction(image_array)
        
        # Method 3: Hybrid approach (uncomment to use)
        # return self._hybrid_table_extraction(image_array)
    
    def _ocr_based_table_extraction(self, image_array):
        """
        Extract table structure using OCR + heuristics
        This is a simple example - you can make it more sophisticated
        """
        # Convert to BGR for OCR
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            bgr_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        else:
            bgr_image = image_array
            
        # Run OCR to get text and positions
        ocr_result = self.ocr_engine.ocr(bgr_image)[0]
        
        if not ocr_result:
            return "<table><tr><td>No text detected</td></tr></table>"
        
        # Extract text and bounding boxes
        text_boxes = []
        for item in ocr_result:
            if len(item) == 2 and isinstance(item[1], tuple):
                bbox = item[0]
                text = item[1][0]
                confidence = item[1][1]
                
                if confidence >= self.confidence_threshold:
                    text_boxes.append({
                        'text': text,
                        'bbox': bbox,
                        'confidence': confidence
                    })
        
        # Simple table structure detection
        html_table = self._build_html_from_text_boxes(text_boxes)
        return html_table
    
    def _build_html_from_text_boxes(self, text_boxes):
        """
        Build HTML table from detected text boxes
        This is a simplified approach - you can implement more sophisticated logic
        """
        if not text_boxes:
            return "<table><tr><td>No content detected</td></tr></table>"
        
        # Sort text boxes by vertical position (top to bottom)
        text_boxes.sort(key=lambda x: x['bbox'][0][1])
        
        # Group text boxes into rows based on Y-coordinate proximity
        rows = self._group_into_rows(text_boxes)
        
        # Build HTML
        html_parts = ["<table border='1'>"]
        
        for row_idx, row in enumerate(rows):
            html_parts.append("<tr>")
            
            # Sort cells in row by X-coordinate (left to right)
            row.sort(key=lambda x: x['bbox'][0][0])
            
            for cell in row:
                # Clean text
                clean_text = self._clean_text(cell['text'])
                html_parts.append(f"<td>{clean_text}</td>")
            
            html_parts.append("</tr>")
        
        html_parts.append("</table>")
        
        return "".join(html_parts)
    
    def _group_into_rows(self, text_boxes, y_threshold=20):
        """
        Group text boxes into rows based on Y-coordinate proximity
        """
        if not text_boxes:
            return []
        
        rows = []
        current_row = [text_boxes[0]]
        current_y = text_boxes[0]['bbox'][0][1]
        
        for box in text_boxes[1:]:
            box_y = box['bbox'][0][1]
            
            # If Y-coordinate is close to current row, add to current row
            if abs(box_y - current_y) <= y_threshold:
                current_row.append(box)
            else:
                # Start new row
                rows.append(current_row)
                current_row = [box]
                current_y = box_y
        
        # Add the last row
        if current_row:
            rows.append(current_row)
        
        return rows
    
    def _clean_text(self, text):
        """
        Clean and normalize text content
        """
        if not text:
            return ""
        
        # Remove extra whitespace
        cleaned = " ".join(text.split())
        
        # Escape HTML special characters
        cleaned = cleaned.replace("&", "&amp;")
        cleaned = cleaned.replace("<", "&lt;")
        cleaned = cleaned.replace(">", "&gt;")
        cleaned = cleaned.replace('"', "&quot;")
        cleaned = cleaned.replace("'", "&#x27;")
        
        return cleaned
    
    def _vision_based_table_extraction(self, image_array):
        """
        Extract table structure using computer vision techniques
        Implement this if you have a vision-based table detection model
        """
        # Placeholder for vision-based approach
        # You can implement:
        # - Line detection for table borders
        # - Cell segmentation
        # - Structure analysis
        
        logger.info("Vision-based table extraction not implemented yet")
        return "<table><tr><td>Vision-based extraction placeholder</td></tr></table>"
    
    def _hybrid_table_extraction(self, image_array):
        """
        Combine OCR and vision-based approaches
        """
        # Get OCR results
        ocr_html = self._ocr_based_table_extraction(image_array)
        
        # Get vision results (if implemented)
        # vision_html = self._vision_based_table_extraction(image_array)
        
        # Combine or choose best result
        return ocr_html
    
    def extract_cell_bboxes(self, image_array):
        """
        Extract cell bounding boxes (optional)
        """
        # Implement if you need cell-level bounding boxes
        return []
    
    def extract_logic_points(self, image_array):
        """
        Extract logical structure points (optional)
        """
        # Implement if you need logical structure information
        return []
    
    def set_confidence_threshold(self, threshold):
        """
        Update confidence threshold
        """
        self.confidence_threshold = threshold
        logger.info(f"Updated confidence threshold to {threshold}")
    
    def get_model_info(self):
        """
        Get model information
        """
        return {
            "model_name": "CustomTableModel",
            "version": "1.0.0",
            "model_path": self.model_path,
            "confidence_threshold": self.confidence_threshold
        } 