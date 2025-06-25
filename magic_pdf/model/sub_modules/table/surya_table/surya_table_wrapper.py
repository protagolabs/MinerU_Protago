import re
import time
from collections import defaultdict
from copy import deepcopy
from typing import Annotated, List, Union, Tuple, Optional
from collections import Counter

from PIL import Image as PILImage
import numpy as np
from ftfy import fix_text
from surya.detection import DetectionPredictor
from surya.recognition import RecognitionPredictor, OCRResult
from surya.table_rec import TableRecPredictor
from surya.table_rec.schema import TableResult, TableCell as SuryaTableCell
from pdftext.extraction import table_output

from marker.processors import BaseProcessor
from marker.schema import BlockTypes
from marker.schema.blocks.tablecell import TableCell
from marker.schema.document import Document
from marker.schema.polygon import PolygonBox
from marker.settings import settings
from marker.util import matrix_intersection_area
from loguru import logger


class SuryaTableWrapper:
    """
    Wrapper for Surya table recognition models with improved processing
    based on Marker's TableConverter approach
    """
    
    def __init__(self, config=None):
        """
        Initialize the wrapper with enhanced configuration
        
        Args:
            config: Configuration dictionary with optional parameters:
                - detection_batch_size: Batch size for detection model
                - table_rec_batch_size: Batch size for table recognition
                - recognition_batch_size: Batch size for OCR recognition
                - row_split_threshold: Threshold for row splitting (default: 0.6)
                - enable_text_cleaning: Enable enhanced text cleaning (default: True)
                - preserve_whitespace: Preserve meaningful whitespace (default: True)
                - disable_tqdm: Disable progress bars (default: True)
                - language: Default language for OCR (default: 'en')
        """
        self.config = config or {}
        
        # Enhanced configuration parameters
        self.detection_batch_size = self.config.get('detection_batch_size', None)
        self.table_rec_batch_size = self.config.get('table_rec_batch_size', None) 
        self.recognition_batch_size = self.config.get('recognition_batch_size', None)
        self.row_split_threshold = self.config.get('row_split_threshold', 0.6)  # Increased from 0.5
        self.enable_text_cleaning = self.config.get('enable_text_cleaning', True)
        self.preserve_whitespace = self.config.get('preserve_whitespace', True)
        
        # Add missing attributes to match TableProcessor
        self.disable_tqdm = self.config.get('disable_tqdm', True)
        self.language = self.config.get('language', 'en')
        
        # Initialize models
        try:
            from surya.detection import DetectionPredictor
            from surya.recognition import RecognitionPredictor
            from surya.table_rec import TableRecPredictor
            
            self.detection_model = DetectionPredictor()
            self.recognition_model = RecognitionPredictor()
            self.table_rec_model = TableRecPredictor()
            
            logger.debug("SuryaTableWrapper initialized with enhanced processing")
        except Exception as e:
            logger.error(f"Failed to initialize Surya models: {e}")
            raise

    def predict(self, image, language=None):
        """
        Predict table structure from image with enhanced processing
        """
        start_time = time.time()
        
        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = PILImage.fromarray(image)
        
        try:
            # Get table structure
            tables = self.table_rec_model([image], batch_size=1)
            
            if not tables or len(tables) == 0:
                return None, [], [], time.time() - start_time
            
            table = tables[0]
            
            # Get OCR text lines
            ocr_lines = self._get_ocr_lines_batch([image], [language or 'en'])
            
            if ocr_lines and len(ocr_lines) > 0:
                # Enhanced text assignment with better intersection logic
                self._assign_text_to_cells_enhanced([table], [{'table_text_lines': ocr_lines[0]}])
            
            # Enhanced post-processing
            self._split_combined_rows_enhanced([table])
            self._combine_dollar_column_enhanced([table])
            
            # Generate enhanced HTML
            html_code = self._generate_html_enhanced(table)
            
            elapse = time.time() - start_time
            return html_code, [], [], elapse
            
        except Exception as e:
            logger.error(f"SuryaTableWrapper prediction failed: {e}")
            return None, [], [], time.time() - start_time

    def predict_batch(self, images: List[Union[PILImage.Image, np.ndarray]], 
                     languages: Optional[List[str]] = None) -> List[Tuple]:
        """
        Enhanced batch prediction with improved processing and error handling
        """
        if not images:
            return []
        
        # Convert all images to PIL format
        pil_images = []
        for img in images:
            if isinstance(img, np.ndarray):
                pil_images.append(PILImage.fromarray(img))
            else:
                pil_images.append(img)
        
        # Set default languages
        if languages is None:
            languages = ['en'] * len(pil_images)
        elif len(languages) != len(pil_images):
            languages = [languages[0] if languages else 'en'] * len(pil_images)
        
        batch_results = []
        batch_size = self.get_table_rec_batch_size()
        
        for i in range(0, len(pil_images), batch_size):
            batch_images = pil_images[i:i + batch_size]
            batch_langs = languages[i:i + batch_size]
            
            try:
                # Get table structures for batch
                tables = self.table_rec_model(batch_images, batch_size=len(batch_images))
                
                if not tables:
                    logger.warning(f"No table structures found for batch {i}")
                    # Add empty results for failed batch
                    for _ in range(len(batch_images)):
                        batch_results.append((None, [], [], 0))
                    continue
                
                # Get OCR for batch
                ocr_batch = self._get_ocr_lines_batch(batch_images, batch_langs)
                
                # Ensure OCR batch matches table count
                if len(ocr_batch) != len(tables):
                    logger.warning(f"OCR batch size ({len(ocr_batch)}) doesn't match table count ({len(tables)})")
                    # Pad with empty lists if needed
                    while len(ocr_batch) < len(tables):
                        ocr_batch.append([])
                
                # Process each table in the batch
                for j, table in enumerate(tables):
                    start_time = time.time()
                    
                    try:
                        # Get OCR lines for this table (or empty list if failed)
                        ocr_lines = ocr_batch[j] if j < len(ocr_batch) else []
                        
                        # Assign text to cells even if OCR is empty
                        if ocr_lines:
                            self._assign_text_to_cells_enhanced([table], [{'table_text_lines': ocr_lines}])
                        else:
                            logger.debug(f"No OCR lines for table {j}, proceeding with structure only")
                        
                        # Enhanced post-processing
                        self._split_combined_rows_enhanced([table])
                        self._combine_dollar_column_enhanced([table])
                        
                        # Generate enhanced HTML even without OCR text
                        html_code = self._generate_html_enhanced(table)
                        
                        elapse = time.time() - start_time
                        batch_results.append((html_code, [], [], elapse))
                        
                    except Exception as table_error:
                        logger.error(f"Error processing table {j}: {table_error}")
                        batch_results.append((None, [], [], time.time() - start_time))
                    
            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
                logger.debug(f"Batch processing error details: {type(e).__name__}: {str(e)}")
                # Add empty results for failed batch
                for _ in range(len(batch_images)):
                    batch_results.append((None, [], [], 0))
        
        return batch_results

    def _get_ocr_lines_batch(self, images: List[PILImage.Image], languages: List[str]) -> List[List[dict]]:
        """
        Enhanced OCR processing with better text extraction and error handling
        """
        try:
            # Validate inputs
            if not images:
                logger.warning("No images provided for OCR processing")
                return []
            
            # Set up tqdm disable flags
            self.recognition_model.disable_tqdm = self.disable_tqdm if hasattr(self, 'disable_tqdm') else True
            self.detection_model.disable_tqdm = self.disable_tqdm if hasattr(self, 'disable_tqdm') else True
            
            # Prepare language lists for each image (Surya expects list of lists)
            langs_per_image = [[lang] for lang in languages]
            
            logger.debug(f"Processing {len(images)} images with languages: {languages}")
            
            # Get OCR results using the correct Surya API (matching TableProcessor.assign_ocr_lines)
            ocr_results: List[OCRResult] = self.recognition_model(
                images=images,
                langs=langs_per_image,
                det_predictor=self.detection_model,
                recognition_batch_size=self.get_recognition_batch_size(),
                detection_batch_size=self.get_detection_batch_size(),
            )
            
            # Check if OCR results are valid
            if ocr_results is None:
                logger.error("OCR model returned None results")
                return [[] for _ in images]
            
            if len(ocr_results) != len(images):
                logger.error(f"OCR results count ({len(ocr_results)}) doesn't match images count ({len(images)})")
                return [[] for _ in images]
            
            batch_lines = []
            for idx, ocr_result in enumerate(ocr_results):
                lines = []
                
                # Handle case where ocr_result might be None
                if ocr_result is None:
                    logger.warning(f"OCR result for image {idx} is None")
                    batch_lines.append([])
                    continue
                
                # Check if text_lines attribute exists
                if not hasattr(ocr_result, 'text_lines') or ocr_result.text_lines is None:
                    logger.warning(f"OCR result for image {idx} has no text_lines")
                    batch_lines.append([])
                    continue
                
                for line in ocr_result.text_lines:
                    try:
                        # Enhanced text cleaning and formatting
                        text = line.text.strip() if hasattr(line, 'text') and line.text else ""
                        if self.enable_text_cleaning and text:
                            text = self._clean_text_enhanced(text)
                        
                        if text:  # Only include non-empty text
                            bbox = getattr(line, 'bbox', [0, 0, 0, 0])
                            confidence = getattr(line, 'confidence', 1.0)
                            
                            lines.append({
                                'text': text,
                                'bbox': bbox,
                                'confidence': confidence
                            })
                    except Exception as line_error:
                        logger.warning(f"Error processing text line: {line_error}")
                        continue
                
                batch_lines.append(lines)
                logger.debug(f"Processed image {idx}: found {len(lines)} text lines")
            
            return batch_lines
            
        except Exception as e:
            logger.error(f"OCR processing failed: {e}")
            logger.debug(f"OCR error details: {type(e).__name__}: {str(e)}")
            # Return empty results for all images to prevent downstream errors
            return [[] for _ in images]

    def _assign_text_to_cells_enhanced(self, tables: List[TableResult], table_data: list):
        """
        Enhanced text assignment with improved intersection logic
        """
        for table_result, table_page_data in zip(tables, table_data):
            table_text_lines = table_page_data["table_text_lines"]
            table_cells: List[SuryaTableCell] = table_result.cells
            
            if not table_text_lines or not table_cells:
                continue
                
            text_line_bboxes = [t["bbox"] for t in table_text_lines]
            table_cell_bboxes = [c.bbox for c in table_cells]

            # Enhanced intersection matrix calculation
            intersection_matrix = matrix_intersection_area(
                text_line_bboxes, table_cell_bboxes
            )

            cell_text = defaultdict(list)
            
            # Improved text assignment logic
            for text_line_idx, table_text_line in enumerate(table_text_lines):
                try:
                    intersections = intersection_matrix[text_line_idx]
                    if intersections.sum() == 0:
                        continue

                    # Find best matching cell with threshold
                    max_intersection_idx = intersections.argmax()
                    max_intersection_value = intersections[max_intersection_idx]
                    
                    # Validate the index is within bounds
                    if max_intersection_idx >= len(table_cells):
                        logger.warning(f"Intersection index {max_intersection_idx} out of bounds for {len(table_cells)} cells")
                        continue
                    
                    # Calculate intersection ratio for better accuracy
                    text_bbox = text_line_bboxes[text_line_idx]
                    text_area = (text_bbox[2] - text_bbox[0]) * (text_bbox[3] - text_bbox[1])
                    intersection_ratio = max_intersection_value / max(text_area, 1)
                    
                    # Only assign if intersection is significant (>= 0.1 overlap)
                    if intersection_ratio >= 0.1:
                        cell_text[max_intersection_idx].append(table_text_line)
                        
                except Exception as assign_error:
                    logger.warning(f"Error assigning text line {text_line_idx}: {assign_error}")
                    continue

            # Assign text to cells with enhanced sorting
            for cell_idx, text_lines in cell_text.items():
                try:
                    if cell_idx < len(table_cells):
                        # Sort text lines by vertical position for better reading order
                        sorted_text = sorted(text_lines, key=lambda x: (x["bbox"][1], x["bbox"][0]))
                        table_cells[cell_idx].text_lines = sorted_text
                    else:
                        logger.warning(f"Cell index {cell_idx} out of bounds for {len(table_cells)} cells")
                except Exception as cell_assign_error:
                    logger.warning(f"Error assigning text to cell {cell_idx}: {cell_assign_error}")
                    continue

    def _split_combined_rows_enhanced(self, tables: List[TableResult]):
        """
        Enhanced row splitting with improved logic based on Marker's approach
        """
        for table in tables:
            try:
                if not table or not hasattr(table, 'cells') or len(table.cells) == 0:
                    continue
                    
                unique_rows = sorted(list(set([c.row_id for c in table.cells])))
                row_info = []
                
                for row in unique_rows:
                    try:
                        row_cells = deepcopy([c for c in table.cells if c.row_id == row])
                        if not row_cells:  # Skip if no cells in this row
                            continue
                            
                        rowspans = [c.rowspan for c in row_cells]
                        line_lens = [
                            len(c.text_lines) if isinstance(c.text_lines, list) and c.text_lines else 1
                            for c in row_cells
                        ]

                        # Ensure line_lens is not empty
                        if not line_lens:
                            continue

                        # Check for rowspan cells that might interfere
                        rowspan_cells = [
                            c for c in table.cells
                            if c.row_id != row and c.row_id + c.rowspan > row > c.row_id
                        ]
                        
                        # Enhanced splitting criteria
                        should_split_entire_row = all([
                            len(row_cells) > 1,
                            len(rowspan_cells) == 0,
                            all([rowspan == 1 for rowspan in rowspans]),
                            all([line_len > 1 for line_len in line_lens]),
                            len(set(line_lens)) == 1,  # All line lengths are the same
                        ])
                        
                        # Improved partial row splitting logic
                        line_lens_counter = Counter(line_lens)
                        counter_keys = sorted(list(line_lens_counter.keys()))
                        
                        # Ensure we have enough keys before accessing them
                        should_split_partial_row = False
                        if len(counter_keys) >= 2:
                            should_split_partial_row = all([
                                len(row_cells) > 2,  # Reduced from 3 for better detection
                                len(rowspan_cells) == 0,
                                all([r == 1 for r in rowspans]),
                                len(line_lens_counter) == 2,
                                counter_keys[0] <= 1,
                                counter_keys[1] > 1,
                                line_lens_counter[counter_keys[0]] <= len(row_cells) // 2,  # Allow more flexibility
                            ])
                        
                        should_split = should_split_entire_row or should_split_partial_row
                        row_info.append({
                            "should_split": should_split,
                            "row_cells": row_cells,
                            "line_lens": line_lens,
                        })
                        
                    except Exception as row_error:
                        logger.warning(f"Error processing row {row}: {row_error}")
                        continue

                # Enhanced threshold for row splitting
                split_ratio = sum([r["should_split"] for r in row_info]) / len(row_info) if row_info else 0
                if split_ratio < self.row_split_threshold:
                    continue

                # Process row splitting
                new_cells = []
                shift_up = 0
                max_cell_id = max([c.cell_id for c in table.cells]) if table.cells else 0
                new_cell_count = 0
                
                # Ensure unique_rows and row_info have the same length
                if len(unique_rows) != len(row_info):
                    logger.warning(f"Mismatch between unique_rows ({len(unique_rows)}) and row_info ({len(row_info)})")
                    continue
                
                for row, item_info in zip(unique_rows, row_info):
                    try:
                        if not item_info or "line_lens" not in item_info:
                            logger.warning(f"Invalid item_info for row {row}")
                            continue
                            
                        max_lines = max(item_info["line_lens"]) if item_info["line_lens"] else 1
                        
                        if item_info["should_split"]:
                            for i in range(max_lines):
                                for cell in item_info["row_cells"]:
                                    try:
                                        # Validate cell has required attributes
                                        if not hasattr(cell, 'bbox') or not cell.bbox or len(cell.bbox) < 4:
                                            logger.warning(f"Cell missing valid bbox")
                                            continue
                                            
                                        # Enhanced cell splitting with better height calculation
                                        split_height = (cell.bbox[3] - cell.bbox[1]) / max_lines
                                        current_bbox = [
                                            cell.bbox[0],
                                            cell.bbox[1] + i * split_height,
                                            cell.bbox[2],
                                            cell.bbox[1] + (i + 1) * split_height,
                                        ]

                                        # Get text line for this split
                                        line = None
                                        if (hasattr(cell, 'text_lines') and cell.text_lines and 
                                            isinstance(cell.text_lines, list) and i < len(cell.text_lines)):
                                            line = [cell.text_lines[i]]
                                        
                                        cell_id = max_cell_id + new_cell_count
                                        new_cells.append(
                                            SuryaTableCell(
                                                polygon=current_bbox,
                                                text_lines=line,
                                                rowspan=1,
                                                colspan=getattr(cell, 'colspan', 1),
                                                row_id=cell.row_id + shift_up + i,
                                                col_id=getattr(cell, 'col_id', 0),
                                                is_header=getattr(cell, 'is_header', False) and i == 0,
                                                within_row_id=getattr(cell, 'within_row_id', 0),
                                                cell_id=cell_id,
                                            )
                                        )
                                        new_cell_count += 1
                                    except Exception as cell_split_error:
                                        logger.warning(f"Error splitting cell: {cell_split_error}")
                                        continue
                            
                            shift_up += max_lines - 1
                        else:
                            for cell in item_info["row_cells"]:
                                try:
                                    cell.row_id += shift_up
                                    new_cells.append(cell)
                                except Exception as cell_add_error:
                                    logger.warning(f"Error adding cell: {cell_add_error}")
                                    continue
                                
                    except Exception as split_error:
                        logger.warning(f"Error processing row split: {split_error}")
                        continue

                if len(new_cells) > len(table.cells):
                    table.cells = new_cells
                    
            except Exception as table_error:
                logger.warning(f"Error processing table row splitting: {table_error}")
                continue

    def _combine_dollar_column_enhanced(self, tables: List[TableResult]):
        """
        Enhanced dollar column combining with improved detection
        """
        for table in tables:
            try:
                if not table or not hasattr(table, 'cells') or len(table.cells) == 0:
                    continue
                    
                unique_cols = sorted(list(set([c.col_id for c in table.cells])))
                max_col = max(unique_cols) if unique_cols else 0
                dollar_cols = []
                
                for col in unique_cols:
                    try:
                        col_cells = [c for c in table.cells if c.col_id == col]
                        col_text = [
                            "\n".join(self._finalize_cell_text_enhanced(c)).strip() 
                            for c in col_cells
                        ]
                        
                        # Enhanced dollar detection - include common currency symbols
                        currency_symbols = ["", "$", "¥", "€", "£", "₹", "¢"]
                        all_currency = all([ct in currency_symbols for ct in col_text])
                        colspans = [c.colspan for c in col_cells]
                        
                        # Check for cells spanning into this column
                        span_into_col = [
                            c for c in table.cells
                            if c.col_id != col and c.col_id + c.colspan > col > c.col_id
                        ]

                        if all([
                            all_currency,
                            len(col_cells) > 1,
                            len(span_into_col) == 0,
                            all([c == 1 for c in colspans]),
                            col < max_col,
                        ]):
                            # Check if next column exists and has matching rows
                            next_col_cells = [c for c in table.cells if c.col_id == col + 1]
                            next_col_rows = [c.row_id for c in next_col_cells]
                            col_rows = [c.row_id for c in col_cells]
                            
                            if (len(next_col_cells) == len(col_cells) and 
                                set(next_col_rows) == set(col_rows)):
                                dollar_cols.append(col)
                                
                    except Exception as col_error:
                        logger.warning(f"Error processing column {col}: {col_error}")
                        continue

                if len(dollar_cols) == 0:
                    continue

                # Process column combining
                dollar_cols = sorted(dollar_cols)
                col_offset = 0
                
                for col in unique_cols:
                    try:
                        col_cells = [c for c in table.cells if c.col_id == col]
                        
                        if col_offset == 0 and col not in dollar_cols:
                            continue

                        if col in dollar_cols:
                            col_offset += 1
                            for cell in col_cells:
                                try:
                                    text_lines = cell.text_lines if cell.text_lines else []
                                    next_row_col = [
                                        c for c in table.cells
                                        if c.row_id == cell.row_id and c.col_id == col + 1
                                    ]

                                    if next_row_col:
                                        next_text_lines = next_row_col[0].text_lines if next_row_col[0].text_lines else []
                                        # Enhanced text combining
                                        combined_lines = deepcopy(text_lines) + deepcopy(next_text_lines)
                                        next_row_col[0].text_lines = combined_lines
                                        next_row_col[0].col_id -= col_offset
                                        
                                    # Remove the dollar column cell
                                    table.cells = [c for c in table.cells if c.cell_id != cell.cell_id]
                                    
                                except Exception as cell_combine_error:
                                    logger.warning(f"Error combining cell: {cell_combine_error}")
                                    continue
                        else:
                            for cell in col_cells:
                                try:
                                    cell.col_id -= col_offset
                                except Exception as cell_offset_error:
                                    logger.warning(f"Error offsetting cell column: {cell_offset_error}")
                                    continue
                                    
                    except Exception as col_offset_error:
                        logger.warning(f"Error processing column offset: {col_offset_error}")
                        continue
                        
            except Exception as table_error:
                logger.warning(f"Error processing table column combining: {table_error}")
                continue

    def _finalize_cell_text_enhanced(self, cell: SuryaTableCell):
        """
        Enhanced cell text finalization with better cleaning
        """
        if not cell.text_lines:
            return []
            
        fixed_text = []
        text_lines = cell.text_lines if isinstance(cell.text_lines, list) else []
        
        for line in text_lines:
            if isinstance(line, dict):
                text = line.get("text", "").strip()
            else:
                text = str(line).strip()
                
            if not text or text == ".":
                continue
                
            # Enhanced text cleaning
            text = self._clean_text_enhanced(text)
            if text:
                fixed_text.append(text)
                
        return fixed_text

    def _clean_text_enhanced(self, text: str) -> str:
        """
        Enhanced text cleaning based on Marker's approach
        """
        if not text:
            return ""
        
        # Apply ftfy for text fixing
        text = fix_text(text)
        
        # Remove excessive dots and spaces
        text = re.sub(r"(\s\.){2,}", "", text)
        text = re.sub(r"\.{2,}", "", text)
        
        # Clean up whitespace while preserving meaningful spaces
        if self.preserve_whitespace:
            # Preserve single spaces but clean up excessive whitespace
            text = re.sub(r'\s{2,}', ' ', text)
        else:
            text = self._normalize_spaces(text)
        
        # Remove standalone dots
        if text.strip() == ".":
            return ""
            
        return text.strip()

    @staticmethod
    def _normalize_spaces(text):
        """Enhanced space normalization"""
        if not text:
            return ""
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _generate_html_enhanced(self, table: TableResult):
        """
        Enhanced HTML generation with better structure and formatting
        """
        if not table or not hasattr(table, 'cells'):
            logger.warning("Table object is invalid or missing cells attribute")
            return "<table><tr><td>Table structure not available</td></tr></table>"
        
        if not table.cells or len(table.cells) == 0:
            logger.debug("Table has no cells, generating minimal structure")
            return "<table><tr><td>No table content detected</td></tr></table>"
        
        try:
            # Sort cells by row and column for proper table structure
            sorted_cells = sorted(table.cells, key=lambda x: (x.row_id, x.col_id))
            
            # Group cells by row
            rows = defaultdict(list)
            for cell in sorted_cells:
                rows[cell.row_id].append(cell)
            
            # Determine if table has headers
            has_headers = any(cell.is_header for cell in table.cells)
            header_rows = set()
            if has_headers:
                for cell in table.cells:
                    if cell.is_header:
                        header_rows.add(cell.row_id)
            
            html_parts = ['<table>']
            
            # Add thead if headers exist
            if header_rows:
                html_parts.append('<thead>')
                for row_id in sorted(header_rows):
                    if row_id in rows:
                        html_parts.append('<tr>')
                        for cell in rows[row_id]:
                            self._add_cell_html_enhanced(html_parts, cell, is_header=True)
                        html_parts.append('</tr>')
                html_parts.append('</thead>')
            
            # Add tbody for data rows
            data_rows = [r for r in sorted(rows.keys()) if r not in header_rows]
            if data_rows:
                html_parts.append('<tbody>')
                for row_id in data_rows:
                    html_parts.append('<tr>')
                    for cell in rows[row_id]:
                        self._add_cell_html_enhanced(html_parts, cell, is_header=False)
                    html_parts.append('</tr>')
                html_parts.append('</tbody>')
            
            html_parts.append('</table>')
            return '\n'.join(html_parts)
        
        except Exception as e:
            logger.error(f"Error generating HTML from table: {e}")
            return "<table><tr><td>Error generating table HTML</td></tr></table>"

    def _add_cell_html_enhanced(self, html_parts: list, cell: SuryaTableCell, is_header: bool = False):
        """
        Add enhanced cell HTML with proper attributes and content
        """
        try:
            tag = 'th' if is_header or cell.is_header else 'td'
            
            # Build attributes
            attrs = []
            if hasattr(cell, 'rowspan') and cell.rowspan > 1:
                attrs.append(f'rowspan="{cell.rowspan}"')
            if hasattr(cell, 'colspan') and cell.colspan > 1:
                attrs.append(f'colspan="{cell.colspan}"')
            
            attr_str = ' ' + ' '.join(attrs) if attrs else ''
            
            # Get enhanced cell content
            cell_text = self._finalize_cell_text_enhanced(cell)
            content = ' '.join(cell_text) if cell_text else ''
            
            # If no content, show placeholder
            if not content:
                content = '&nbsp;'  # Non-breaking space for empty cells
            
            # Enhanced HTML escaping while preserving structure
            content = (content.replace('&', '&amp;')
                             .replace('<', '&lt;')
                             .replace('>', '&gt;')
                             .replace('"', '&quot;'))
            
            html_parts.append(f'<{tag}{attr_str}>{content}</{tag}>')
            
        except Exception as e:
            logger.warning(f"Error adding cell HTML: {e}")
            # Add a basic cell as fallback
            tag = 'th' if is_header else 'td'
            html_parts.append(f'<{tag}>&nbsp;</{tag}>')

    def _finalize_cell_text(self, cell: SuryaTableCell):
        """
        Finalize and clean cell text (legacy method for compatibility)
        """
        return self._finalize_cell_text_enhanced(cell)

    def _generate_html_from_table(self, table: TableResult):
        """
        Generate HTML from table result (legacy method for compatibility)
        """
        return self._generate_html_enhanced(table)

    def _assign_text_to_cells(self, tables: List[TableResult], table_data: list):
        """
        Assign text to cells (legacy method for compatibility)
        """
        return self._assign_text_to_cells_enhanced(tables, table_data)

    def _split_combined_rows(self, tables: List[TableResult]):
        """
        Split combined rows (legacy method for compatibility)
        """
        return self._split_combined_rows_enhanced(tables)

    def _combine_dollar_column(self, tables: List[TableResult]):
        """
        Combine dollar columns (legacy method for compatibility)
        """
        return self._combine_dollar_column_enhanced(tables)

    def get_detection_batch_size(self):
        if self.detection_batch_size is not None:
            return self.detection_batch_size
        elif settings.TORCH_DEVICE_MODEL == "cuda":
            return 10
        return 4

    def get_table_rec_batch_size(self):
        if self.table_rec_batch_size is not None:
            return self.table_rec_batch_size
        elif settings.TORCH_DEVICE_MODEL == "mps":
            return 6
        elif settings.TORCH_DEVICE_MODEL == "cuda":
            return 14
        return 6

    def get_recognition_batch_size(self):
        if self.recognition_batch_size is not None:
            return self.recognition_batch_size
        elif settings.TORCH_DEVICE_MODEL == "mps":
            return 32
        elif settings.TORCH_DEVICE_MODEL == "cuda":
            return 32
        return 32


class TableProcessor(BaseProcessor):
    """
    A processor for recognizing tables in the document.
    """

    block_types = (BlockTypes.Table, BlockTypes.TableOfContents, BlockTypes.Form)
    detect_boxes: Annotated[
        bool,
        "Whether to detect boxes for the table recognition model.",
    ] = False
    detection_batch_size: Annotated[
        int,
        "The batch size to use for the table detection model.",
        "Default is None, which will use the default batch size for the model.",
    ] = None
    table_rec_batch_size: Annotated[
        int,
        "The batch size to use for the table recognition model.",
        "Default is None, which will use the default batch size for the model.",
    ] = None
    recognition_batch_size: Annotated[
        int,
        "The batch size to use for the table recognition model.",
        "Default is None, which will use the default batch size for the model.",
    ] = None
    contained_block_types: Annotated[
        List[BlockTypes],
        "Block types to remove if they're contained inside the tables.",
    ] = (BlockTypes.Text, BlockTypes.TextInlineMath)
    row_split_threshold: Annotated[
        float,
        "The percentage of rows that need to be split across the table before row splitting is active.",
    ] = 0.5
    pdftext_workers: Annotated[
        int,
        "The number of workers to use for pdftext.",
    ] = 1
    disable_tqdm: Annotated[
        bool,
        "Whether to disable the tqdm progress bar.",
    ] = False
    format_lines: Annotated[
        bool,
        "Whether to format the lines.",
    ] = False
    drop_repeated_text: Annotated[bool, "Drop repeated text in OCR results."] = False
    language: Annotated[str, "Default language for OCR processing."] = "en"

    def __init__(
        self,
        detection_model: DetectionPredictor,
        recognition_model: RecognitionPredictor,
        table_rec_model: TableRecPredictor,
        config=None,
    ):
        super().__init__(config)

        self.detection_model = detection_model
        self.recognition_model = recognition_model
        self.table_rec_model = table_rec_model

    def __call__(self, document: Document):
        filepath = document.filepath  # Path to original pdf file

        table_data = []
        for page in document.pages:
            for block in page.contained_blocks(document, self.block_types):
                image = block.get_image(document, highres=True)
                image_poly = block.polygon.rescale(
                    (page.polygon.width, page.polygon.height),
                    page.get_image(highres=True).size,
                )

                table_data.append(
                    {
                        "block_id": block.id,
                        "page_id": page.page_id,
                        "table_image": image,
                        "table_bbox": image_poly.bbox,
                        "img_size": page.get_image(highres=True).size,
                        "ocr_block": any(
                            [
                                page.text_extraction_method in ["surya", "hybrid"],
                                page.ocr_errors_detected,
                                self.format_lines,
                            ]
                        ),
                    }
                )

        extract_blocks = [t for t in table_data if not t["ocr_block"]]
        self.assign_pdftext_lines(
            extract_blocks, filepath
        )  # Handle tables where good text exists in the PDF

        ocr_blocks = [t for t in table_data if t["ocr_block"]]
        self.assign_ocr_lines(ocr_blocks)  # Handle tables where OCR is needed
        for table_item in table_data:
            if "table_text_lines" not in table_item:
                logger.warning(
                    f"No text lines found for table {table_item['block_id']}"
                )
                table_item["table_text_lines"] = []

        self.table_rec_model.disable_tqdm = self.disable_tqdm
        tables: List[TableResult] = self.table_rec_model(
            [t["table_image"] for t in table_data],
            batch_size=self.get_table_rec_batch_size(),
        )
        self.assign_text_to_cells(tables, table_data)
        self.split_combined_rows(tables)  # Split up rows that were combined
        self.combine_dollar_column(tables)  # Combine columns that are just dollar signs

        # Assign table cells to the table
        table_idx = 0
        for page in document.pages:
            for block in page.contained_blocks(document, self.block_types):
                block.structure = []  # Remove any existing lines, spans, etc.
                cells: List[SuryaTableCell] = tables[table_idx].cells
                for cell in cells:
                    # Rescale the cell polygon to the page size
                    cell_polygon = PolygonBox(polygon=cell.polygon).rescale(
                        page.get_image(highres=True).size, page.polygon.size
                    )

                    # Rescale cell polygon to be relative to the page instead of the table
                    for corner in cell_polygon.polygon:
                        corner[0] += block.polygon.bbox[0]
                        corner[1] += block.polygon.bbox[1]

                    cell_block = TableCell(
                        polygon=cell_polygon,
                        text_lines=self.finalize_cell_text(cell),
                        rowspan=cell.rowspan,
                        colspan=cell.colspan,
                        row_id=cell.row_id,
                        col_id=cell.col_id,
                        is_header=bool(cell.is_header),
                        page_id=page.page_id,
                    )
                    page.add_full_block(cell_block)
                    block.add_structure(cell_block)
                table_idx += 1

        # Clean out other blocks inside the table
        # This can happen with stray text blocks inside the table post-merging
        for page in document.pages:
            child_contained_blocks = page.contained_blocks(
                document, self.contained_block_types
            )
            for block in page.contained_blocks(document, self.block_types):
                intersections = matrix_intersection_area(
                    [c.polygon.bbox for c in child_contained_blocks],
                    [block.polygon.bbox],
                )
                for child, intersection in zip(child_contained_blocks, intersections):
                    # Adjust this to percentage of the child block that is enclosed by the table
                    intersection_pct = intersection / max(child.polygon.area, 1)
                    if intersection_pct > 0.95 and child.id in page.structure:
                        page.structure.remove(child.id)

    def finalize_cell_text(self, cell: SuryaTableCell):
        fixed_text = []
        text_lines = cell.text_lines if cell.text_lines else []
        for line in text_lines:
            text = line["text"].strip()
            if not text or text == ".":
                continue
            text = re.sub(r"(\s\.){2,}", "", text)  # Replace . . .
            text = re.sub(r"\.{2,}", "", text)  # Replace ..., like in table of contents
            text = self.normalize_spaces(fix_text(text))
            fixed_text.append(text)
        return fixed_text

    @staticmethod
    def normalize_spaces(text):
        space_chars = [
            "\u2003",  # em space
            "\u2002",  # en space
            "\u00a0",  # non-breaking space
            "\u200b",  # zero-width space
            "\u3000",  # ideographic space
        ]
        for space in space_chars:
            text = text.replace(space, " ")
        return text

    def combine_dollar_column(self, tables: List[TableResult]):
        for table in tables:
            if len(table.cells) == 0:
                # Skip empty tables
                continue
            unique_cols = sorted(list(set([c.col_id for c in table.cells])))
            max_col = max(unique_cols)
            dollar_cols = []
            for col in unique_cols:
                # Cells in this col
                col_cells = [c for c in table.cells if c.col_id == col]
                col_text = [
                    "\n".join(self.finalize_cell_text(c)).strip() for c in col_cells
                ]
                all_dollars = all([ct in ["", "$"] for ct in col_text])
                colspans = [c.colspan for c in col_cells]
                span_into_col = [
                    c
                    for c in table.cells
                    if c.col_id != col and c.col_id + c.colspan > col > c.col_id
                ]

                # This is a column that is entirely dollar signs
                if all(
                    [
                        all_dollars,
                        len(col_cells) > 1,
                        len(span_into_col) == 0,
                        all([c == 1 for c in colspans]),
                        col < max_col,
                    ]
                ):
                    next_col_cells = [c for c in table.cells if c.col_id == col + 1]
                    next_col_rows = [c.row_id for c in next_col_cells]
                    col_rows = [c.row_id for c in col_cells]
                    
                    if (
                        len(next_col_cells) == len(col_cells)
                        and next_col_rows == col_rows
                    ):
                        dollar_cols.append(col)

            if len(dollar_cols) == 0:
                continue

            dollar_cols = sorted(dollar_cols)
            col_offset = 0
            for col in unique_cols:
                col_cells = [c for c in table.cells if c.col_id == col]
                
                if col_offset == 0 and col not in dollar_cols:
                    continue

                if col in dollar_cols:
                    col_offset += 1
                    for cell in col_cells:
                        text_lines = cell.text_lines if cell.text_lines else []
                        next_row_col = [
                            c
                            for c in table.cells
                            if c.row_id == cell.row_id and c.col_id == col + 1
                        ]

                        # Add dollar to start of the next column
                        next_text_lines = (
                            next_row_col[0].text_lines
                            if next_row_col[0].text_lines
                            else []
                        )
                        next_row_col[0].text_lines = deepcopy(text_lines) + deepcopy(
                            next_text_lines
                        )
                        table.cells = [
                            c for c in table.cells if c.cell_id != cell.cell_id
                        ]  # Remove original cell
                        next_row_col[0].col_id -= col_offset
                else:
                    for cell in col_cells:
                        cell.col_id -= col_offset

    def split_combined_rows(self, tables: List[TableResult]):
        for table in tables:
            if len(table.cells) == 0:
                # Skip empty tables
                continue
            unique_rows = sorted(list(set([c.row_id for c in table.cells])))
            row_info = []
            for row in unique_rows:
                # Cells in this row
                # Deepcopy is because we do an in-place mutation later, and that can cause rows to shift to match rows in unique_rows
                # making them be processed twice
                row_cells = deepcopy([c for c in table.cells if c.row_id == row])
                rowspans = [c.rowspan for c in row_cells]
                line_lens = [
                    len(c.text_lines) if isinstance(c.text_lines, list) and c.text_lines else 1
                    for c in row_cells
                ]

                # Ensure line_lens is not empty
                if not line_lens:
                    continue

                # Check for rowspan cells that might interfere
                rowspan_cells = [
                    c
                    for c in table.cells
                    if c.row_id != row and c.row_id + c.rowspan > row > c.row_id
                ]
                should_split_entire_row = all(
                    [
                        len(row_cells) > 1,
                        len(rowspan_cells) == 0,
                        all([rowspan == 1 for rowspan in rowspans]),
                        all([line_len > 1 for line_len in line_lens]),
                        len(set(line_lens)) == 1,  # All line lengths are the same
                    ]
                )
                line_lens_counter = Counter(line_lens)
                counter_keys = sorted(list(line_lens_counter.keys()))
                
                # Ensure we have enough keys before accessing them
                should_split_partial_row = False
                if len(counter_keys) >= 2:
                    should_split_partial_row = all([
                        len(row_cells) > 2,  # Reduced from 3 for better detection
                        len(rowspan_cells) == 0,
                        all([r == 1 for r in rowspans]),
                        len(line_lens_counter) == 2,
                        counter_keys[0] <= 1,
                        counter_keys[1] > 1,
                        line_lens_counter[counter_keys[0]] <= len(row_cells) // 2,  # Allow more flexibility
                    ])
                
                should_split = should_split_entire_row or should_split_partial_row
                row_info.append(
                    {
                        "should_split": should_split,
                        "row_cells": row_cells,
                        "line_lens": line_lens,
                    }
                )

            # Don't split if we're not splitting most of the rows in the table.  This avoids splitting stray multiline rows.
            if (
                sum([r["should_split"] for r in row_info]) / len(row_info)
                < self.row_split_threshold
            ):
                continue

            new_cells = []
            shift_up = 0
            max_cell_id = max([c.cell_id for c in table.cells])
            new_cell_count = 0
            for row, item_info in zip(unique_rows, row_info):
                max_lines = max(item_info["line_lens"])
                if item_info["should_split"]:
                    for i in range(0, max_lines):
                        for cell in item_info["row_cells"]:
                            # Calculate height based on number of splits
                            split_height = cell.bbox[3] - cell.bbox[1]
                            current_bbox = [
                                cell.bbox[0],
                                cell.bbox[1] + i * split_height,
                                cell.bbox[2],
                                cell.bbox[1] + (i + 1) * split_height,
                            ]

                            line = (
                                [cell.text_lines[i]]
                                if cell.text_lines and i < len(cell.text_lines)
                                else None
                            )
                            cell_id = max_cell_id + new_cell_count
                            new_cells.append(
                                SuryaTableCell(
                                    polygon=current_bbox,
                                    text_lines=line,
                                    rowspan=1,
                                    colspan=cell.colspan,
                                    row_id=cell.row_id + shift_up + i,
                                    col_id=cell.col_id,
                                    is_header=cell.is_header
                                    and i == 0,  # Only first line is header
                                    within_row_id=cell.within_row_id,
                                    cell_id=cell_id,
                                )
                            )
                            new_cell_count += 1

                    shift_up += max_lines - 1
                else:
                    for cell in item_info["row_cells"]:
                        cell.row_id += shift_up
                        new_cells.append(cell)

            if len(new_cells) > len(table.cells):
                table.cells = new_cells

    def assign_text_to_cells(self, tables: List[TableResult], table_data: list):
        for table_result, table_page_data in zip(tables, table_data):
            table_text_lines = table_page_data["table_text_lines"]
            table_cells: List[SuryaTableCell] = table_result.cells
            text_line_bboxes = [t["bbox"] for t in table_text_lines]
            table_cell_bboxes = [c.bbox for c in table_cells]

            intersection_matrix = matrix_intersection_area(
                text_line_bboxes, table_cell_bboxes
            )

            cell_text = defaultdict(list)
            for text_line_idx, table_text_line in enumerate(table_text_lines):
                intersections = intersection_matrix[text_line_idx]
                if intersections.sum() == 0:
                    continue

                max_intersection = intersections.argmax()
                cell_text[max_intersection].append(table_text_line)

            for k in cell_text:
                # TODO: see if the text needs to be sorted (based on rotation)
                text = cell_text[k]
                assert all("text" in t for t in text), "All text lines must have text"
                assert all("bbox" in t for t in text), "All text lines must have a bbox"
                table_cells[k].text_lines = text

    def assign_pdftext_lines(self, extract_blocks: list, filepath: str):
        table_inputs = []
        unique_pages = list(set([t["page_id"] for t in extract_blocks]))
        if len(unique_pages) == 0:
            return

        for page in unique_pages:
            tables = []
            img_size = None
            for block in extract_blocks:
                if block["page_id"] == page:
                    tables.append(block["table_bbox"])
                    img_size = block["img_size"]

            table_inputs.append({"tables": tables, "img_size": img_size})
        cell_text = table_output(
            filepath,
            table_inputs,
            page_range=unique_pages,
            workers=self.pdftext_workers,
        )
        assert len(cell_text) == len(unique_pages), (
            "Number of pages and table inputs must match"
        )

        for pidx, (page_tables, pnum) in enumerate(zip(cell_text, unique_pages)):
            table_idx = 0
            for block in extract_blocks:
                if block["page_id"] == pnum:
                    table_text = page_tables[table_idx]
                    if len(table_text) == 0:
                        block["ocr_block"] = (
                            True  # Re-OCR the block if pdftext didn't find any text
                        )
                    else:
                        block["table_text_lines"] = page_tables[table_idx]
                    table_idx += 1
            assert table_idx == len(page_tables), (
                "Number of tables and table inputs must match"
            )

    def assign_ocr_lines(self, ocr_blocks: list):
        det_images = [t["table_image"] for t in ocr_blocks]
        self.recognition_model.disable_tqdm = self.disable_tqdm
        self.detection_model.disable_tqdm = self.disable_tqdm
        
        # Prepare language lists for each image (using default language)
        langs_per_image = [[self.language] for _ in det_images]
        
        ocr_results: List[OCRResult] = self.recognition_model(
            images=det_images,
            langs=langs_per_image,
            det_predictor=self.detection_model,
            recognition_batch_size=self.get_recognition_batch_size(),
            detection_batch_size=self.get_detection_batch_size(),
        )

        for block, ocr_res in zip(ocr_blocks, ocr_results):
            table_cells = []
            for line in ocr_res.text_lines:
                # Don't need to correct back to image size
                # Table rec boxes are relative to the table
                table_cells.append({"bbox": line.bbox, "text": line.text})
            block["table_text_lines"] = table_cells

    def get_detection_batch_size(self):
        if self.detection_batch_size is not None:
            return self.detection_batch_size
        elif settings.TORCH_DEVICE_MODEL == "cuda":
            return 10
        return 4

    def get_table_rec_batch_size(self):
        if self.table_rec_batch_size is not None:
            return self.table_rec_batch_size
        elif settings.TORCH_DEVICE_MODEL == "mps":
            return 6
        elif settings.TORCH_DEVICE_MODEL == "cuda":
            return 14
        return 6

    def get_recognition_batch_size(self):
        if self.recognition_batch_size is not None:
            return self.recognition_batch_size
        elif settings.TORCH_DEVICE_MODEL == "mps":
            return 32
        elif settings.TORCH_DEVICE_MODEL == "cuda":
            return 32
        return 32
