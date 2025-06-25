import os
import time
import re
from collections import defaultdict
from copy import deepcopy
from typing import List, Union, Tuple, Optional
from collections import Counter

from PIL import Image as PILImage
import numpy as np
from loguru import logger
from ftfy import fix_text

from surya.detection import DetectionPredictor
from surya.recognition import RecognitionPredictor, OCRResult
from surya.table_rec import TableRecPredictor
from surya.table_rec.schema import TableResult, TableCell as SuryaTableCell


class SuryaTableWrapper:
    """
    Wrapper class for Surya's table recognition that provides a predict interface
    compatible with MinerU's table model interface.
    """
    
    def __init__(self, config=None):
        """
        Initialize SuryaTableWrapper
        
        Args:
            config: Configuration dictionary for Surya models
                   If None, uses default config
        """
        try:
            # Initialize all required models
            logger.info("Initializing Surya models...")
            self.detection_model = DetectionPredictor()
            self.recognition_model = RecognitionPredictor()
            self.table_rec_model = TableRecPredictor()
            
            # Check device availability
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                logger.info(f"Surya models initialized on device: {device}")
            except ImportError:
                device = "cpu"
                logger.info("PyTorch not available, using CPU")
            
            # Set configuration parameters - handle None config properly
            if config is None:
                config = {}
            
            self.disable_tqdm = config.get('disable_tqdm', True)
            self.drop_repeated_text = config.get('drop_repeated_text', False)
            self.format_lines = config.get('format_lines', False)
            self.language = config.get('language', 'en')  # Default to English
            
            # Set batch sizes
            self.detection_batch_size = config.get('detection_batch_size', None)
            self.recognition_batch_size = config.get('recognition_batch_size', None)
            self.table_rec_batch_size = config.get('table_rec_batch_size', None)
            
            # Log batch size configuration and GPU info
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_name = torch.cuda.get_device_name(0)
                    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    logger.info(f"GPU: {gpu_name} ({gpu_memory_gb:.1f}GB)")
            except ImportError:
                pass
                
            logger.info(f"Batch sizes - Detection: {self.get_detection_batch_size()}, "
                       f"Recognition: {self.get_recognition_batch_size()}, "
                       f"Table Recognition: {self.get_table_rec_batch_size()}")
            
            # Flag to track if models are properly initialized
            self.models_initialized = True
        except Exception as e:
            # Set flag to indicate models are not available
            self.models_initialized = False
            self.detection_model = None
            self.recognition_model = None
            self.table_rec_model = None
            
            # Still set configuration for potential fallback
            if config is None:
                config = {}
            self.disable_tqdm = config.get('disable_tqdm', True)
            self.language = config.get('language', 'en')
            
            raise
        
    def predict(self, image, language=None):
        """
        Predict table structure from image
        
        Args:
            image: PIL Image or numpy array
            language: Optional language code to override default language (e.g., 'en', 'zh', 'ja')
            
        Returns:
            tuple: (html_code, table_cell_bboxes, logic_points, elapse)
                  Following the same interface as RapidTable
        """
        # Use batch predict for single image for consistent behavior
        results = self.predict_batch([image], [language] if language else None)
        if results and len(results) > 0:
            return results[0]
        return None, None, None, None
    
    def predict_batch(self, images: List[Union[PILImage.Image, np.ndarray]], 
                     languages: Optional[List[str]] = None) -> List[Tuple]:
        """
        Predict table structure from multiple images in batch
        
        Args:
            images: List of PIL Images or numpy arrays
            languages: Optional list of language codes for each image (e.g., ['en', 'zh', 'ja'])
                      If None, uses default language for all images
                      If provided, must have same length as images list
            
        Returns:
            List of tuples: Each tuple contains (html_code, table_cell_bboxes, logic_points, elapse)
                           Following the same interface as RapidTable
        """
        start_time = time.time()
        
        # Check if models are properly initialized
        if not hasattr(self, 'models_initialized') or not self.models_initialized:
            return [(None, None, None, None) for _ in images]
        
        if not images:
            return []
        
        # Convert numpy arrays to PIL Images if needed
        processed_images = []
        for image in images:
            if isinstance(image, np.ndarray):
                processed_images.append(PILImage.fromarray(image))
            else:
                processed_images.append(image)
        
        # Handle languages parameter
        if languages is None:
            languages = [self.language] * len(processed_images)
        elif len(languages) != len(processed_images):
            logger.warning(f"Languages list length ({len(languages)}) doesn't match images list length ({len(processed_images)}). Using default language.")
            languages = [self.language] * len(processed_images)
        else:
            # Replace any None values with the default language
            languages = [lang if lang is not None else self.language for lang in languages]
        
        try:
            # Option 1: Sequential processing (current approach)
            # Step 1: Get OCR lines for all images
            ocr_start = time.time()
            
            
            all_table_text_lines = self._get_ocr_lines_batch(processed_images, languages)
            ocr_time = time.time() - ocr_start
            # Only show timing summaries at INFO level instead of detailed DEBUG
            # logger.debug(f"OCR processing took {ocr_time:.2f}s for {len(processed_images)} images")
            
            # Step 2: Run table recognition on all images
            table_rec_start = time.time()
            self.table_rec_model.disable_tqdm = self.disable_tqdm
            # Use adaptive batch size - at least as large as the input batch
            adaptive_table_rec_batch_size = max(self.get_table_rec_batch_size(), len(processed_images))
            tables: List[TableResult] = self.table_rec_model(
                processed_images,
                batch_size=adaptive_table_rec_batch_size,
            )
            table_rec_time = time.time() - table_rec_start
            # logger.debug(f"Table recognition took {table_rec_time:.2f}s for {len(processed_images)} images (batch_size: {adaptive_table_rec_batch_size})")
            
            # TODO: For future optimization, we could try parallel processing:
            # - OCR and table recognition could potentially run in parallel on different GPU streams
            # - This would require careful GPU memory management
            
            # If no tables found, return None results
            if not tables:
                return [(None, None, None, None) for _ in processed_images]
            
            # Step 3: Assign text to cells for all tables
            assign_start = time.time()
            table_data = [{"table_text_lines": text_lines} for text_lines in all_table_text_lines]
            self._assign_text_to_cells(tables, table_data)
            assign_time = time.time() - assign_start
            # logger.debug(f"Text assignment took {assign_time:.2f}s for {len(processed_images)} images")
            
            # Step 4: Apply post-processing to all tables
            postproc_start = time.time()
            self._split_combined_rows(tables)
            self._combine_dollar_column(tables)
            postproc_time = time.time() - postproc_start
            # logger.debug(f"Post-processing took {postproc_time:.2f}s for {len(processed_images)} images")
             
            # Step 5: Generate HTML for all tables
            html_start = time.time()
            # Pre-allocate results list for better memory efficiency
            results = [None] * len(processed_images)
            batch_elapse = time.time() - start_time
            
            for i, table_result in enumerate(tables):
                if table_result is not None:
                    html_code = self._generate_html_from_table(table_result)
                    # For batch processing, we divide the total time equally among all images
                    # Individual timing would require processing images separately
                    elapse = batch_elapse / len(processed_images)
                    results[i] = (html_code, [], [], elapse)
                else:
                    results[i] = (None, None, None, None)
            
            html_time = time.time() - html_start
            # logger.debug(f"HTML generation took {html_time:.2f}s for {len(processed_images)} images")
            # logger.info(f"Total batch processing: {batch_elapse:.2f}s for {len(processed_images)} images ({batch_elapse/len(processed_images):.3f}s per image)")
            
            # Ensure we return the same number of results as input images
            while len(results) < len(processed_images):
                results.append((None, None, None, None))
            
            return results[:len(processed_images)]
            
        except Exception as e:
            import traceback
            logger.error(f"Error in batch prediction: {str(e)}\n{traceback.format_exc()}")
            return [(None, None, None, None) for _ in processed_images]
    
    def _get_ocr_lines_batch(self, images: List[PILImage.Image], languages: List[str]) -> List[List[dict]]:
        """Extract OCR text lines from multiple table images in batch"""
        try:
            self.recognition_model.disable_tqdm = self.disable_tqdm
            self.detection_model.disable_tqdm = self.disable_tqdm
            
            # Ensure no None values in languages list
            safe_languages = [lang if lang is not None else self.language for lang in languages]
            
            # Prepare language lists for each image
            langs_per_image = [[lang] for lang in safe_languages]
            
            # Use adaptive batch sizes - at least as large as the input batch
            adaptive_detection_batch_size = max(self.get_detection_batch_size(), len(images))
            adaptive_recognition_batch_size = max(self.get_recognition_batch_size(), len(images))
            
            ocr_results: List[OCRResult] = self.recognition_model(
                images=images,
                langs=langs_per_image,  # List of language lists for each image
                det_predictor=self.detection_model,
                detection_batch_size=adaptive_detection_batch_size,
                recognition_batch_size=adaptive_recognition_batch_size,
            )
            
            # logger.debug(f"OCR used batch sizes - Detection: {adaptive_detection_batch_size}, Recognition: {adaptive_recognition_batch_size}")
            
            all_table_cells = []
            for ocr_result in ocr_results:
                table_cells = []
                if ocr_result and ocr_result.text_lines:
                    for line in ocr_result.text_lines:
                        table_cells.append({"bbox": line.bbox, "text": line.text})
                all_table_cells.append(table_cells)
            
            return all_table_cells
            
        except Exception as e:
            import traceback
            logger.error(f"Error in batch OCR: {str(e)}\n{traceback.format_exc()}")
            return [[] for _ in images]

    def _get_ocr_lines(self, image, language):
        """Extract OCR text lines from table images"""
        # Use batch method for single image
        result = self._get_ocr_lines_batch([image], [language])
        return result[0] if result else []
    
    def _assign_text_to_cells(self, tables: List[TableResult], table_data: list):
        """Assign OCR text to table cells based on intersection"""
        for table_result, table_page_data in zip(tables, table_data):
            table_text_lines = table_page_data["table_text_lines"]
            table_cells: List[SuryaTableCell] = table_result.cells
            text_line_bboxes = [t["bbox"] for t in table_text_lines]
            table_cell_bboxes = [c.bbox for c in table_cells]

            # Simple intersection calculation
            intersection_matrix = self._calculate_intersection_matrix(
                text_line_bboxes, table_cell_bboxes
            )

            cell_text = defaultdict(list)
            for text_line_idx, table_text_line in enumerate(table_text_lines):
                intersections = intersection_matrix[text_line_idx]
                if sum(intersections) == 0:
                    continue

                max_intersection = intersections.index(max(intersections))
                cell_text[max_intersection].append(table_text_line)

            for k in cell_text:
                text = cell_text[k]
                assert all("text" in t for t in text), "All text lines must have text"
                assert all("bbox" in t for t in text), "All text lines must have a bbox"
                table_cells[k].text_lines = text
    
    def _calculate_intersection_matrix(self, text_bboxes, cell_bboxes):
        """Calculate intersection areas between text and cell bounding boxes"""
        matrix = []
        for text_bbox in text_bboxes:
            row = []
            for cell_bbox in cell_bboxes:
                intersection = self._calculate_bbox_intersection(text_bbox, cell_bbox)
                row.append(intersection)
            matrix.append(row)
        return matrix
    
    def _calculate_bbox_intersection(self, bbox1, bbox2):
        """Calculate intersection area between two bounding boxes"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0
        
        return (x2 - x1) * (y2 - y1)
    
    def _split_combined_rows(self, tables: List[TableResult]):
        """Split rows that were combined during recognition"""
        for table in tables:
            if len(table.cells) == 0:
                continue
                
            unique_rows = sorted(list(set([c.row_id for c in table.cells])))
            row_info = []
            
            for row in unique_rows:
                row_cells = deepcopy([c for c in table.cells if c.row_id == row])
                rowspans = [c.rowspan for c in row_cells]
                line_lens = [
                    len(c.text_lines) if isinstance(c.text_lines, list) else 1
                    for c in row_cells
                ]

                rowspan_cells = [
                    c
                    for c in table.cells
                    if c.row_id != row and c.row_id + c.rowspan > row > c.row_id
                ]
                
                should_split_entire_row = all([
                    len(row_cells) > 1,
                    len(rowspan_cells) == 0,
                    all([rowspan == 1 for rowspan in rowspans]),
                    all([line_len > 1 for line_len in line_lens]),
                    all([line_len == line_lens[0] for line_len in line_lens]),
                ])
                
                line_lens_counter = Counter(line_lens)
                counter_keys = sorted(list(line_lens_counter.keys()))
                should_split_partial_row = all([
                    len(row_cells) > 3,
                    len(rowspan_cells) == 0,
                    all([r == 1 for r in rowspans]),
                    len(line_lens_counter) == 2
                    and counter_keys[0] <= 1
                    and counter_keys[1] > 1
                    and line_lens_counter[counter_keys[0]] == 1,
                ])
                
                should_split = should_split_entire_row or should_split_partial_row
                row_info.append({
                    "should_split": should_split,
                    "row_cells": row_cells,
                    "line_lens": line_lens,
                })

            # Don't split if we're not splitting most of the rows
            if sum([r["should_split"] for r in row_info]) / len(row_info) < 0.5:
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
                                    is_header=cell.is_header and i == 0,
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
    
    def _combine_dollar_column(self, tables: List[TableResult]):
        """Combine columns that are just dollar signs"""
        for table in tables:
            if len(table.cells) == 0:
                continue
                
            unique_cols = sorted(list(set([c.col_id for c in table.cells])))
            max_col = max(unique_cols)
            dollar_cols = []
            
            for col in unique_cols:
                col_cells = [c for c in table.cells if c.col_id == col]
                col_text = [
                    "\n".join(self._finalize_cell_text(c)).strip() for c in col_cells
                ]
                all_dollars = all([ct in ["", "$"] for ct in col_text])
                colspans = [c.colspan for c in col_cells]
                span_into_col = [
                    c
                    for c in table.cells
                    if c.col_id != col and c.col_id + c.colspan > col > c.col_id
                ]

                if all([
                    all_dollars,
                    len(col_cells) > 1,
                    len(span_into_col) == 0,
                    all([c == 1 for c in colspans]),
                    col < max_col,
                ]):
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
                        ]
                        next_row_col[0].col_id -= col_offset
                else:
                    for cell in col_cells:
                        cell.col_id -= col_offset
    
    def _finalize_cell_text(self, cell: SuryaTableCell):
        """Clean and normalize cell text"""
        fixed_text = []
        text_lines = cell.text_lines if cell.text_lines else []
        for line in text_lines:
            text = line["text"].strip()
            if not text or text == ".":
                continue
            text = re.sub(r"(\s\.){2,}", "", text)
            text = re.sub(r"\.{2,}", "", text)
            text = self._normalize_spaces(fix_text(text))
            fixed_text.append(text)
        return fixed_text

    @staticmethod
    def _normalize_spaces(text):
        """Normalize various space characters"""
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
    
    def _generate_html_from_table(self, table: TableResult):
        """Generate HTML from table result"""
        if not table.cells:
            return "<table></table>"
        
        # Get table dimensions
        max_row = max([c.row_id for c in table.cells])
        max_col = max([c.col_id for c in table.cells])
        
        # Create 2D grid
        grid = [[None for _ in range(max_col + 1)] for _ in range(max_row + 1)]
        
        # Fill grid with cells
        for cell in table.cells:
            for r in range(cell.row_id, cell.row_id + cell.rowspan):
                for c in range(cell.col_id, cell.col_id + cell.colspan):
                    if r <= max_row and c <= max_col:
                        grid[r][c] = cell
        
        # Generate HTML (compact format to match expected output)
        html_parts = ["<table><tbody>"]
        
        for row_idx in range(max_row + 1):
            html_parts.append("<tr>")
            for col_idx in range(max_col + 1):
                cell = grid[row_idx][col_idx]
                if cell is not None and cell.row_id == row_idx and cell.col_id == col_idx:
                    # This is the top-left cell of a span
                    tag = "th" if cell.is_header else "td"
                    rowspan_attr = f' rowspan={cell.rowspan}' if cell.rowspan > 1 else ""
                    colspan_attr = f' colspan={cell.colspan}' if cell.colspan > 1 else ""
                    
                    cell_text = " ".join(self._finalize_cell_text(cell))
                    cell_text = cell_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                    
                    html_parts.append(f"<{tag}{rowspan_attr}{colspan_attr}>{cell_text}</{tag}>")
                elif cell is None:
                    # Empty cell
                    html_parts.append("<td></td>")
            
            html_parts.append("</tr>")
        
        html_parts.append("</tbody></table>")
        return "".join(html_parts)
    
    def get_detection_batch_size(self):
        if self.detection_batch_size is not None:
            return self.detection_batch_size
        return 10

    def get_table_rec_batch_size(self):
        if self.table_rec_batch_size is not None:
            return self.table_rec_batch_size
        return 14

    def get_recognition_batch_size(self):
        if self.recognition_batch_size is not None:
            return self.recognition_batch_size
        return 32
    
    
    def _should_skip_ocr(self, image):
        """
        Heuristic to determine if OCR can be skipped for simple tables
        This is a placeholder for future optimization
        """
        # For now, always do OCR, but this could be optimized in the future
        # by checking if the table has very regular structure with minimal text
        return False
