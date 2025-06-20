import os
import time
import re
from collections import defaultdict
from copy import deepcopy
from typing import List
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
            self.detection_model = DetectionPredictor()
            self.recognition_model = RecognitionPredictor()
            self.table_rec_model = TableRecPredictor()
            
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
        start_time = time.time()
        
        # Check if models are properly initialized
        if not hasattr(self, 'models_initialized') or not self.models_initialized:
            return None, None, None, None
        
        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = PILImage.fromarray(image)
        
        # Use provided language or fall back to default
        current_language = language if language is not None else self.language
        
        try:
            table_text_lines = self._get_ocr_lines(image, current_language)
            
            # Step 2: Run table recognition
            self.table_rec_model.disable_tqdm = self.disable_tqdm
            tables: List[TableResult] = self.table_rec_model(
                [image],
                batch_size=self.get_table_rec_batch_size(),
            )
            
            if not tables or len(tables) == 0:
                return None, None, None, None
            
            table_result = tables[0]  # Take the first table
            
            # Step 3: Assign text to cells
            self._assign_text_to_cells([table_result], [{"table_text_lines": table_text_lines}])
            
            # Step 4: Apply post-processing
            self._split_combined_rows([table_result])
            self._combine_dollar_column([table_result])
             
            # Step 5: Generate HTML
            html_code = self._generate_html_from_table(table_result)
            
            # Calculate elapsed time
            elapse = time.time() - start_time
            
            return html_code, [], [], elapse
            
        except Exception as e:
            import traceback
            return None, None, None, None
    
    def _get_ocr_lines(self, image, language):
        """Extract OCR text lines from table images"""
        try:
            self.recognition_model.disable_tqdm = self.disable_tqdm
            self.detection_model.disable_tqdm = self.disable_tqdm
            
            ocr_results: List[OCRResult] = self.recognition_model(
                images=[image],
                langs=[[language]],  # List of language codes for each image
                det_predictor=self.detection_model,
                detection_batch_size=self.get_detection_batch_size(),
                recognition_batch_size=self.get_recognition_batch_size(),
            )

            
            table_cells = []
            if ocr_results and len(ocr_results) > 0:
                for line in ocr_results[0].text_lines:
                    table_cells.append({"bbox": line.bbox, "text": line.text})
            else:
                pass
            
            return table_cells
            
        except Exception as e:
            import traceback
            return []
    
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
        
        # Generate HTML
        html_parts = ["<table>"]
        
        for row_idx in range(max_row + 1):
            html_parts.append("  <tr>")
            for col_idx in range(max_col + 1):
                cell = grid[row_idx][col_idx]
                if cell is not None and cell.row_id == row_idx and cell.col_id == col_idx:
                    # This is the top-left cell of a span
                    tag = "th" if cell.is_header else "td"
                    rowspan_attr = f' rowspan="{cell.rowspan}"' if cell.rowspan > 1 else ""
                    colspan_attr = f' colspan="{cell.colspan}"' if cell.colspan > 1 else ""
                    
                    cell_text = " ".join(self._finalize_cell_text(cell))
                    cell_text = cell_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                    
                    html_parts.append(f"    <{tag}{rowspan_attr}{colspan_attr}>{cell_text}</{tag}>")
                elif cell is None:
                    # Empty cell
                    html_parts.append("    <td></td>")
            
            html_parts.append("  </tr>")
        
        html_parts.append("</table>")
        return "\n".join(html_parts)
    
    def get_detection_batch_size(self):
        if self.detection_batch_size is not None:
            return self.detection_batch_size
        return 4

    def get_table_rec_batch_size(self):
        if self.table_rec_batch_size is not None:
            return self.table_rec_batch_size
        return 6

    def get_recognition_batch_size(self):
        if self.recognition_batch_size is not None:
            return self.recognition_batch_size
        return 32
