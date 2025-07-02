import re
import time
from collections import defaultdict
from copy import deepcopy
from typing import Annotated, List, Union, Tuple, Optional, Dict, Any
from collections import Counter
import json

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
            
        self.detection_model = DetectionPredictor()
        self.recognition_model = RecognitionPredictor()
        self.table_rec_model = TableRecPredictor()
        
        self.table_rec_model.disable_tqdm = self.disable_tqdm
        self.recognition_model.disable_tqdm = self.disable_tqdm
        self.detection_model.disable_tqdm = self.disable_tqdm


    def predict(self, image, language=None):
        """
        Predict table structure from image with optimized single-image processing
        Falls back to batch processing for consistency and efficiency
        """
        # Use batch processing for single image to leverage optimizations
        self.language = language
        results = self.predict_batch([image], [self.language])
        
        if results and len(results) > 0:
            return results[0]
        else:
            return None, [], [], 0.0

    def predict_batch(self, images: List[Union[np.ndarray]], 
                     languages: Optional[List[str]] = None) -> List[Tuple]:
        """
        Optimized batch prediction following Marker's high-performance approach
        Key optimizations:
        1. Efficient batch processing with optimal GPU utilization
        2. Parallel OCR and table structure recognition
        3. Memory-efficient processing
        """
        if not images:
            return []
        
        start_time = time.time()
        
        # Collect all table data first (like Marker does)
        table_data = []

        # Store table data with metadata
        for j, image in enumerate(images):
            # Resize image to have at least one dimension as 2048 while preserving aspect ratio
            pil_image = PILImage.fromarray(image)
            original_width, original_height = pil_image.size
            aspect_ratio = original_width / original_height
            
            if aspect_ratio >= 1:  # Width >= Height
                new_width = 2048
                new_height = int(2048 / aspect_ratio)
            else:  # Height > Width
                new_height = 2048
                new_width = int(2048 * aspect_ratio)
            
            table_data.append({
                'table_image': pil_image.resize((new_width, new_height)),
                'result_idx': j,
                'language': languages[j],
                "ocr_block": False,
            })

        # Process OCR in batches for tables that need it
        ocr_blocks = [t for t in table_data if not t["ocr_block"]]
        table_data = self.assign_ocr_lines(ocr_blocks)  # Handle tables where OCR is needed
        
        for table_item in table_data:
            # print(table_item)
            if "table_text_lines" not in table_item:
                logger.warning(
                    f"No text lines found for table {table_item['block_id']}"
                )
                table_item["table_text_lines"] = []

        tables: List[TableResult] = self.table_rec_model(
            [t["table_image"] for t in table_data],
            batch_size=self.get_table_rec_batch_size(),
        )
        # get the cell coordinates, tables[0] example:
        # TableCell(polygon=[[0.6611328125, 0.63134765625], [207.760986328125, 0.63134765625], [207.760986328125, 72.56552124023438], [0.6611328125, 72.56552124023438]], 
        # confidence=None, row_id=0, colspan=1, within_row_id=0, cell_id=0, is_header=True, rowspan=1, merge_up=False, merge_down=False, col_id=0, text_lines=None, 
        # bbox=[0.6611328125, 0.63134765625, 207.760986328125, 72.56552124023438]

        tables = self.assign_text_to_cells(tables, table_data)
        tables = self.split_combined_rows(tables)  # Split up rows that were combined
        tables = self.combine_dollar_column(tables)  # Combine columns that are just dollar signs

        # logger.debug(f"one table cells' example: {tables[0].cells}")

        result = []
        for tb in tables:
            cells: List[SuryaTableCell] = tb.cells
            for i, cell in enumerate(cells):

                cell_block = TableCell(
                        polygon=PolygonBox(polygon=cell.polygon),
                        text_lines=self.finalize_cell_text(cell),
                        rowspan=cell.rowspan,
                        colspan=cell.colspan,
                        row_id=cell.row_id,
                        col_id=cell.col_id,
                        is_header=bool(cell.is_header),
                    )
                cells[i] = cell_block
            # logger.debug(f"one table cells' example: {cells}")
            result.append((self.render_html(cells), None, None, None))
            
        return result




        # return [(self.render_html(tb.cells), None, None, None) for tb in tables] 
        # TODO: return the tables
        # logger.debug(f"Batch processing completed for {len(tables)} tables")
        # # Assign table cells to the table
        # table_idx = 0
        # for page in document.pages:
        #     for block in page.contained_blocks(document, self.block_types):
        #         block.structure = []  # Remove any existing lines, spans, etc.
        #         cells: List[SuryaTableCell] = tables[table_idx].cells
        #         for cell in cells:
        #             # Rescale the cell polygon to the page size
        #             cell_polygon = PolygonBox(polygon=cell.polygon).rescale(
        #                 page.get_image(highres=True).size, page.polygon.size
        #             )

        #             # Rescale cell polygon to be relative to the page instead of the table
        #             for corner in cell_polygon.polygon:
        #                 corner[0] += block.polygon.bbox[0]
        #                 corner[1] += block.polygon.bbox[1]

        #             cell_block = TableCell(
        #                 polygon=cell_polygon,
        #                 text_lines=self.finalize_cell_text(cell),
        #                 rowspan=cell.rowspan,
        #                 colspan=cell.colspan,
        #                 row_id=cell.row_id,
        #                 col_id=cell.col_id,
        #                 is_header=bool(cell.is_header),
        #                 page_id=page.page_id,
        #             )
        #             page.add_full_block(cell_block)
        #             block.add_structure(cell_block)
        #         table_idx += 1


        total_time = time.time() - start_time
        logger.debug(f"Batch processing completed in {total_time:.3f}s for {len(tables)} tables")
        
        # print(tables[0])
        # return all_results
        return None
        

    def assign_ocr_lines(self, ocr_blocks: list):
        det_images = [t["table_image"] for t in ocr_blocks]
        det_languages = [t["language"] for t in ocr_blocks]

        ocr_results: List[OCRResult] = self.recognition_model(
            images=det_images,
            langs=det_languages,
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
        
        return ocr_blocks


        
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
        return tables
    
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
                    len(c.text_lines) if isinstance(c.text_lines, list) else 1
                    for c in row_cells
                ]

                # Other cells that span into this row
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
                        all([line_len == line_lens[0] for line_len in line_lens]),
                    ]
                )
                line_lens_counter = Counter(line_lens)
                counter_keys = sorted(list(line_lens_counter.keys()))
                should_split_partial_row = all(
                    [
                        len(row_cells) > 3,  # Only split if there are more than 3 cells
                        len(rowspan_cells) == 0,
                        all([r == 1 for r in rowspans]),
                        len(line_lens_counter) == 2
                        and counter_keys[0] <= 1
                        and counter_keys[1] > 1
                        and line_lens_counter[counter_keys[0]]
                        == 1,  # Allow a single column with a single line - keys are the line lens, values are the counts
                    ]
                )
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

                    # For each new row we add, shift up subsequent rows
                    # The max is to account for partial rows
                    shift_up += max_lines - 1
                else:
                    for cell in item_info["row_cells"]:
                        cell.row_id += shift_up
                        new_cells.append(cell)

            # Only update the cells if we added new cells
            if len(new_cells) > len(table.cells):
                table.cells = new_cells
        
        return tables

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
        
        return tables

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

    def render_html(self, table_cells) -> str:
        """
        Render table cells to HTML format
        
        Args:
            table_cells: List of TableCell or SuryaTableCell objects
            
        Returns:
            HTML table string
        """
        if not table_cells:
            return "<table></table>"
        
        # Organize cells by position
        cell_matrix = {}
        for cell in table_cells:
            cell_matrix[(cell.row_id, cell.col_id)] = cell
        
        # Extract and clean text data from cells
        cell_data = {}
        for cell in table_cells:
            text_content = self._extract_cell_text(cell)
            cell_data[(cell.row_id, cell.col_id)] = text_content
        
        # Get table dimensions
        max_row = max(pos[0] for pos in cell_matrix.keys())
        max_col = max(pos[1] for pos in cell_matrix.keys())
        
        html_parts = ["<table>"]
        
        # Track cells that are part of spans to skip them
        skip_cells = set()
        
        for row_id in range(max_row + 1):
            html_parts.append("  <tr>")
            
            for col_id in range(max_col + 1):
                if (row_id, col_id) in skip_cells:
                    continue
                    
                if (row_id, col_id) in cell_matrix:
                    cell = cell_matrix[(row_id, col_id)]
                    text = cell_data.get((row_id, col_id), "")
                    
                    # Determine cell type
                    tag = "th" if cell.is_header else "td"
                    
                    # Build cell attributes
                    attrs = []
                    if cell.rowspan > 1:
                        attrs.append(f'rowspan="{cell.rowspan}"')
                        # Mark spanned cells to skip
                        for r in range(1, cell.rowspan):
                            skip_cells.add((row_id + r, col_id))
                    
                    if cell.colspan > 1:
                        attrs.append(f'colspan="{cell.colspan}"')
                        # Mark spanned cells to skip
                        for c in range(1, cell.colspan):
                            skip_cells.add((row_id, col_id + c))
                            # Also mark the intersection of rowspan and colspan
                            if cell.rowspan > 1:
                                for r in range(1, cell.rowspan):
                                    skip_cells.add((row_id + r, col_id + c))
                    
                    attr_str = " " + " ".join(attrs) if attrs else ""
                    
                    # Escape HTML special characters
                    escaped_text = (text.replace("&", "&amp;")
                                         .replace("<", "&lt;")
                                         .replace(">", "&gt;")
                                         .replace('"', "&quot;"))
                    
                    html_parts.append(f"    <{tag}{attr_str}>{escaped_text}</{tag}>")
                else:
                    # Empty cell
                    html_parts.append("    <td></td>")
            
            html_parts.append("  </tr>")
        
        html_parts.append("</table>")
        return "\n".join(html_parts)
    
    def _extract_cell_text(self, cell) -> str:
        """Extract and clean text from a single cell (handles both TableCell and SuryaTableCell)"""
        if not hasattr(cell, 'text_lines') or not cell.text_lines:
            return ""
        
        text_parts = []
        
        # Handle TableCell objects (text_lines is a list of strings)
        if isinstance(cell.text_lines, list) and len(cell.text_lines) > 0:
            if isinstance(cell.text_lines[0], str):
                # TableCell format: text_lines is a list of strings
                for line in cell.text_lines:
                    text = line.strip()
                    if text and text != ".":
                        # Apply text cleaning
                        text = re.sub(r"(\s\.){2,}", "", text)  # Replace . . .
                        text = re.sub(r"\.{2,}", "", text)  # Replace ..., like in table of contents
                        text = self.normalize_spaces(fix_text(text))
                        text_parts.append(text)
            else:
                # SuryaTableCell format: text_lines is a list of dicts with 'text' key
                for line in cell.text_lines:
                    if isinstance(line, dict) and 'text' in line:
                        text = line['text'].strip()
                        if text and text != ".":
                            # Apply text cleaning
                            text = re.sub(r"(\s\.){2,}", "", text)  # Replace . . .
                            text = re.sub(r"\.{2,}", "", text)  # Replace ..., like in table of contents
                            text = self.normalize_spaces(fix_text(text))
                            text_parts.append(text)
        
        return " ".join(text_parts)