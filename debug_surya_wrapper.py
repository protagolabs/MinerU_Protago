#!/usr/bin/env python3
"""
Debug script for SuryaTableWrapper to identify issues
"""

import os
import sys
import logging
from PIL import Image
import numpy as np

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_surya_imports():
    """Test if Surya modules can be imported"""
    try:
        from surya.detection import DetectionPredictor
        logger.info("✓ DetectionPredictor imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import DetectionPredictor: {e}")
        return False
    
    try:
        from surya.recognition import RecognitionPredictor, OCRResult
        logger.info("✓ RecognitionPredictor imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import RecognitionPredictor: {e}")
        return False
    
    try:
        from surya.table_rec import TableRecPredictor
        logger.info("✓ TableRecPredictor imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import TableRecPredictor: {e}")
        return False
    
    try:
        from surya.table_rec.schema import TableResult, TableCell as SuryaTableCell
        logger.info("✓ TableResult and TableCell imported successfully")
    except Exception as e:
        logger.error(f"✗ Failed to import TableResult/TableCell: {e}")
        return False
    
    return True

def test_model_initialization():
    """Test if Surya models can be initialized"""
    try:
        from surya.detection import DetectionPredictor
        from surya.recognition import RecognitionPredictor
        from surya.table_rec import TableRecPredictor
        
        logger.info("Testing model initialization...")
        
        detection_model = DetectionPredictor()
        logger.info("✓ DetectionPredictor initialized")
        
        recognition_model = RecognitionPredictor()
        logger.info("✓ RecognitionPredictor initialized")
        
        table_rec_model = TableRecPredictor()
        logger.info("✓ TableRecPredictor initialized")
        
        return True
    except Exception as e:
        logger.error(f"✗ Model initialization failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def test_wrapper_initialization():
    """Test if SuryaTableWrapper can be initialized"""
    try:
        from magic_pdf.model.sub_modules.table.surya_table.surya_table_wrapper import SuryaTableWrapper
        
        logger.info("Testing SuryaTableWrapper initialization...")
        
        wrapper = SuryaTableWrapper()
        logger.info("✓ SuryaTableWrapper initialized successfully")
        
        return wrapper
    except Exception as e:
        logger.error(f"✗ SuryaTableWrapper initialization failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

def create_test_image():
    """Create a simple test image with some text"""
    # Create a simple image with text
    img = Image.new('RGB', (400, 200), color='white')
    
    # Add some simple text-like content (this is just for testing)
    from PIL import ImageDraw, ImageFont
    
    draw = ImageDraw.Draw(img)
    
    # Try to use a default font
    try:
        font = ImageFont.load_default()
    except:
        font = None
    
    # Draw some text
    draw.text((50, 50), "Test Table", fill='black', font=font)
    draw.text((50, 80), "Column 1 | Column 2", fill='black', font=font)
    draw.text((50, 110), "Data 1   | Data 2", fill='black', font=font)
    
    return img

def test_prediction(wrapper):
    """Test the prediction method"""
    try:
        logger.info("Testing prediction with test image...")
        
        # Create a test image
        test_image = create_test_image()
        logger.info(f"Created test image: {test_image.size}")
        
        # Test prediction
        result = wrapper.predict(test_image, language='en')
        logger.info(f"Prediction result: {result}")
        
        if result[0] is not None:
            logger.info("✓ Prediction successful")
            logger.info(f"HTML output: {result[0][:200]}...")  # Show first 200 chars
        else:
            logger.warning("⚠ Prediction returned None")
        
        return result
    except Exception as e:
        logger.error(f"✗ Prediction test failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

def main():
    """Main debug function"""
    logger.info("=== SuryaTableWrapper Debug Session ===")
    
    # Test 1: Import tests
    logger.info("\n1. Testing imports...")
    if not test_surya_imports():
        logger.error("Import tests failed. Please check Surya installation.")
        return
    
    # Test 2: Model initialization
    logger.info("\n2. Testing model initialization...")
    if not test_model_initialization():
        logger.error("Model initialization failed.")
        return
    
    # Test 3: Wrapper initialization
    logger.info("\n3. Testing wrapper initialization...")
    wrapper = test_wrapper_initialization()
    if wrapper is None:
        logger.error("Wrapper initialization failed.")
        return
    
    # Test 4: Prediction
    logger.info("\n4. Testing prediction...")
    test_prediction(wrapper)
    
    logger.info("\n=== Debug session completed ===")

if __name__ == "__main__":
    main() 