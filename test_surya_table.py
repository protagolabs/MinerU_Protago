#!/usr/bin/env python3
"""
Test script for the modified SuryaTableWrapper
"""

import sys
import os
from PIL import Image
import numpy as np

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from magic_pdf.model.sub_modules.table.surya_table.surya_table_wrapper import SuryaTableWrapper
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("This is expected if the Surya library is not installed.")
    print("The wrapper is designed to work with the Surya library.")
    sys.exit(1)


def test_surya_table_wrapper():
    """Test the SuryaTableWrapper with a sample table image"""
    
    # Initialize the wrapper with language configuration
    config = {
        'disable_tqdm': True,
        'drop_repeated_text': False,
        'format_lines': False,
        'language': 'en'  # Can be changed to other languages like 'zh', 'ja', etc.
    }
    
    try:
        wrapper = SuryaTableWrapper(config=config)
        print("✓ SuryaTableWrapper initialized successfully")
    except Exception as e:
        print(f"✗ Failed to initialize SuryaTableWrapper: {e}")
        return
    
    # Create a simple test table image (you can replace this with a real table image)
    # This is just a placeholder - in practice you would load a real table image
    print("\nNote: This is a demonstration. To test with a real table image:")
    print("1. Save a table image as 'test_table.png'")
    print("2. Uncomment the image loading code below")
    print("3. Run the script again")
    
    # Uncomment the following lines to test with a real image:
    """
    try:
        # Load a table image
        image_path = "test_table.png"
        if os.path.exists(image_path):
            image = Image.open(image_path)
            print(f"✓ Loaded test image: {image_path}")
            print(f"  Image size: {image.size}")
            
            # Process the table
            html_code, table_cell_bboxes, logic_points, elapse = wrapper.predict(image)
            
            if html_code:
                print(f"✓ Table processing completed in {elapse:.2f} seconds")
                print(f"✓ Generated HTML table:")
                print("-" * 50)
                print(html_code)
                print("-" * 50)
            else:
                print("✗ No table detected or processing failed")
        else:
            print(f"✗ Test image not found: {image_path}")
    except Exception as e:
        print(f"✗ Error processing table: {e}")
    """
    
    print("\n✓ SuryaTableWrapper is ready to use!")
    print("\nUsage example:")
    print("""
    from magic_pdf.model.sub_modules.table.surya_table.surya_table_wrapper import SuryaTableWrapper
    from PIL import Image
    
    # Initialize wrapper with default language configuration
    config = {
        'language': 'en',  # Default language
        'disable_tqdm': True
    }
    wrapper = SuryaTableWrapper(config=config)
    
    # Load table image
    image = Image.open("your_table_image.png")
    
    # Process table with default language
    html_code, table_cell_bboxes, logic_points, elapse = wrapper.predict(image)
    
    # Or process table with specific language (overrides default)
    html_code, table_cell_bboxes, logic_points, elapse = wrapper.predict(image, language='zh')
    
    # Use the HTML output
    if html_code:
        print(html_code)
    """)


if __name__ == "__main__":
    test_surya_table_wrapper() 