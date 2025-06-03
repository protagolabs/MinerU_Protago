#!/usr/bin/env python3
"""
Test script for custom table model
"""

import os
import sys
import shutil
from PIL import Image
from loguru import logger

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set up configuration before importing MinerU modules
def setup_config():
    """Set up configuration for testing"""
    home_config = os.path.expanduser("~/magic-pdf.json")
    if os.path.exists(home_config):
        logger.info(f"✅ Using existing config: {home_config}")
        return True
    else:
        logger.error(f"❌ Configuration file {home_config} not found!")
        logger.error("Please make sure you have a valid magic-pdf.json in your home directory")
        return False

# Set up config before importing MinerU
if not setup_config():
    sys.exit(1)

from magic_pdf.model.sub_modules.model_init import AtomModelSingleton


def test_custom_table_model():
    """Test the custom table model with a sample image"""
    
    # Initialize model manager
    logger.info("Initializing AtomModelSingleton...")
    atom_model_manager = AtomModelSingleton()
    
    try:
        # Get custom table model
        logger.info("Loading custom table model...")
        table_model = atom_model_manager.get_atom_model(
            atom_model_name='table',
            table_model_name='custom_table',
            table_model_path='',  # No specific model path needed for this example
            table_max_time=400,
            device='cpu',
            lang='ch'  # Use Chinese OCR
        )
        
        logger.info("Custom table model loaded successfully!")
        
        # Print model info
        if hasattr(table_model, 'get_model_info'):
            model_info = table_model.get_model_info()
            logger.info(f"Model info: {model_info}")
        
        # Test with a sample image (create a simple test image if no real table image is available)
        test_image_path = "test_table_image.jpg"
        
        if not os.path.exists(test_image_path):
            # Create a simple test image with some text
            logger.info("Creating a simple test image...")
            create_test_table_image(test_image_path)
        
        # Load and process the image
        logger.info(f"Processing image: {test_image_path}")
        img = Image.open(test_image_path)
        
        # Run prediction
        html_code, cell_bboxes, logic_points, elapsed_time = table_model.predict(img)
        
        # Display results
        logger.info(f"Processing completed in {elapsed_time:.3f} seconds")
        logger.info(f"Generated HTML:\n{html_code}")
        
        if cell_bboxes:
            logger.info(f"Cell bboxes: {len(cell_bboxes)} detected")
        
        if logic_points:
            logger.info(f"Logic points: {len(logic_points)} detected")
        
        # Save results
        output_file = "test_table_output.html"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"""
<!DOCTYPE html>
<html>
<head>
    <title>Custom Table Model Test Result</title>
    <style>
        table {{ border-collapse: collapse; margin: 20px 0; }}
        td {{ border: 1px solid #ccc; padding: 8px; }}
        .info {{ margin: 20px 0; font-family: Arial, sans-serif; }}
    </style>
</head>
<body>
    <div class="info">
        <h2>Custom Table Model Test Result</h2>
        <p><strong>Processing time:</strong> {elapsed_time:.3f} seconds</p>
        <p><strong>Image:</strong> {test_image_path}</p>
    </div>
    
    <h3>Extracted Table:</h3>
    {html_code}
</body>
</html>
""")
        
        logger.info(f"Results saved to: {output_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing custom table model: {e}")
        import traceback
        traceback.print_exc()
        return False


def create_test_table_image(output_path):
    """Create a simple test image with table-like text"""
    from PIL import Image, ImageDraw, ImageFont
    
    # Create a white image
    width, height = 600, 300
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    
    try:
        # Try to use a default font
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        # Fallback to default font
        font = ImageFont.load_default()
    
    # Draw table-like text
    table_data = [
        ["Name", "Age", "City"],
        ["Alice", "25", "New York"],
        ["Bob", "30", "London"],
        ["Charlie", "35", "Tokyo"]
    ]
    
    # Draw the table
    start_x, start_y = 50, 50
    cell_width, cell_height = 150, 40
    
    for row_idx, row in enumerate(table_data):
        for col_idx, cell_text in enumerate(row):
            x = start_x + col_idx * cell_width
            y = start_y + row_idx * cell_height
            
            # Draw cell border
            draw.rectangle([x, y, x + cell_width, y + cell_height], outline='black', width=1)
            
            # Draw text
            text_x = x + 10
            text_y = y + 10
            draw.text((text_x, text_y), cell_text, fill='black', font=font)
    
    # Save the image
    image.save(output_path)
    logger.info(f"Test image created: {output_path}")


def main():
    """Main function"""
    logger.info("Starting custom table model test...")
    
    success = test_custom_table_model()
    
    if success:
        logger.info("✅ Custom table model test completed successfully!")
        logger.info("\n🎉 Next steps:")
        logger.info("   1. Check the generated test_table_output.html file")
        logger.info("   2. Modify the custom table model for your specific needs")
        logger.info("   3. Test with real table images from your PDFs")
    else:
        logger.error("❌ Custom table model test failed!")
        sys.exit(1)


if __name__ == "__main__":
    main() 