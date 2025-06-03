#!/usr/bin/env python3
"""
Test the custom table model through MinerU CLI
"""

import os
import sys
import shutil
import tempfile
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def create_sample_pdf_with_table():
    """Create a sample PDF with a table for testing"""
    try:
        # Try to import reportlab for PDF creation
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
        from reportlab.lib import colors
        
        # Create sample data
        data = [
            ['Product', 'Price', 'Quantity', 'Total'],
            ['Laptop', '$999', '2', '$1998'],
            ['Mouse', '$25', '5', '$125'],
            ['Keyboard', '$75', '3', '$225'],
            ['Monitor', '$299', '1', '$299'],
            ['Total', '', '', '$2647']
        ]
        
        # Create PDF
        filename = "sample_table.pdf"
        doc = SimpleDocTemplate(filename, pagesize=letter)
        
        # Create table
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        # Build PDF
        elements = [table]
        doc.build(elements)
        
        print(f"✅ Created sample PDF: {filename}")
        return filename
        
    except ImportError:
        print("⚠️  reportlab not available, creating image-based PDF instead...")
        return create_image_based_pdf()

def create_image_based_pdf():
    """Create a PDF from a table image"""
    import fitz  # PyMuPDF
    
    # Create table image
    img_path = "temp_table.png"
    create_table_image(img_path)
    
    # Convert image to PDF
    pdf_path = "sample_table.pdf"
    img_doc = fitz.open(img_path)
    pdf_bytes = img_doc.convert_to_pdf()
    
    with open(pdf_path, 'wb') as f:
        f.write(pdf_bytes)
    
    # Clean up
    os.remove(img_path)
    img_doc.close()
    
    print(f"✅ Created image-based PDF: {pdf_path}")
    return pdf_path

def create_table_image(output_path):
    """Create a table image"""
    # Create image
    width, height = 800, 400
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except:
        font = ImageFont.load_default()
    
    # Table data
    data = [
        ['Product', 'Price', 'Quantity', 'Total'],
        ['Laptop', '$999', '2', '$1998'],
        ['Mouse', '$25', '5', '$125'],
        ['Keyboard', '$75', '3', '$225'],
        ['Monitor', '$299', '1', '$299'],
        ['Total', '', '', '$2647']
    ]
    
    # Draw table
    start_x, start_y = 50, 50
    cell_width, cell_height = 150, 50
    
    for row_idx, row in enumerate(data):
        for col_idx, cell_text in enumerate(row):
            x = start_x + col_idx * cell_width
            y = start_y + row_idx * cell_height
            
            # Header row background
            if row_idx == 0:
                draw.rectangle([x, y, x + cell_width, y + cell_height], 
                             fill='lightgray', outline='black', width=2)
            else:
                draw.rectangle([x, y, x + cell_width, y + cell_height], 
                             outline='black', width=1)
            
            # Text
            text_x = x + 10
            text_y = y + 15
            draw.text((text_x, text_y), cell_text, fill='black', font=font)
    
    image.save(output_path)

def test_custom_table_with_cli():
    """Test custom table model using CLI"""
    
    print("🧪 Testing Custom Table Model with MinerU CLI")
    print("=" * 50)
    
    # 1. Create sample PDF
    print("\n📄 Step 1: Creating sample PDF with table...")
    pdf_file = create_sample_pdf_with_table()
    
    if not os.path.exists(pdf_file):
        print("❌ Failed to create sample PDF")
        return False
    
    # 2. Create output directory
    output_dir = "test_output_custom_table"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    # 3. Backup and temporarily modify configuration
    home_config = os.path.expanduser("~/magic-pdf.json")
    backup_config = None
    
    # Backup existing config if it exists
    if os.path.exists(home_config):
        backup_config = home_config + ".backup"
        shutil.copy2(home_config, backup_config)
        print(f"📋 Backed up existing config to: {backup_config}")
        
        # Read existing config and modify only the table section
        import json
        with open(home_config, 'r') as f:
            existing_config = json.load(f)
        
        # Update only the table config for testing
        existing_config["table-config"] = {
            "model": "custom_table",
            "enable": True,
            "max_time": 400,
            "confidence_threshold": 0.5,
            "custom_model_path": ""
        }
        
        # Write back the modified config
        with open(home_config, 'w') as f:
            json.dump(existing_config, f, indent=4)
        print(f"✅ Temporarily modified table config for testing")
    else:
        # Copy custom config if no existing config
        config_file = "magic-pdf-custom.json"
        if os.path.exists(config_file):
            shutil.copy2(config_file, home_config)
            print(f"✅ Copied custom config to: {home_config}")
        else:
            print("⚠️  Custom config file not found, using default configuration")
    
    # 4. Run MinerU CLI with custom table model
    print(f"\n🚀 Step 2: Running MinerU CLI...")
    print(f"Input PDF: {pdf_file}")
    print(f"Output directory: {output_dir}")
    
    # Build CLI command
    cli_command = f"magic-pdf -p {pdf_file} -o {output_dir} -m auto"
    print(f"Command: {cli_command}")
    
    # Execute command
    result = os.system(cli_command)
    
    # 5. Check results
    print(f"\n📊 Step 3: Checking results...")
    
    if result == 0:
        print("✅ CLI command executed successfully!")
        
        # Look for output files
        output_files = []
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if file.endswith(('.md', '.html', '.json')):
                    output_files.append(os.path.join(root, file))
        
        print(f"📁 Output files found: {len(output_files)}")
        for file in output_files:
            print(f"   - {file}")
            
        # Check if markdown contains table HTML
        markdown_files = [f for f in output_files if f.endswith('.md')]
        if markdown_files:
            with open(markdown_files[0], 'r', encoding='utf-8') as f:
                content = f.read()
                if '<table' in content:
                    print("✅ Table HTML found in markdown output!")
                    print("🎉 Custom table model is working!")
                else:
                    print("⚠️  No table HTML found in markdown output")
        
        return True
    else:
        print(f"❌ CLI command failed with return code: {result}")
        return False

def cleanup(restore_config=True):
    """Clean up test files and optionally restore config"""
    files_to_remove = [
        "sample_table.pdf",
        "test_table_image.jpg",
        "test_table_output.html"
    ]
    
    dirs_to_remove = [
        "test_output_custom_table"
    ]
    
    for file in files_to_remove:
        if os.path.exists(file):
            os.remove(file)
            print(f"🗑️  Removed: {file}")
    
    for dir in dirs_to_remove:
        if os.path.exists(dir):
            shutil.rmtree(dir)
            print(f"🗑️  Removed directory: {dir}")
    
    # Restore original config if backup exists
    if restore_config:
        home_config = os.path.expanduser("~/magic-pdf.json")
        backup_config = home_config + ".backup"
        
        if os.path.exists(backup_config):
            shutil.copy2(backup_config, home_config)
            os.remove(backup_config)
            print(f"🔄 Restored original config from backup")
        else:
            print("ℹ️  No backup config found to restore")

def main():
    """Main function"""
    print("🔧 MinerU Custom Table Model CLI Test")
    print("=" * 40)
    
    success = False
    try:
        success = test_custom_table_with_cli()
        
        if success:
            print("\n🎉 SUCCESS: Custom table model CLI test completed!")
            print("\n💡 Next steps:")
            print("   1. Check the output files in test_output_custom_table/")
            print("   2. Modify the custom table model in magic_pdf/model/sub_modules/table/custom_table/")
            print("   3. Adjust confidence_threshold and other parameters in magic-pdf-custom.json")
            print("   4. Test with your own PDF files")
        else:
            print("\n❌ FAILED: Custom table model CLI test failed!")
            print("\n🔍 Troubleshooting:")
            print("   1. Make sure MinerU is properly installed")
            print("   2. Check if all dependencies are available")
            print("   3. Verify the custom model files are in the correct location")
            print("   4. Check the configuration file")
        
        # Ask about cleanup
        cleanup_response = input("\n🗑️  Clean up test files? (y/n): ").strip().lower()
        if cleanup_response in ['y', 'yes']:
            cleanup(restore_config=True)
        else:
            # Always restore config even if user doesn't want to clean up files
            cleanup(restore_config=True)
            print("🔄 Config restored (keeping test files)")
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrupted by user")
        # Always restore config on interruption
        cleanup(restore_config=True)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        # Always restore config on error
        cleanup(restore_config=True)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 