# magic-pdf --version

# too large pdf file, will make the progress killed by the system
# magic-pdf -p inputs/export_pdf/pdf -o ./output133_orbit_default -m auto 



# Input and output directories
INPUT_DIR="inputs/export_pdf1000/pdf"
OUTPUT_DIR="./output133_orbit1000_default"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Process each PDF file individually
for pdf_file in "$INPUT_DIR"/*.pdf; do
    # Get just the filename without path
    filename=$(basename "$pdf_file")
    
    echo "Processing $filename..."

    
    # Process the individual PDF file
    magic-pdf -p "$pdf_file" -o "$OUTPUT_DIR" -m ocr
    
    echo "Completed processing $filename"
    echo "------------------------"
done

echo "All PDF files have been processed."




