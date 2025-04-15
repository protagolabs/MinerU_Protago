# magic-pdf --version

# too large pdf file, will make the progress killed by the system
# magic-pdf -p inputs/export_pdf/pdf -o ./output133_orbit_default -m auto 



# Input and output directories
INPUT_DIR="inputs/export_pdf/pdf"
OUTPUT_DIR="./output133_orbit_default"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Process each PDF file individually
for pdf_file in "$INPUT_DIR"/*.pdf; do
    # Get just the filename without path
    filename=$(basename "$pdf_file")
    
    echo "Processing $filename..."
    
    # Create a subdirectory for each PDF's output
    file_output_dir="$OUTPUT_DIR/$(basename "$filename" .pdf)"
    mkdir -p "$file_output_dir"
    
    # Process the individual PDF file
    magic-pdf -p "$pdf_file" -o "$file_output_dir" -m auto
    
    echo "Completed processing $filename"
    echo "------------------------"
done

echo "All PDF files have been processed."
