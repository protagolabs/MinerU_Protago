#!/bin/bash

src_dir="raw_orbit_v3"
dest_dir="orbit_v3"

mkdir -p "$dest_dir"  # create destination if it doesn't exist
mkdir -p "$dest_dir/azure_pkl"
mkdir -p "$dest_dir/pdf"

for subdir in "$src_dir"/*/; do
    full_dirname=$(basename "$subdir")
    base_name="${full_dirname%.pdf}"  # strip the .pdf extension

    # echo "$subdir"
    # echo "$base_name"
    # Check if the azure_reg.pkl file exists before copying
    if [[ -f "$subdir/azure_reg.pkl" ]]; then
        cp "$subdir/azure_reg.pkl" "$dest_dir/azure_pkl/${base_name}.pkl"  # Changed destination to azure_pkl
    else
        echo "Warning: $subdir/azure_reg.pkl does not exist."
    fi

    # Handle PDF files correctly - check for both .pdf and .PDF
    pdf_file=$(find "$subdir" -maxdepth 1 -type f \( -name "*.pdf" -o -name "*.PDF" \) | head -n 1)
    if [ -n "$pdf_file" ]; then
        cp "$pdf_file" "$dest_dir/pdf/${base_name}.pdf"  # Ensure PDF is copied to pdf directory
    else
        echo "Warning: No PDF file found in $subdir."
    fi
done

# Check if each .pkl file has a corresponding .pdf file
for pkl_file in "$dest_dir/azure_pkl/"*.pkl; do
    base_name=$(basename "$pkl_file" .pkl)
    corresponding_pdf="$dest_dir/pdf/$base_name.pdf"

    if [[ ! -f "$corresponding_pdf" && ! -f "$corresponding_pdf_upper" ]]; then
        echo "Warning: Corresponding PDF for $pkl_file does not exist: $corresponding_pdf or $corresponding_pdf_upper"
    fi
done