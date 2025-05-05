#!/bin/bash

src_dir="raw_orbit_v2"
dest_dir="orbit_v2"

mkdir -p "$dest_dir"  # create destination if it doesn't exist
mkdir -p "$dest_dir/azure_pkl"
mkdir -p "$dest_dir/azure_blocks"
mkdir -p "$dest_dir/azure_pages"
mkdir -p "$dest_dir/pdf"

for subdir in "$src_dir"/*/; do
    # echo "$subdir"
    full_dirname=$(basename "$subdir")
    base_name="${full_dirname%.pdf}"  # strip the .pdf extension
    # echo "$base_name"
    # Copy and rename each file
    cp "$subdir/azure_reg.pkl" "$dest_dir/azure_pkl/${base_name}.pkl"
    cp "$subdir/blocks.txt" "$dest_dir/azure_blocks/${base_name}.blocks.txt"
    cp "$subdir/pages.txt" "$dest_dir/azure_pages/${base_name}.pages.txt"
    # Handle PDF files correctly - don't add .pdf if it already has .PDF
    if [[ "$full_dirname" == *.PDF ]]; then
        cp "$subdir/$full_dirname" "$dest_dir/pdf/$full_dirname"
    else
        # Handle special characters in filenames
        pdf_file=$(find "$subdir" -maxdepth 1 -type f -name "*.pdf" -o -name "*.PDF" | head -n 1)
        if [ -n "$pdf_file" ]; then
            cp "$pdf_file" "$dest_dir/pdf/$(basename "$pdf_file")"
        fi
    fi
done
