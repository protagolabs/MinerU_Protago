#!/bin/bash

# Download and setup Orbit datasets
# This script downloads v1, v2, and v3 versions of the Orbit dataset

set -e  # Exit on any error

echo "Starting dataset download and setup..."

# Install gdown if not already installed
echo "Checking for gdown..."
if ! command -v gdown &> /dev/null; then
    echo "Installing gdown..."
    pip install gdown
fi

# Download and process v1 dataset
echo "Downloading Orbit v1 dataset..."
gdown "https://drive.google.com/uc?id=1PzmTsmBIAXAcUXQHjWwY6o6T0IjKMtct"

echo "Processing v1 dataset..."
unzip export_pdf.zip -d .
mv export_pdf orbit_v1
rm -rf __MACOSX # clean up
rm export_pdf.zip

# Download and process v2 dataset
echo "Downloading Orbit v2 dataset..."
gdown "https://drive.google.com/uc?id=11qRpGk8bbQfChQ6pOFdOnUqtkTZAd_yJ"

echo "Processing v2 dataset..."
mkdir -p raw_orbit_v2
unzip pdf4.zip -d raw_orbit_v2
bash preprocess_raw_orbit_v2.sh
rm -rf raw_orbit_v2
rm pdf4.zip

# Download and process v3 dataset
echo "Downloading Orbit v3 dataset..."
gdown "https://drive.google.com/uc?id=1Uyb-ImPfH6UirS33mSHGkAyC836pwrgf"

echo "Processing v3 dataset..."
unzip raw_pdf5000.zip -d .
mv raw_pdf5000 raw_orbit_v3
bash preprocess_raw_orbit_v3.sh
rm -rf raw_orbit_v3
rm raw_pdf5000.zip

echo "All datasets downloaded and processed successfully!"

# Check for overlaps and remove duplicates (priority: v1 > v2 > v3)
echo "Checking for overlapping files between datasets..."

# Function to get PDF filenames without extension from a directory
get_pdf_names() {
    local dir="$1"
    if [ -d "$dir/pdf" ]; then
        find "$dir/pdf" -name "*.pdf" -exec basename {} .pdf \; | sort
    else
        echo ""
    fi
}

# Get PDF names from each dataset
if [ -d "orbit_v1" ]; then
    v1_files=$(get_pdf_names "orbit_v1")
    v1_count=$(echo "$v1_files" | wc -l)
    echo "Found $v1_count files in orbit_v1"
else
    v1_files=""
    v1_count=0
fi

if [ -d "orbit_v2" ]; then
    v2_files=$(get_pdf_names "orbit_v2")
    v2_count=$(echo "$v2_files" | wc -l)
    echo "Found $v2_count files in orbit_v2"
else
    v2_files=""
    v2_count=0
fi

if [ -d "orbit_v3" ]; then
    v3_files=$(get_pdf_names "orbit_v3")
    v3_count=$(echo "$v3_files" | wc -l)
    echo "Found $v3_count files in orbit_v3"
else
    v3_files=""
    v3_count=0
fi

# Function to remove overlapping files from a dataset
remove_overlaps() {
    local target_dir="$1"
    local target_files="$2"
    local reference_files="$3"
    local target_name="$4"
    local reference_name="$5"
    
    if [ -z "$target_files" ] || [ -z "$reference_files" ]; then
        return
    fi
    
    echo "Checking overlaps between $target_name and $reference_name..."
    
    # Find common files
    common_files=$(comm -12 <(echo "$target_files") <(echo "$reference_files"))
    
    if [ -n "$common_files" ]; then
        overlap_count=$(echo "$common_files" | wc -l)
        echo "Found $overlap_count overlapping files between $target_name and $reference_name"
        
        # Remove overlapping files from target dataset
        while IFS= read -r filename; do
            if [ -n "$filename" ]; then
                echo "Removing $filename from $target_name..."
                
                # Remove from all subdirectories
                find "$target_dir" -name "$filename.*" -delete 2>/dev/null || true
            fi
        done <<< "$common_files"
        
        echo "Removed $overlap_count overlapping files from $target_name"
    else
        echo "No overlaps found between $target_name and $reference_name"
    fi
}

# Remove overlaps with priority v1 > v2 > v3
# First, remove v3 files that overlap with v1 or v2
if [ -n "$v3_files" ]; then
    if [ -n "$v1_files" ]; then
        remove_overlaps "orbit_v3" "$v3_files" "$v1_files" "orbit_v3" "orbit_v1"
        # Update v3_files list after removal
        v3_files=$(get_pdf_names "orbit_v3")
    fi
    if [ -n "$v2_files" ] && [ -n "$v3_files" ]; then
        remove_overlaps "orbit_v3" "$v3_files" "$v2_files" "orbit_v3" "orbit_v2"
    fi
fi

# Then, remove v2 files that overlap with v1
if [ -n "$v2_files" ] && [ -n "$v1_files" ]; then
    remove_overlaps "orbit_v2" "$v2_files" "$v1_files" "orbit_v2" "orbit_v1"
fi

# Final count after deduplication
echo ""
echo "Final dataset statistics after deduplication:"
if [ -d "orbit_v1" ]; then
    final_v1_count=$(get_pdf_names "orbit_v1" | wc -l)
    echo "- orbit_v1/: $final_v1_count files"
fi
if [ -d "orbit_v2" ]; then
    final_v2_count=$(get_pdf_names "orbit_v2" | wc -l)
    echo "- orbit_v2/: $final_v2_count files"
fi
if [ -d "orbit_v3" ]; then
    final_v3_count=$(get_pdf_names "orbit_v3" | wc -l)
    echo "- orbit_v3/: $final_v3_count files"
fi

echo ""
echo "Dataset setup completed successfully!"


# 