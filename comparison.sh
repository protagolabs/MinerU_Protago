# Use --table-threads to control how many tables are compared simultaneously within each file.
# Use --processes to control how many files are compared simultaneously. Default: Number of CPU cores
python table_comparison_v4.py \
    --mineru-tables comparison/orbit_v1_mineru133_outputs_tables \
    --azure-tables comparison/orbit_v1_azure_outputs_tables \
    --output-dir comparison/comparison_orbit_v1_mineru133_azure \
    --processes 8 --table-threads 16

# python table_comparison_mj_v3.py --mineru-tables comparison/mineru122_outputs_tables --azure-tables comparison/orbit100_azure_outputs_tables --output-dir comparison/comparison_results --processes 8


# python table_comparison_mj_v3.py --mineru-tables comparison/mineru133_orbit1000_outputs_tables --azure-tables comparison/orbit1000_azure_outputs_tables --output-dir comparison/comparison_results --processes 8


# python table_comparison_mj_v3.py --mineru-tables comparison/mineru122_orbit1000_outputs_tables --azure-tables comparison/orbit1000_azure_outputs_tables --output-dir comparison/comparison_results --processes 8


# python table_comparison_vlm_refined.py --mineru-tables comparison/mineru133_outputs_tables --azure-tables comparison/orbit100_azure_outputs_tables --output-dir comparison/comparison_results --processes 4

