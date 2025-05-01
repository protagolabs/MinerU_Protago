# python table_comparison_mj_v3.py --mineru-tables comparison/mineru133_outputs_tables --azure-tables comparison/orbit100_azure_outputs_tables --output-dir comparison/comparison_results --processes 8


# python table_comparison_mj_v3.py --mineru-tables comparison/mineru122_outputs_tables --azure-tables comparison/orbit100_azure_outputs_tables --output-dir comparison/comparison_results --processes 8


# python table_comparison_mj_v3.py --mineru-tables comparison/mineru133_orbit1000_outputs_tables --azure-tables comparison/orbit1000_azure_outputs_tables --output-dir comparison/comparison_results --processes 8


# python table_comparison_mj_v3.py --mineru-tables comparison/mineru122_orbit1000_outputs_tables --azure-tables comparison/orbit1000_azure_outputs_tables --output-dir comparison/comparison_results --processes 8


# python table_comparison_vlm_refined.py \
#     --mineru-tables mineru133_outputs_tables_olmocr \
#     --azure-tables orbit100_azure_outputs_tables \
#     --output-dir comparison/comparison_results \
#     --processes 8

# python table_comparison_vlm_refined_md.py \
#     --mineru-file mineru133_outputs_tables_olmocr/f_0ifCZuJ4mHzEKMbr9frAA3.tables.refined_md.json \
#     --azure-file orbit100_azure_outputs_tables/f_0ifCZuJ4mHzEKMbr9frAA3.pages.tables.json \
#     --output-dir comparison/single_file_comparison_results 


python table_comparison_vlm_refined_md.py \
    --mineru-tables mineru133_outputs_tables_olmocr \
    --azure-tables orbit_v1_azure_outputs_tables \
    --output-dir comparison/comparison_results 
