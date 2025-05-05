python table_comparison_azure_vs_mineru_v2.py \
    --mineru-tables comparison/orbit_v1_mineru133_outputs_tables \
    --azure-tables comparison/orbit_v1_azure_outputs_tables \
    --output-dir comparison/comparison_results_ted_v2 \
    --processes 8

# python table_comparison_vlm_refined.py \
#     --mineru-tables comparison/orbit_v1_mineru133_outputs_tables_olmocr \
#     --azure-tables comparison/orbit_v1_azure_outputs_tables \
#     --output-dir comparison/comparison_results \
#     --processes 8


# python table_comparison_vlm_refined_md.py \
#     --mineru-file comparison/orbit_v1_mineru133_outputs_tables_olmocr_md/f_0AibR1dz.tables.json \
#     --azure-file comparison/orbit_v1_azure_outputs_tables/f_0AibR1dz.pages.tables.json \
#     --output-dir comparison/comparison_results \
    
# python table_comparison_vlm_refined_md.py \
#     --mineru-tables comparison/orbit_v1_mineru133_outputs_tables_olmocr_md \
#     --azure-tables comparison/orbit_v1_azure_outputs_tables \
#     --output-dir comparison/comparison_results \
#     --processes 8