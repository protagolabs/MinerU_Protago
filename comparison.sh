python table_comparison_azure_vs_mineru.py \
    --mineru-tables comparison/orbit_v1_mineru133_outputs_tables \
    --azure-tables comparison/orbit_v1_azure_outputs_tables \
    --output-dir comparison/comparison_results \
    --processes 8

python table_comparison_vlm_refined.py \
    --mineru-tables comparison/mineru133_outputs_tables \
    --azure-tables comparison/orbit_v1_azure_outputs_tables \
    --output-dir comparison/comparison_results \
    --processes 8
