python comparison_tables.py \
    --mineru-tables comparison/orbit_v1_marker_outputs_tables \
    --azure-tables comparison/orbit_v1_azure_outputs_tables \
    --output-dir comparison/comparison_orbit_v1_marker_azure_original \
    --processes 8

python comparison_tables.py \
    --mineru-tables comparison/orbit_v1_mineru133_outputs_tables \
    --azure-tables comparison/orbit_v1_azure_outputs_tables \
    --output-dir comparison/comparison_orbit_v1_mineru133_azure_original \
    --processes 8

# python comparison_markdown.py \
#     --gt_path ./comparison/orbit_v1_azure_outputs_markdowns/f_7Gw6LrbnfntIX6MSaqRXvW.md \
#     --method_path ./comparison/orbit_v1_maker_outputs_markdowns/f_7Gw6LrbnfntIX6MSaqRXvW.md 

# python comparison_markdown.py \
#     --gt_path ./comparison/orbit_v1_azure_outputs_markdowns \
#     --method_path ./comparison/orbit_v1_maker_outputs_markdowns \
#     --output_file ./comparison/azure_marker_md_comparison \
#     --max_workers 12


# python comparison_markdown.py \
#     --gt_path ./comparison/orbit_v1_azure_outputs_markdowns \
#     --method_path ./comparison/orbit_v1_mineru133_outputs_markdowns \
#     --output_file ./comparison/azure_mineru133_md_comparison \
#     --max_workers 12