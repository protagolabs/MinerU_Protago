# python comparison_tables.py \
#     --mineru-tables comparison/orbit_v1_marker_outputs_tables \
#     --azure-tables comparison/orbit_v1_azure_outputs_tables \
#     --output-dir comparison/comparison_orbit_v1_marker_azure_original \
#     --processes 8

# python comparison_tables.py \
#     --mineru-tables comparison/orbit_v1_mineru133_outputs_tables \
#     --azure-tables comparison/orbit_v1_azure_outputs_tables \
#     --output-dir comparison/comparison_orbit_v1_mineru133_azure_original \
#     --processes 8

# python comparison_refined_tables.py \
#     --mineru-tables comparison/orbit_v1_mineru133_outputs_refined_tables \
#     --azure-tables comparison/orbit_v1_azure_outputs_tables \
#     --output-dir comparison/comparison_orbit_v1_refined_mineru133_azure \
#     --processes 8

python comparison_refined_tables_md.py \
    --mineru-tables comparison/orbit_v1_mineru133_outputs_refined_tables_md \
    --azure-tables comparison/orbit_v1_azure_outputs_tables \
    --output-dir comparison/comparison_orbit_v1_refined_mineru133_azure_md \
    --processes 6

# python comparison_refined_tables_md.py \
#     --mineru-tables comparison/orbit_v1_mineru133_outputs_refined_tables_md \
#     --azure-tables comparison/orbit_v1_azure_outputs_tables \
#     --output-dir comparison/comparison_orbit_v1_refined_mineru133_azure_md \
#     --processes 1

# python comparison_tables_grits.py \
#     --mineru-tables comparison/orbit_v1_mineru133_outputs_tables \
#     --azure-tables comparison/orbit_v1_azure_outputs_tables \
#     --output-dir comparison/comparison_orbit_v1_mineru133_azure_grits \
#     --processes 8

########################################## compare markdown files ##########################################

# python comparison_markdown.py \
#     --gt_path ./comparison/orbit_v1_azure_outputs_md/f_7Gw6LrbnfntIX6MSaqRXvW.md \
#     --method_path ./comparison/orbit_v1_marker_outputs_md/f_7Gw6LrbnfntIX6MSaqRXvW.md 


# python comparison_markdown.py \
#     --gt_path ./comparison/orbit_v1_azure_outputs_md/f_7Gw6LrbnfntIX6MSaqRXvW.md \
#     --method_path ./comparison/orbit_v1_mineru133_outputs_md/f_7Gw6LrbnfntIX6MSaqRXvW.md 


# python comparison_markdown.py \
#     --gt_path ./comparison/orbit_v1_azure_outputs_md/f_7Gw6LrbnfntIX6MSaqRXvW.md \
#     --method_path ./comparison/orbit_v1_mineru133_outputs_md_converted/f_7Gw6LrbnfntIX6MSaqRXvW.md 



# python comparison_markdown.py \
#     --gt_path ./comparison/orbit_v1_azure_outputs_md \
#     --method_path ./comparison/orbit_v1_mineru133_outputs_md_converted \
#     --output_file ./comparison/azure_mineru133_md_comparison \
#     --max_workers 8


# python comparison_markdown.py \
#     --gt_path ./comparison/orbit_v1_azure_outputs_md \
#     --method_path ./comparison/orbit_v1_marker_outputs_md \
#     --output_file ./comparison/azure_marker_md_comparison \
#     --max_workers 8