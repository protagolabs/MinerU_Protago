# python comparison_tables.py \
#     --mineru-tables comparison/orbit_v1_marker_outputs_tables \
#     --azure-tables comparison/orbit_v1_azure_outputs_tables \
#     --output-dir comparison/comparison_orbit_v1_marker_azure \
#     --processes 8


python comparison_markdown.py \
    comparison/orbit_v1_azure_outputs_markdowns \
    comparison/orbit_v1_maker_outputs_markdowns \
    comparison/azure_marker_md_comparison \
    8