python scripts_orbit/azure_table_converter.py \
    -i inputs/orbit_v1/azure_pages \
    -o comparison/orbit_v1_azure_outputs_tables

python scripts_orbit/mineru_table_converter.py \
    -i outputs/orbit_v1_mineru133_outputs \
    -o comparison/orbit_v1_mineru133_outputs_tables

# If you want, we also add the table extraction from the marker outputs

python scripts_orbit/marker_table_converter.py \
    -i outputs/orbit_v1_marker_outputs \
    -o comparison/orbit_v1_marker_outputs_tables







