# visualize azure pdf pages
# python scripts/azure_convert.py \
#     --pdf_dir inputs/export_pdf/pdf \
#     --pkl_dir inputs/export_pdf/azure_pkl \
#     --output_dir inputs/export_pdf/visualizations \
#     --num_processes 2

# extract tables from azure blocks
# python scripts/convert_azure_tables_from_blocks.py \
#     --input inputs/export_pdf/azure_blocks \
#     --output inputs/export_pdf/tables_from_blocks 

# python scripts/convert_azure_tables_from_pages.py \
#     --input inputs/export_pdf/azure_pages \
#     --output inputs/export_pdf/tables_from_pages 

# # # single file extract tables from pkl
# python scripts/convert_azure_tables_from_pkl.py \
#     --pkl-file inputs/export_pdf/azure_pkl/f_QGrLZ33G.pkl \
#     --pdf-file inputs/export_pdf/pdf/f_QGrLZ33G.pdf \
#     --output inputs/export_pdf/tables_from_pkl/

# # batch extract tables from pkl
python scripts/convert_azure_tables_from_pkl.py \
    --input inputs/export_pdf/azure_pkl \
    --pdf-dir inputs/export_pdf/pdf \
    --output inputs/export_pdf/tables_from_pkl

# conver the extracted tables into the orbit dataset, and skip those tables with issues
python scripts/prepare_orbit_dataset.py \
    --input_dir inputs/export_pdf/tables_from_pkl \
    --output_dir outputs/orbit_data \
    --combine






# python scripts/compare_azure_tables_from_blocks_and_pages.py \
#     -m1 inputs/export_pdf/tables_from_blocks \
#     -m2 inputs/export_pdf/tables_from_pages \
#     --output results/extracted_tables_from_blocks_and_pages

# python scripts/convert_azure_tables_from_blocks_with_images.py \
#     --pdf_dir inputs/export_pdf/pdf \
#     --json_dir inputs/export_pdf/tables_from_blocks \
#     --output inputs/export_pdf/tables_from_blocks_with_images \
#     --single \
#     --pdf inputs/export_pdf/pdf/f_0k7zJjVH7M5zjAH8tgbpU6.pdf \
#     --json inputs/export_pdf/tables_from_blocks/f_0k7zJjVH7M5zjAH8tgbpU6.blocks.tables.json

# python scripts/convert_azure_tables_from_blocks_with_images_pd_ft.py \
#     --pdf_dir inputs/export_pdf/pdf \
#     --json_dir inputs/export_pdf/tables_from_blocks \
#     --output inputs/export_pdf/tables_from_blocks_with_images_pd_ft 

# python scripts/convert_azure_tables_from_pages.py \
#     --input inputs/export_pdf/azure_pages \
#     --output inputs/export_pdf/tables_from_pages 



# python azure_convert.py \
#     --pdf_dir inputs/export_pdf/pdf \
#     --pkl_dir inputs/export_pdf/azure_pkl \
#     --output_dir inputs/export_pdf/visualizations \
#     --num_processes 2


# python table_comparison_mj_v2.py \
#     --mineru-tables ./azure_outputs_tables \
#     --azure-tables ./inputs/export_pdf/azure_tables \
#     --output-dir ./table_comparison_results \
#     --processes 8


# k=100


# # Extract top k matches
# rm -r table_comparison_results/top${k}_samples
# mkdir -p table_comparison_results/top${k}_samples

# python extract_top_similarity.py table_comparison_results/comparison_results_20250306_145634.json \
#     --top ${k} \
#     --output "table_comparison_results/top${k}_samples/top${k}.json"

