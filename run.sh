# # magic-pdf -p demo/pdfs/small_ocr.pdf -o output/ -m auto

# # magic-pdf -p /home/xing/MinerU_Protago/datasets/orbit_v1/pdf/  -o output/mineru1310 -m ocr 2>&1 | tee logs/$(date +%Y%m%d_%H%M%S)_magic_pdf.log

# # Record start time
# start_time=$(date +%s)

# # magic-pdf -p demo/pdfs/small_ocr.pdf -o output/ -m auto
# # Run the command
# magic-pdf -p /home/xing/MinerU_Protago/demo/pdfs/f_0AibR1dz_page_9.pdf  -o output/mineru1310_marker162 -m auto 2>&1 | tee logs/$(date +%Y%m%d_%H%M%S)_magic_pdf.log

# # Calculate and display execution time
# end_time=$(date +%s)
# execution_time=$((end_time - start_time))
# echo "Execution time: $execution_time seconds" | tee -a logs/$(date +%Y%m%d_%H%M%S)_magic_pdf.log



magic-pdf -p ./demo/pdfs/demo6.pdf  -o output -m auto 2>&1 | tee logs/$(date +%Y%m%d_%H%M%S)_magic_pdf.log
# magic-pdf -p ./demo/pdfs/f_0AibR1dz_page_9.pdf  -o output -m auto 2>&1 | tee logs/$(date +%Y%m%d_%H%M%S)_magic_pdf.log
