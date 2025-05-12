CUDA_VISIBLE_DEVICES=0 magic-pdf -p inputs/orbit_v1/pdf -o outputs/orbit_v1_mineru133_outputs -m ocr  >> logs/orbit_v1_mineru133_outputs.log 2>&1

# magic-pdf -p inputs/orbit_v2/pdf -o outputs/orbit_v2_mineru133_outputs -m ocr  >> logs/orbit_v2_mineru133_outputs.log 2>&1

# VIRTUAL_VRAM_SIZE=8 magic-pdf -p inputs/orbit_v2/pdf -o outputs/orbit_v2_mineru133_outputs -m ocr