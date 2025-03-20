

# python dataset/orbit_data_v1/split_train_val.py

# # dataset check
python main.py -c paddlex/configs/modules/table_structure_recognition/SLANet_plus.yaml \
    -o Global.mode=check_dataset \
    -o Global.dataset_dir=./dataset/orbit_data_v1


# python main.py -c paddlex/configs/modules/table_structure_recognition/SLANet_plus.yaml \
#     -o Global.mode=evaluate \
#     -o Global.dataset_dir=./dataset/orbit_data_v1 \
#     -o Global.device=gpu:0 \
#     -o Global.output=./output/orbit_data_v1_slanet_plus \
    
# # # train
python main.py -c paddlex/configs/modules/table_structure_recognition/SLANet_plus.yaml \
    -o Global.mode=train \
    -o Global.dataset_dir=./dataset/orbit_data_v1 \
    -o Global.device=gpu:0,1 \
    -o Global.output=./output/orbit_data_v1_slanet_plus \
    -o Train.epoch_num=100 \
    -o Train.learning_rate=0.0001 \
    -o Train.pretrain_weight_path="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/SLANet_plus_pretrained.pdparams" \
    -o CheckDataset.enable=False \
    -o CheckDataset.split.enable=False 
    # -o CheckDataset.split.enable=True \
    # -o CheckDataset.split.train_percent=0.8 \
    # -o CheckDataset.split.val_percent=0.2 \


# # eval
python main.py -c addlex/configs/modules/table_structure_recognition/SLANet_plus.yaml \
    -o Global.mode=predict \
    -o Predict.model_dir="./output/orbit_data_v1_slanet_plus/best_accuracy/inference" \
    -o Predict.input="./dataset/orbit_data_v1/images/f_4EJcfSgW_table_0_page_2.png" \
    -o Global.output="./output/orbit_data_v1_slanet_plus/predict"


# paddle2onnx, need to change the conda environment to paddle2onnx

# paddle2onnx --model_dir ./output/best_accuracy/inference \
#             --model_filename inference.pdmodel \
#             --params_filename inference.pdiparams \
#             --save_file ./output/best_accuracy.onnx 



# python main.py -c paddlex/configs/modules/table_structure_recognition/SLANet_plus.yaml \
#     -o Global.mode=check_dataset \
#     -o Global.dataset_dir=./dataset/table_rec_dataset_examples

# python main.py -c paddlex/configs/modules/table_structure_recognition/SLANet_plus.yaml \
#     -o Global.mode=train \
#     -o Global.dataset_dir=./dataset/table_rec_dataset_examples \
#     -o Global.device=gpu:0,1 \
#     -o Global.output=./output/table_rec_dataset_examples_slanet_plus

