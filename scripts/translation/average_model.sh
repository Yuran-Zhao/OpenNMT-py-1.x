BASE_PATH=../../OpenNMT-py
MODEL_PATH=$BASE_PATH/checkpoints

python ../OpenNMT-py/average_models.py \
		-m $MODEL_PATH/wmt14-de-en_step_1*0000.pt \
		-o $MODEL_PATH/wmt14-de-en_step_average.pt
