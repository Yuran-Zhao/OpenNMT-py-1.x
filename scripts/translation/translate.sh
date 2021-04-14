# onmt_translate -model /home/zyr/Projects/Graduation-Project/models/style-model_step_200000.pt -src ./test.txt -output ./pred.txt -replace_unk -verbose
BASE_PATH=../../OpenNMT-py
DATA_PATH=$BASE_PATH/data
MODEL_PATH=$BASE_PATH/checkpoints

for checkpoint in $MODEL_PATH/wmt14-de-en_step*.pt; do
		echo "# Translating with checkpoint $checkpoint"
		base=$(basename $checkpoint)
		suffix=${base##*_}
		tgt=$DATA_PATH/wmt14-de-en/test.en.hyp_${suffix%.*}
		if [ ! -f "$tgt" ];then
				python ../OpenNMT-py/translate.py \
						-model $checkpoint \
						-src $DATA_PATH/wmt14-de-en/test.de.sp \
						-output $tgt.sp \
						-replace_unk -verbose
		fi
done

for checkpoint in $MODEL_PATH/wmt14-de-en_step*.pt; do
		base=$(basename $checkpoint)
		suffix=${base##*_}
		tgt=$DATA_PATH/wmt14-de-en/test.en.hyp_${suffix%.*}
		if [ ! -f "$tgt" ];then
				spm_decode \
						-model=$DATA_PATH/wmt14-de-en/wmtdeen.model \
						-input_format=piece \
						< $tgt.sp > $tgt
		fi
done

for checkpoint in $MODEL_PATH/wmt14-de-en_step*.pt; do
		echo "$checkpoint"
		base=$(basename $checkpoint)
		suffix=${base##*_}
		tgt=$DATA_PATH/wmt14-de-en/test.en.hyp_${suffix%.*}
		sacrebleu $DATA_PATH/wmt14-de-en/test.en < $tgt
done

