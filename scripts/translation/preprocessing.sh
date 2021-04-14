DATA_PATH=/home/zyr/Projects/Graduation-Project/OpenNMT-py/data/wmt14-de-en

src_train=$DATA_PATH/train.de.sp
tgt_train=$DATA_PATH/train.en.sp
src_valid=$DATA_PATH/valid.de.sp
tgt_valid=$DATA_PATH/valid.en.sp

target_dir=$DATA_PATH/onmt

src_vocab=$DATA_PATH/deen.vocab
tgt_vocab=$DATA_PATH/deen.vocab

python ../OpenNMT-py/preprocess.py \
		-train_src $src_train -train_tgt $tgt_train \
		-valid_src $src_valid -valid_tgt $tgt_valid \
		-save_data $target_dir \
		-src_vocab_size 32000 -tgt_vocab_size 32000 \
		-share_vocab -overwrite -lower
		# -src_vocab $src_vocab -tgt_vocab $tgt_vocab

		
