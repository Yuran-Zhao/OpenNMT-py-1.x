sl=de
tl=en
vocab_size=32000

DATA_PATH=$1
tmp_train=$DATA_PATH/tmp.train.txt

rm -rf $tmp_train
for f in $DATA_PATH/train.$sl $DATA_PATH/train.$tl
do
		cat $f >> $tmp_train
done

spm_train --input=$tmp_train --model_prefix=$DATA_PATH/wmt$sl$tl \
		--vocab_size=$vocab_size --character_coverage=1 --model_type=bpe

rm -rf $tmp_train
