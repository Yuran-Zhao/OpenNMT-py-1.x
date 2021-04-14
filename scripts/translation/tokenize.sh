DATA_PATH=$1
sl=de
tl=en

for f in $DATA_PATH/train.$sl $DATA_PATH/train.$tl $DATA_PATH/valid.$sl $DATA_PATH/valid.$tl
do
		file=$(basename $f)
		echo "# Tokenizing $file"
		spm_encode --model=$DATA_PATH/wmt$sl$tl.model --output_format=piece < $f > $DATA_PATH/$file.sp
done

