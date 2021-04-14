
# Translation Examples

## 1.Reproduce WMT14-de-en Result in [Attention is All You Need](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)

### 1) Prepare the data

I refer to the [script](https://github.com/OpenNMT/OpenNMT-py/blob/master/examples/scripts/prepare_wmt_data.sh) provided by OpenNMT-py. The key steps of downloading dataset is as follows:

```shell
echo "Downloading and extracting Commoncrawl data (919 MB) for training..."
wget --trust-server-names http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz
tar zxvf training-parallel-commoncrawl.tgz
ls | grep -v 'commoncrawl.de-en.[de,en]' | xargs rm

echo "Downloading and extracting Europarl data (658 MB) for training..."
wget --trust-server-names http://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz
tar zxvf training-parallel-europarl-v7.tgz
cd training && ls | grep -v 'europarl-v7.de-en.[de,en]' | xargs rm
cd .. && mv training/europarl* . && rm -r training training-parallel-europarl-v7.tgz

echo "Downloading and extracting News Commentary data (76 MB) for training..."
wget --trust-server-names http://data.statmt.org/wmt16/translation-task/training-parallel-nc-v11.tgz
tar zxvf training-parallel-nc-v11.tgz
cd training-parallel-nc-v11 && ls | grep -v news-commentary-v11.de-en.[de,en] | xargs rm
cd .. && mv training-parallel-nc-v11/* . && rm -r training-parallel-nc-v11 training-parallel-nc-v11.tgz

Validation and test data are put into the $DATA_PATH/test folder
echo "Downloading and extracting newstest2014 data (4 MB) for validation..."
wget --trust-server-names http://www.statmt.org/wmt14/test-filtered.tgz
echo "Downloading and extracting newstest2017 data (5 MB) for testing..."
wget --trust-server-names http://data.statmt.org/wmt17/translation-task/test.tgz
tar zxvf test-filtered.tgz && tar zxvf test.tgz
cd test && ls | grep -v '.*deen\|.*ende' | xargs rm
cd .. && rm test-filtered.tgz test.tgz && cd ..
```

Actually, we can just run the above [script](https://github.com/OpenNMT/OpenNMT-py/blob/master/examples/scripts/prepare_wmt_data.sh) provided by OpenNMT-py directly. Due to it has already set the condition to `false`, we will not get a SentencePiece Model. 

After that, I concatenate the files that belong to the same language to a single file for convenience.

```shell
cat commoncrawl.de-en.de europarl-v7.de-en.de news-commentary-v11.de-en.de > train.de
cat commoncrawl.de-en.en europarl-v7.de-en.en news-commentary-v11.de-en.en > train.en
```

### 2) Preprocess the data

I use SentencePiece, an unsupervised text tokenizer of Google, to preprocess the data. You can get the concrete method of installation from [here](https://github.com/google/sentencepiece#installation).

##### First, we should train on the raw dataset to obtain a tokenization model. 

```shell
sl=de
tl=en
vocab_size=32000

DATA_PATH=$1
tmp_train = $DATA_PATH/tmp.train.txt

rm -rf $tmp_train

for f in $DATA_PATH/train.$sl $DATA_PATH/train.$tl
do
    cat $f >> $tmp_train
done

spm_train --input=$tmp_train --model_prefix-$DATA_PATH/wmt$sl$tl \
    --vocab_size=$vocab_size --character_coverage=1 --model_tpye=bpe

rm -rf $tmp_train
```

**NOTE**: 
- We use `--model_type=bpe` here. There are other types, like `unigram`, `char`, and `word`. Get more information from [here](https://github.com/google/sentencepiece#train-sentencepiece-model).
- We set `--vocav_size=32000` here. And it's up to the specific task.

##### After that, we tokenize the raw data by model we've trained. The commands are shown below.

```shell
sl=de
tl=en
DATA_PATH=$1

for f in $DATA_PATH/train.$sl $DATA_PATH/train.$tl $DATA_PATH/valid.$sl $DATA_PATH/valid.$tl $DATA_PATH/test.$sl
do
    file=$(basename $f)
    echo "# Tokenizing $file"
    spm_encode --model=$DATA_PATH/wmt$sl$tl.model --output_format=piece < $f > $DATA_PATH/$file.sp
done
```

##### Last, we run `preprocess.py` to process the dataset

```shell
python ../OpenNMT-py/preprocess.py \
    -train_src $src_train -train_tgt $tgt_train \
    -valid_src $src_valid -valid_tgt $tgt_valid \
    -save_data $target_dir \
    -src_vocab_size 32000 -tgt_vocab_size 32000 \
    -share_vocab -overwrite -lower
```

**NOTE**:
- We set `src_vocab_size = 32000` and `tgt_vocab_size = 32000` according to the original experiment setting. (But in practice I get a vocabulary size 30583 after the processing.)
- We set `-share_vocab` as well.
- `-overwrite` is used to overwrite the already exist data files.
- `-lower` used to convert the data into lower format.

### 3) Train the translation model

We use train a typical Transformer model by following parameters.

```shell
export CUDA_VISIBLE_DEVICES=0,1
python ../OpenNMT-py/train.py -data $DATA_PATH -save_model $MODEL_PATH \
    -layers 6 -rnn_size 512 -word_vec_size 512 -transformer_ff 2048 -heads 8 \
    -encoder_type transformer -decoder_type transformer -position_encoding \
    -train_steps 200000 -max_generator_batches 2 -dropout 0.1 \
    -batch_size 4096 -batch_type tokens -normalization tokens -accum_count 2 \
    -optim adam -adam_beta2 0.998 -decay_method noam -warmup_steps 8000 -learning_rate 2\
    -max_grad_norm 0 -param_init 0 -param_init_glorot \
    -label_smoothing 0.1 -valid_steps 10000 -save_checkpoint_steps 10000 \
    -world_size 2 -gpu_ranks 0 1 -share_embeddings
    
```

**NOTE**:
- We set `-share_embeddings` as what original paper said.

### 4) Translate

When the training finished, we can obtain the final translation results through `translate.py`.

##### First, get the generating results of model.

```shell
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
```

##### Secondly, use the sentencepiece model to detokenize the translation results.

```shell
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
```

### 5) Evaluate

I use SacreBLEU to obtain comparable BLEU scores. You can find the installation and other tutorials [here](https://github.com/mjpost/sacrebleu#quick-start).

And commands I used are as follows:

```shell
for checkpoint in $MODEL_PATH/wmt14-de-en_step*.pt; do
    echo "$checkpoint"
    base=$(basename $checkpoint)
    suffix=${base##*_}
    tgt=$DATA_PATH/wmt14-de-en/test.en.hyp_${suffix%.*}
    sacrebleu $DATA_PATH/wmt14-de-en/test.en < $tgt
done
```

### 6) Average

I use `average_models.py` in OpenNMT-py to get the average result. The specific commands are shown below:

```shell
python ../OpenNMT-py/average_models.py \
		-m $MODEL_PATH/wmt14-de-en_step_1*0000.pt \
		-o $MODEL_PATH/wmt14-de-en_step_average.pt
```

### 7) Result

I train the Transformer model on 2 Nvidia 1080Ti and get a `BLEU = 28.4` by averaging last 10 checkpoints finally.  