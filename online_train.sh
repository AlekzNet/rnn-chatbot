#!/bin/ksh

# Take the files to train from here
FROM_DIR=/data/from
# Move, when done
DONE_DIR=/data/done

# NN checkpoint to init from

INIT_FROM=/nn/online-test-1.t7

# NN checkpoint to save to
SAVE_TO=/nn/cp_3x1000-0.1.online-2.t7

# NN prog dir
PROG_DIR=/nn/rnn-chatbot


for file in ${FROM_DIR}/*
do
	echo "Processing $file"
	python $PROG_DIR/scripts/preprocess-my.py --input_txt $file --output_h5 ${file}.h5
	echo "Training with $file"
	th $PROG_DIR/mytrain.lua -gpu 0 -online -init_from $INIT_FROM -outfile $SAVE_TO -input_h5 ${file}.h5
	echo "Training with $file is complete"
	echo "The result is saved to $SAVE_TO"
	mv $file $DONE_DIR
	rm ${file}.h5
	INIT_FROM=$SAVE_TO
done
th $PROG_DIR/reset.lua -gpu 0 -init_from $SAVE_TO
th $PROG_DIR/mysample.lua -temperature 0.6 -checkpoint ${SAVE_TO}.reset.t7
