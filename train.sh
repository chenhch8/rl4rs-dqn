#!/bin/bash
OUTDIR=output
DUELING=0

#rm $OUTDIR/train.log
#nohup CUDA_VISIBLE_DEVICES="0,1" python3 -u src/train.py --outdir "./output_env1"  > output_env1/train.log 2>&1 | tail -F output_env1/train.log &
#CUDA_VISIBLE_DEVICES="-1" python3 -u src2/train.py --outdir $OUTDIR --dueling $DUELING  >> $OUTDIR/train.log 2>&1 | tail -F $OUTDIR/train.log
CUDA_VISIBLE_DEVICES="0" python3 src/train.py --outdir $OUTDIR --dueling $DUELING >> $OUTDIR/train.log 2>&1 &
tail -F $OUTDIR/train.log
#CUDA_VISIBLE_DEVICES="0" python3 src/train.py --outdir $OUTDIR --dueling $DUELING
