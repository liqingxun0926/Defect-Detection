#shell
python MyNet.py \
        --EVAL 0 \
        --BATCH 12 \
        --STEP 4000 \
        --IMGPATH ./train/ \
        --VALPATH ./eval/ \
        --TRAINLOGS ./train_logs/ \
        --VALLOGS ./eval_logs/ \
        --WIDTH 64 \
        --HEIGHT 64 \
        --CHANNELS 3 \
        --LRATE 0.0001 \
        --RATIO 0.8 \

