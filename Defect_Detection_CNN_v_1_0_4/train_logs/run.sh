python /usr/local/DNN/python_tools/freeze_graph.py \
       	--input_graph=saved_model.pb \
        --input_checkpoint=saved_model.ckpt-3500 \
	--input_binary=true \
        --output_graph=frozen_model.pb \
        --output_node_name=output

