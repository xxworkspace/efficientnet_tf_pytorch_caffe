### TensorFlow to PyTorch Conversion
  * Download tf-model
    * cd pretrained_tensorflow
    * bash download.sh efficientnet-b*
  * Convert tf-model to pytorch-model
    * cd convert_tf_pt
    * python3 convert_params_tf_pytorch.py --model_name efficientnet-b0 --tf_checkpoint ../pretrained_tensorflow/efficientnet-b0/ --output_file ../pretrained_pytorch/efficientnet-b0.pth
      * Using python3 convert_params_tf_pytorch.py -h
  * Test
    * python test_pytorch.py

### PyTorch to Caffe
  * Convert pytorch-model to caffe .prototxt
    * python pytorch2caffe.py efficientnet-b*
  * Convert pytorch-model to caffe .caffemodel
    * python pytorch2caffe_model.py efficientnet-b*
  * Test
    * pytorch test_caffe.py efficientnet-b*

### data process RGB
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
