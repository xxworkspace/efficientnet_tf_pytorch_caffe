import numpy as np
import tensorflow as tf
import torch

'''
def load_param(checkpoint_file, conversion_table, model_name):
    """
    Load parameters according to conversion_table.

    Args:
        checkpoint_file (string): pretrained checkpoint model file in tensorflow
        conversion_table (dict): { pytorch tensor in a model : checkpoint variable name }
    """
    dict = {}
    for pyt_param, tf_param_name in conversion_table.items():
        param_name = tf_param_name
        tf_param_name = str(model_name) + '/' +  tf_param_name
        tf_param = tf.train.load_variable(checkpoint_file, tf_param_name)
        if 'conv' in tf_param_name and 'kernel' in tf_param_name:
            tf_param = np.transpose(tf_param, (3, 2, 0, 1))
            if 'depthwise' in tf_param_name:
                tf_param = np.transpose(tf_param, (1, 0, 2, 3))
        elif tf_param_name.endswith('kernel'):  # for weight(kernel), we should do transpose
            tf_param = np.transpose(tf_param)
        assert pyt_param.size() == tf_param.shape, \
            'Dim Mismatch: %s vs %s ; %s' % (tuple(pyt_param.size()), tf_param.shape, tf_param_name)
        print(tf_param_name," --> ",type(pyt_param))
        #print(type(pyt_param))
        pyt_param = torch.from_numpy(tf_param)
        dict[param_name] = tf_param.copy()
        #print(pyt_param[0])
        #[pyt_param] = tf_param_name
        #print(tf_param_name,tf_param)
        #print(pyt_param)
        #dict[tf_param_name] = tf_param.copy()
        #print(tf_param)
        #if "bias" in tf_param_name:
          #print(pyt_param)
    #print(conversion_table)
    return dict
'''

def pytorch_param_from_tf_checkpoint(checkpoint_file, conversion_table, params_shape_table, model_name):
    """
    Load parameters according to conversion_table.

    Args:
        checkpoint_file (string): pretrained checkpoint model file in tensorflow
        conversion_table (dict): { pytorch tensor in a model : checkpoint variable name }
    """
    dict = {}
    params_shape_table_new = {}
    for k,v in params_shape_table.items():
      params_shape_table_new[v] = k

    for pyt_param_name, tf_param_name in conversion_table.items():
        tf_param_name_ = tf_param_name
        tf_param_name = str(model_name) + '/' +  tf_param_name
        tf_param = tf.train.load_variable(checkpoint_file, tf_param_name)
        if 'conv' in tf_param_name and 'kernel' in tf_param_name:
            tf_param = np.transpose(tf_param, (3, 2, 0, 1))
            if 'depthwise' in tf_param_name:
                tf_param = np.transpose(tf_param, (1, 0, 2, 3))
        elif tf_param_name.endswith('kernel'):
            tf_param = np.transpose(tf_param)
        assert params_shape_table_new[tf_param_name_].size() == tf_param.shape, \
            'Dim Mismatch: %s vs %s ; %s' % (tuple(params_shape_table_new[tf_param_name_].size()), tf_param.shape, tf_param_name)
        print(tf_param_name," --> ",pyt_param_name)
        dict[pyt_param_name] = tf_param.copy()

    return dict

def pytorch_tf_name_dict(model):
    # This will store the enire conversion table
    conversion_table = {}
    merge = lambda dict1, dict2: {**dict1, **dict2}

    # All the weights not in the conv blocks
    conversion_table_for_weights_outside_blocks = {
        "_conv_stem.weight": 'stem/conv2d/kernel',  # [3, 3, 3, 32]),
        "_bn0.bias": 'stem/tpu_batch_normalization/beta',  # [32]),
        "_bn0.weight": 'stem/tpu_batch_normalization/gamma',  # [32]),
        "_bn0.running_mean": 'stem/tpu_batch_normalization/moving_mean',  # [32]),
        "_bn0.running_var": 'stem/tpu_batch_normalization/moving_variance',  # [32]),
        "_conv_head.weight": 'head/conv2d/kernel',  # [1, 1, 320, 1280]),
        "_bn1.bias": 'head/tpu_batch_normalization/beta',  # [1280]),
        "_bn1.weight": 'head/tpu_batch_normalization/gamma',  # [1280]),
        "_bn1.running_mean": 'head/tpu_batch_normalization/moving_mean',  # [32]),
        "_bn1.running_var": 'head/tpu_batch_normalization/moving_variance',  # [32]),
        "_fc.bias": 'head/dense/bias',  # [1000]),
        "_fc.weight": 'head/dense/kernel',  # [1280, 1000]),
    }
    conversion_table = merge(conversion_table, conversion_table_for_weights_outside_blocks)

    # The first conv block is special because it does not have _expand_conv
    conversion_table_for_first_block = {
        "_blocks.0._project_conv.weight": 'blocks_0/conv2d/kernel',  # 1, 1, 32, 16]),
        "_blocks.0._depthwise_conv.weight": 'blocks_0/depthwise_conv2d/depthwise_kernel',  # [3, 3, 32, 1]),
        "_blocks.0._se_reduce.bias": 'blocks_0/se/conv2d/bias',  # , [8]),
        "_blocks.0._se_reduce.weight": 'blocks_0/se/conv2d/kernel',  # , [1, 1, 32, 8]),
        "_blocks.0._se_expand.bias": 'blocks_0/se/conv2d_1/bias',  # , [32]),
        "_blocks.0._se_expand.weight": 'blocks_0/se/conv2d_1/kernel',  # , [1, 1, 8, 32]),
        "_blocks.0._bn1.bias": 'blocks_0/tpu_batch_normalization/beta',  # [32]),
        "_blocks.0._bn1.weight": 'blocks_0/tpu_batch_normalization/gamma',  # [32]),
        "_blocks.0._bn1.running_mean": 'blocks_0/tpu_batch_normalization/moving_mean',
        "_blocks.0._bn1.running_var": 'blocks_0/tpu_batch_normalization/moving_variance',
        "_blocks.0._bn2.bias": 'blocks_0/tpu_batch_normalization_1/beta',  # [16]),
        "_blocks.0._bn2.weight": 'blocks_0/tpu_batch_normalization_1/gamma',  # [16]),
        "_blocks.0._bn2.running_mean": 'blocks_0/tpu_batch_normalization_1/moving_mean',
        "_blocks.0._bn2.running_var": 'blocks_0/tpu_batch_normalization_1/moving_variance',
    }
    conversion_table = merge(conversion_table, conversion_table_for_first_block)

    # Conv blocks
    for i in range(len(model._blocks)):

        is_first_block = '_expand_conv.weight' not in [n for n, p in model._blocks[i].named_parameters()]

        if is_first_block:
            conversion_table_block = {
                "_blocks." + str(i) +"._project_conv.weight": 'blocks_' + str(i) + '/conv2d/kernel',  # 1, 1, 32, 16]),
                "_blocks." + str(i) +"._depthwise_conv.weight": 'blocks_' + str(i) + '/depthwise_conv2d/depthwise_kernel',
                # [3, 3, 32, 1]),
                "_blocks." + str(i) +"._se_reduce.bias": 'blocks_' + str(i) + '/se/conv2d/bias',  # , [8]),
                "_blocks." + str(i) +"._se_reduce.weight": 'blocks_' + str(i) + '/se/conv2d/kernel',  # , [1, 1, 32, 8]),
                "_blocks." + str(i) +"._se_expand.bias": 'blocks_' + str(i) + '/se/conv2d_1/bias',  # , [32]),
                "_blocks." + str(i) +"._se_expand.weight": 'blocks_' + str(i) + '/se/conv2d_1/kernel',  # , [1, 1, 8, 32]),
                "_blocks." + str(i) +"._bn1.bias": 'blocks_' + str(i) + '/tpu_batch_normalization/beta',  # [32]),
                "_blocks." + str(i) +"._bn1.weight": 'blocks_' + str(i) + '/tpu_batch_normalization/gamma',  # [32]),
                "_blocks." + str(i) +"._bn1.running_mean": 'blocks_' + str(i) + '/tpu_batch_normalization/moving_mean',
                "_blocks." + str(i) +"._bn1.running_var": 'blocks_' + str(i) + '/tpu_batch_normalization/moving_variance',
                "_blocks." + str(i) +"._bn2.bias": 'blocks_' + str(i) + '/tpu_batch_normalization_1/beta',  # [16]),
                "_blocks." + str(i) +"._bn2.weight": 'blocks_' + str(i) + '/tpu_batch_normalization_1/gamma',  # [16]),
                "_blocks." + str(i) +"._bn2.running_mean": 'blocks_' + str(i) + '/tpu_batch_normalization_1/moving_mean',
                "_blocks." + str(i) +"._bn2.running_var": 'blocks_' + str(i) + '/tpu_batch_normalization_1/moving_variance',
            }

        else:
            conversion_table_block = {
                "_blocks." + str(i) +"._expand_conv.weight":       'blocks_' + str(i) + '/conv2d/kernel',
                "_blocks." + str(i) +"._project_conv.weight":      'blocks_' + str(i) + '/conv2d_1/kernel',
                "_blocks." + str(i) +"._depthwise_conv.weight":    'blocks_' + str(i) + '/depthwise_conv2d/depthwise_kernel',
                "_blocks." + str(i) +"._se_reduce.bias":           'blocks_' + str(i) + '/se/conv2d/bias',
                "_blocks." + str(i) +"._se_reduce.weight":         'blocks_' + str(i) + '/se/conv2d/kernel',
                "_blocks." + str(i) +"._se_expand.bias":           'blocks_' + str(i) + '/se/conv2d_1/bias',
                "_blocks." + str(i) +"._se_expand.weight":         'blocks_' + str(i) + '/se/conv2d_1/kernel',
                "_blocks." + str(i) +"._bn0.bias":                 'blocks_' + str(i) + '/tpu_batch_normalization/beta',
                "_blocks." + str(i) +"._bn0.weight":               'blocks_' + str(i) + '/tpu_batch_normalization/gamma',
                "_blocks." + str(i) +"._bn0.running_mean":         'blocks_' + str(i) + '/tpu_batch_normalization/moving_mean',
                "_blocks." + str(i) +"._bn0.running_var":          'blocks_' + str(i) + '/tpu_batch_normalization/moving_variance',
                "_blocks." + str(i) +"._bn1.bias":                 'blocks_' + str(i) + '/tpu_batch_normalization_1/beta',
                "_blocks." + str(i) +"._bn1.weight":               'blocks_' + str(i) + '/tpu_batch_normalization_1/gamma',
                "_blocks." + str(i) +"._bn1.running_mean":         'blocks_' + str(i) + '/tpu_batch_normalization_1/moving_mean',
                "_blocks." + str(i) +"._bn1.running_var":          'blocks_' + str(i) + '/tpu_batch_normalization_1/moving_variance',
                "_blocks." + str(i) +"._bn2.bias":                 'blocks_' + str(i) + '/tpu_batch_normalization_2/beta',
                "_blocks." + str(i) +"._bn2.weight":               'blocks_' + str(i) + '/tpu_batch_normalization_2/gamma',
                "_blocks." + str(i) +"._bn2.running_mean":         'blocks_' + str(i) + '/tpu_batch_normalization_2/moving_mean',
                "_blocks." + str(i) +"._bn2.running_var":          'blocks_' + str(i) + '/tpu_batch_normalization_2/moving_variance',
            }

        conversion_table = merge(conversion_table, conversion_table_block)
    #return load_param_from_tf(checkpoint_file,conversion_table,model_name)
    return conversion_table

def pytorch_tf_params_dict(model):
    """
    
    """
    # This will store the enire conversion table
    conversion_table = {}
    merge = lambda dict1, dict2: {**dict1, **dict2}

    # All the weights not in the conv blocks
    conversion_table_for_weights_outside_blocks = {
        model._conv_stem.weight: 'stem/conv2d/kernel',  # [3, 3, 3, 32]),
        model._bn0.bias: 'stem/tpu_batch_normalization/beta',  # [32]),
        model._bn0.weight: 'stem/tpu_batch_normalization/gamma',  # [32]),
        model._bn0.running_mean: 'stem/tpu_batch_normalization/moving_mean',  # [32]),
        model._bn0.running_var: 'stem/tpu_batch_normalization/moving_variance',  # [32]),
        model._conv_head.weight: 'head/conv2d/kernel',  # [1, 1, 320, 1280]),
        model._bn1.bias: 'head/tpu_batch_normalization/beta',  # [1280]),
        model._bn1.weight: 'head/tpu_batch_normalization/gamma',  # [1280]),
        model._bn1.running_mean: 'head/tpu_batch_normalization/moving_mean',  # [32]),
        model._bn1.running_var: 'head/tpu_batch_normalization/moving_variance',  # [32]),
        model._fc.bias: 'head/dense/bias',  # [1000]),
        model._fc.weight: 'head/dense/kernel',  # [1280, 1000]),
    }
    conversion_table = merge(conversion_table, conversion_table_for_weights_outside_blocks)

    # The first conv block is special because it does not have _expand_conv
    conversion_table_for_first_block = {
        model._blocks[0]._project_conv.weight: 'blocks_0/conv2d/kernel',  # 1, 1, 32, 16]),
        model._blocks[0]._depthwise_conv.weight: 'blocks_0/depthwise_conv2d/depthwise_kernel',  # [3, 3, 32, 1]),
        model._blocks[0]._se_reduce.bias: 'blocks_0/se/conv2d/bias',  # , [8]),
        model._blocks[0]._se_reduce.weight: 'blocks_0/se/conv2d/kernel',  # , [1, 1, 32, 8]),
        model._blocks[0]._se_expand.bias: 'blocks_0/se/conv2d_1/bias',  # , [32]),
        model._blocks[0]._se_expand.weight: 'blocks_0/se/conv2d_1/kernel',  # , [1, 1, 8, 32]),
        model._blocks[0]._bn1.bias: 'blocks_0/tpu_batch_normalization/beta',  # [32]),
        model._blocks[0]._bn1.weight: 'blocks_0/tpu_batch_normalization/gamma',  # [32]),
        model._blocks[0]._bn1.running_mean: 'blocks_0/tpu_batch_normalization/moving_mean',
        model._blocks[0]._bn1.running_var: 'blocks_0/tpu_batch_normalization/moving_variance',
        model._blocks[0]._bn2.bias: 'blocks_0/tpu_batch_normalization_1/beta',  # [16]),
        model._blocks[0]._bn2.weight: 'blocks_0/tpu_batch_normalization_1/gamma',  # [16]),
        model._blocks[0]._bn2.running_mean: 'blocks_0/tpu_batch_normalization_1/moving_mean',
        model._blocks[0]._bn2.running_var: 'blocks_0/tpu_batch_normalization_1/moving_variance',
    }
    conversion_table = merge(conversion_table, conversion_table_for_first_block)

    # Conv blocks
    for i in range(len(model._blocks)):

        is_first_block = '_expand_conv.weight' not in [n for n, p in model._blocks[i].named_parameters()]

        if is_first_block:
            conversion_table_block = {
                model._blocks[i]._project_conv.weight: 'blocks_' + str(i) + '/conv2d/kernel',  # 1, 1, 32, 16]),
                model._blocks[i]._depthwise_conv.weight: 'blocks_' + str(i) + '/depthwise_conv2d/depthwise_kernel',
                # [3, 3, 32, 1]),
                model._blocks[i]._se_reduce.bias: 'blocks_' + str(i) + '/se/conv2d/bias',  # , [8]),
                model._blocks[i]._se_reduce.weight: 'blocks_' + str(i) + '/se/conv2d/kernel',  # , [1, 1, 32, 8]),
                model._blocks[i]._se_expand.bias: 'blocks_' + str(i) + '/se/conv2d_1/bias',  # , [32]),
                model._blocks[i]._se_expand.weight: 'blocks_' + str(i) + '/se/conv2d_1/kernel',  # , [1, 1, 8, 32]),
                model._blocks[i]._bn1.bias: 'blocks_' + str(i) + '/tpu_batch_normalization/beta',  # [32]),
                model._blocks[i]._bn1.weight: 'blocks_' + str(i) + '/tpu_batch_normalization/gamma',  # [32]),
                model._blocks[i]._bn1.running_mean: 'blocks_' + str(i) + '/tpu_batch_normalization/moving_mean',
                model._blocks[i]._bn1.running_var: 'blocks_' + str(i) + '/tpu_batch_normalization/moving_variance',
                model._blocks[i]._bn2.bias: 'blocks_' + str(i) + '/tpu_batch_normalization_1/beta',  # [16]),
                model._blocks[i]._bn2.weight: 'blocks_' + str(i) + '/tpu_batch_normalization_1/gamma',  # [16]),
                model._blocks[i]._bn2.running_mean: 'blocks_' + str(i) + '/tpu_batch_normalization_1/moving_mean',
                model._blocks[i]._bn2.running_var: 'blocks_' + str(i) + '/tpu_batch_normalization_1/moving_variance',
            }

        else:
            conversion_table_block = {
                model._blocks[i]._expand_conv.weight:       'blocks_' + str(i) + '/conv2d/kernel',
                model._blocks[i]._project_conv.weight:      'blocks_' + str(i) + '/conv2d_1/kernel',
                model._blocks[i]._depthwise_conv.weight:    'blocks_' + str(i) + '/depthwise_conv2d/depthwise_kernel',
                model._blocks[i]._se_reduce.bias:           'blocks_' + str(i) + '/se/conv2d/bias',
                model._blocks[i]._se_reduce.weight:         'blocks_' + str(i) + '/se/conv2d/kernel',
                model._blocks[i]._se_expand.bias:           'blocks_' + str(i) + '/se/conv2d_1/bias',
                model._blocks[i]._se_expand.weight:         'blocks_' + str(i) + '/se/conv2d_1/kernel',
                model._blocks[i]._bn0.bias:                 'blocks_' + str(i) + '/tpu_batch_normalization/beta',
                model._blocks[i]._bn0.weight:               'blocks_' + str(i) + '/tpu_batch_normalization/gamma',
                model._blocks[i]._bn0.running_mean:         'blocks_' + str(i) + '/tpu_batch_normalization/moving_mean',
                model._blocks[i]._bn0.running_var:          'blocks_' + str(i) + '/tpu_batch_normalization/moving_variance',
                model._blocks[i]._bn1.bias:                 'blocks_' + str(i) + '/tpu_batch_normalization_1/beta',
                model._blocks[i]._bn1.weight:               'blocks_' + str(i) + '/tpu_batch_normalization_1/gamma',
                model._blocks[i]._bn1.running_mean:         'blocks_' + str(i) + '/tpu_batch_normalization_1/moving_mean',
                model._blocks[i]._bn1.running_var:          'blocks_' + str(i) + '/tpu_batch_normalization_1/moving_variance',
                model._blocks[i]._bn2.bias:                 'blocks_' + str(i) + '/tpu_batch_normalization_2/beta',
                model._blocks[i]._bn2.weight:               'blocks_' + str(i) + '/tpu_batch_normalization_2/gamma',
                model._blocks[i]._bn2.running_mean:         'blocks_' + str(i) + '/tpu_batch_normalization_2/moving_mean',
                model._blocks[i]._bn2.running_var:          'blocks_' + str(i) + '/tpu_batch_normalization_2/moving_variance',
            }

        conversion_table = merge(conversion_table, conversion_table_block)

    return conversion_table

def load_and_save_temporary_tensorflow_model(model_name, model_ckpt, example_img= '../../example/img.jpg'):
    """ Loads and saves a TensorFlow model. """
    image_files = [example_img]
    eval_ckpt_driver = eval_ckpt_main.EvalCkptDriver(model_name)
    with tf.Graph().as_default(), tf.Session() as sess:
        images, labels = eval_ckpt_driver.build_dataset(image_files, [0] * len(image_files), False)
        probs = eval_ckpt_driver.build_model(images, is_training=False)
        sess.run(tf.global_variables_initializer())
        print(model_ckpt)
        eval_ckpt_driver.restore_model(sess, model_ckpt)
        tf.train.Saver().save(sess, 'tmp/model.ckpt')


if __name__ == '__main__':

    import sys
    import argparse

    sys.path.append('original_tf')
    import eval_ckpt_main

    from model import *

    parser = argparse.ArgumentParser(
        description='Convert TF model to PyTorch model and save for easier future loading, \
        python3 convert_params_tf_pytorch.py --model_name efficientnet-b0 \
        --tf_checkpoint ../pretrained_tensorflow/efficientnet-b0/ \
        --output_file ../pretrained_pytorch/efficientnet-b0.pth')

    parser.add_argument('--model_name', type=str, default='efficientnet-b0',
                        help='efficientnet-b{N}, where N is an integer 0 <= N <= 7')
    parser.add_argument('--tf_checkpoint', type=str, default='../pretrained_tensorflow/efficientnet-b0/',
                        help='checkpoint file path')
    parser.add_argument('--output_file', type=str, default='../pretrained_pytorch/efficientnet-b0.pth',
                        help='output PyTorch model file name')
    args = parser.parse_args()

    # Build model
    model = get_from_name(args.model_name)

    # Load and save temporary TensorFlow file due to TF nuances
    print(args.tf_checkpoint)
    load_and_save_temporary_tensorflow_model(args.model_name, args.tf_checkpoint)

    pyt_tf_name_dict = pytorch_tf_name_dict(model)
    pyt_tf_params_dict = pytorch_tf_params_dict(model)
    pyt_params_dict = pytorch_param_from_tf_checkpoint('tmp/model.ckpt', pyt_tf_name_dict, pyt_tf_params_dict,args.model_name)

    # load pytorch param dict
    #pytorch_dict = pytorch_tf_dict(model,'tmp/model.ckpt',model_name=args.model_name)
    state_dict = model.state_dict()
    for k,v in state_dict.items():
      if k in pyt_params_dict:
        print(k," : ",v.numpy().shape," --> ",pyt_params_dict[k].shape)
        state_dict[k] = torch.from_numpy(pyt_params_dict[k])
      else:
        print("Warning --> ",k," not in pytorch_params_dict!!!")

    torch.save(state_dict, args.output_file)
    print('Saved model to', args.output_file)
