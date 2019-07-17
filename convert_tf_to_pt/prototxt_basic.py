# prototxt_basic
import sys
import logging
logging.basicConfig(level = logging.INFO)

def data(txt_file, info):
  txt_file.write('name: "efficientnet_pytorch2caffe"\n')
  txt_file.write('layer {\n')
  txt_file.write('  name: "data"\n')
  txt_file.write('  type: "Input"\n')
  txt_file.write('  top: "data"\n')
  txt_file.write('  input_param {\n')
  #txt_file.write('    shape: { dim: 10 dim: 3 dim: 224 dim: 224 }\n') # TODO
  txt_file.write('    shape: { dim: 1 dim: 3 dim: 224 dim: 224 }\n') # TODO
  txt_file.write('  }\n')
  txt_file.write('}\n')
  txt_file.write('\n')

def SliceChannel(txt_file, info):
  txt_file.write('layer {\n')
  txt_file.write('  bottom: "%s"\n'       % info['bottom'][0])
  txt_file.write('  top: "%s"\n'          % info['top'])
  txt_file.write('  name: "%s"\n'         % info['top'])
  txt_file.write('  type: "SliceChannel"\n')
  txt_file.write('}\n')
  txt_file.write('\n')

def L2Normalization(txt_file, info):
  txt_file.write('layer {\n')
  txt_file.write('  bottom: "%s"\n'       % info['bottom'][0])
  txt_file.write('  top: "%s"\n'          % info['top'])
  txt_file.write('  name: "%s"\n'         % info['top'])
  txt_file.write('  type: "L2Normalization"\n')
  txt_file.write('}\n')
  txt_file.write('\n')

def DropOut(txt_file, info):
  txt_file.write('layer {\n')
  txt_file.write('  bottom: "%s"\n'       % info['bottom'][0])
  txt_file.write('  top: "%s"\n'          % info['top'])
  txt_file.write('  name: "%s"\n'         % info['top'])
  txt_file.write('  type: "Dropout"\n')
  
  txt_file.write('  dropout_param {\n')
  txt_file.write('    dropout_ratio: %s\n'% str(1.0 - float(info['attrs']['p'])))
  txt_file.write('  }\n')
  
  txt_file.write('}\n')
  txt_file.write('\n')

def Convolution(txt_file, info):
  if info['attrs']['no_bias'] == 'True':
    bias_term = 'false'
  else:
    bias_term = 'true'  
  txt_file.write('layer {\n')
  txt_file.write('  bottom: "%s"\n'       % info['bottom'][0])
  txt_file.write('  top: "%s"\n'          % info['top'])
  txt_file.write('  name: "%s"\n'         % info['top'])
  txt_file.write('  type: "Convolution"\n')
  txt_file.write('  convolution_param {\n')
  txt_file.write('    num_output: %s\n'   % info['attrs']['num_filter'])
  txt_file.write('    kernel_size: %s\n'  % info['attrs']['kernel'].split('(')[1].split(',')[0]) # TODO
  if 'pad' not in info['attrs']:
    logging.info('miss Conv_pad, make pad default: 0 ')
    txt_file.write('    pad: %s\n' % 0)  # TODO
  else:
    #print(info['attrs']['pad'])
    txt_file.write('    pad: %s\n'          % info['attrs']['pad'].split('(')[1].split(',')[0]) # TODO

  if "num_group" in info['attrs']:
    txt_file.write('    group: %s\n'        % info['attrs']['num_group'])

  txt_file.write('    stride: %s\n'       % info['attrs']['stride'].split('(')[1].split(',')[0])
  txt_file.write('    bias_term: %s\n'    % bias_term)
  txt_file.write('  }\n')

  if 'share' in info.keys() and info['share']: 
    print("Log -> ",info['top'],info['share'])  
    txt_file.write('    param {\n')
    txt_file.write('      name: "%s"\n'     % info['params'][0])
    txt_file.write('    }\n')
  txt_file.write('}\n')
  txt_file.write('\n')

def ChannelwiseConvolution(txt_file, info):
  Convolution(txt_file, info)
  
def BatchNorm(txt_file, info):
  txt_file.write('layer {\n')
  txt_file.write('  bottom: "%s"\n'       % info['bottom'][0])
  txt_file.write('  top: "%s"\n'          % info['top'])
  txt_file.write('  name: "%s"\n'         % info['top'])
  txt_file.write('  type: "BatchNorm"\n')
  txt_file.write('  batch_norm_param {\n')
  txt_file.write('    use_global_stats: true\n')        # TODO
  #txt_file.write('    moving_average_fraction: 0.9\n')  # TODO
  txt_file.write('    eps: %s\n'          %info['attrs']["eps"])# TODO

  txt_file.write('  }\n')
  txt_file.write('}\n')
  # if info['fix_gamma'] is "False":                    # TODO
  txt_file.write('layer {\n')
  txt_file.write('  bottom: "%s"\n'       % info['top'])
  txt_file.write('  top: "%s"\n'          % info['top'])
  txt_file.write('  name: "%s_scale"\n'   % info['top'])
  txt_file.write('  type: "Scale"\n')
  txt_file.write('  scale_param { bias_term: true }\n')
  txt_file.write('}\n')
  txt_file.write('\n')
  pass


def Activation(txt_file, info):
  txt_file.write('layer {\n')
  txt_file.write('  bottom: "%s"\n'       % info['bottom'][0])
  txt_file.write('  top: "%s"\n'          % info['top'])
  txt_file.write('  name: "%s"\n'         % info['top'])
  if info['attrs']['act_type'] == 'sigmoid':   # TODO
    txt_file.write('  type: "Sigmoid"\n')
  elif info['attrs']['act_type'] == 'relu':
    txt_file.write('  type: "ReLU" \n')
  elif info['attrs']['act_type'] == 'swish':
    txt_file.write('  type: "Swish" \n')
  else:
    logging.info("Unknown avtivate_function %s" % info['attrs']['act_type'])
    
  txt_file.write('}\n')
  txt_file.write('\n')
  pass

def Concat(txt_file, info):
  #for dense net for tensorrt
  '''
  if "concat" in info['bottom'][0]:
    txt_file.write('layer {\n')
    txt_file.write('  bottom: "%s"\n'       % (info['bottom'][0]))
    txt_file.write('  top: "%s"\n'          % (info['bottom'][0] + "_Copy"))
    txt_file.write('  name: "%s"\n'         % (info['bottom'][0] + "_Copy"))
    txt_file.write('  type: "Pooling"\n')
    txt_file.write('  pooling_param {\n')
    txt_file.write('    pool: MAX\n')
    txt_file.write('    kernel_size: 1\n')
    txt_file.write('    stride: 1\n')
    txt_file.write('    pad: 0\n')
    txt_file.write('  }\n')
    txt_file.write('}\n')
    txt_file.write('\n')
  '''
  txt_file.write('layer {\n')
  txt_file.write('  name: "%s"\n'         % info['top'])
  txt_file.write('  type: "Concat"\n')
  if len(info['bottom']) > 2:
    for bottom_i in info['bottom']:
      txt_file.write('  bottom: "%s"\n'     % bottom_i)
  else:
    if "concat" in info['bottom'][0]:
      txt_file.write('  bottom: "%s"\n'     % (info['bottom'][0] + ""))
    else:
      txt_file.write('  bottom: "%s"\n'     % (info['bottom'][0]))
    txt_file.write('  bottom: "%s"\n'       % info['bottom'][1])

  txt_file.write('  top: "%s"\n'          % (info['top']))
  txt_file.write('}\n')
  txt_file.write('\n')
  
  '''
  #using slice instead for int8 quantitation
  txt_file.write('layer {\n')
  txt_file.write('  name: "%s"\n'         % (info['top'] + "_Copy"))
  txt_file.write('  type: "Slice"\n')
  txt_file.write('  bottom: "%s"\n'       % (info['top']))
  txt_file.write('  top: "%s"\n'          % (info['top'] + "_Copy"))
  txt_file.write('  slice_param {\n')
  txt_file.write('    axis: 0\n')
  txt_file.write('  }\n')
  txt_file.write('}\n')
  txt_file.write('\n')
  '''
  #using pooling instead for int8 quantitation
  '''
  txt_file.write('layer {\n')
  txt_file.write('  bottom: "%s"\n'       % (info['top']))
  txt_file.write('  top: "%s"\n'          % (info['top'] + "_Copy"))
  txt_file.write('  name: "%s"\n'         % (info['top'] + "_Copy"))
  txt_file.write('  type: "Pooling"\n')
  txt_file.write('  pooling_param {\n')
  txt_file.write('    pool: MAX\n')
  txt_file.write('    kernel_size: 1\n')
  txt_file.write('    stride: 1\n')
  txt_file.write('    pad: 0\n')
  txt_file.write('  }\n')
  txt_file.write('}\n')
  txt_file.write('\n')
  '''

  pass

def ElementWiseSum(txt_file, info):
  txt_file.write('layer {\n')
  txt_file.write('  name: "%s"\n'         % info['top'])
  txt_file.write('  type: "Eltwise"\n')
  for bottom_i in info['bottom']:
    txt_file.write('  bottom: "%s"\n'     % bottom_i)
  txt_file.write('  top: "%s"\n'          % info['top'])
  txt_file.write('  eltwise_param {\n')
  txt_file.write('    operation: SUM\n')
  txt_file.write('  }\n')
  txt_file.write('}\n')
  txt_file.write('\n')
  pass

def Pooling(txt_file, info):
  pool_type = 'AVE' if info['attrs']['pool_type'] == 'avg' else 'MAX'
  txt_file.write('layer {\n')
  txt_file.write('  bottom: "%s"\n'       % info['bottom'][0])
  txt_file.write('  top: "%s"\n'          % info['top'])
  txt_file.write('  name: "%s"\n'         % info['top'])
  txt_file.write('  type: "Pooling"\n')
  txt_file.write('  pooling_param {\n')
  txt_file.write('    pool: %s\n'         % pool_type)       # TODO
  txt_file.write('    kernel_size: %s\n'  % info['attrs']['kernel'].split('(')[1].split(',')[0])
  if 'global_pool' not in info['attrs'] or info['attrs']['global_pool'] == 'False':
    txt_file.write('    stride: %s\n'       % info['attrs']['stride'].split('(')[1].split(',')[0])
    txt_file.write('    pad: %s\n'          % info['attrs']['pad'].split('(')[1].split(',')[0])
  else:
    txt_file.write('    global_pooling: True\n')

  if 'pooling_convention' in info['attrs']:
    if info['attrs']['pooling_convention'] == "valid":
      #must be care for
      txt_file.write('    round_mode: FLOOR\n')    #used for caffe model
      #txt_file.write('    torch_pooling: true\n') #used for tensorrt caffe parser

  txt_file.write('  }\n')
  txt_file.write('}\n')
  txt_file.write('\n')
  pass

def Pooling_global(txt_file, info):
  if 'global_pool' not in info['attrs']:
    Pooling(txt_file, info)
    return

  if info['attrs']['global_pool'] == 'False':
     Pooling(txt_file, info)

  elif info['attrs']['global_pool'] == 'True':
    pool_type = 'AVE' if info['attrs']['pool_type'] == 'avg' else 'MAX'
    txt_file.write('layer {\n')
    txt_file.write('  bottom: "%s"\n'     % info['bottom'][0])
    txt_file.write('  top: "%s"\n'        % info['top'])
    txt_file.write('  name: "%s"\n'       % info['top'])
    txt_file.write('  type: "Pooling"\n')
    txt_file.write('  pooling_param {\n')
    txt_file.write('    pool: %s\n'       % pool_type)
    txt_file.write('    global_pooling: true\n')
    txt_file.write('  }\n')
    txt_file.write('}\n')
    txt_file.write('\n')
  pass

def FullyConnected(txt_file, info):
  txt_file.write('layer {\n')
  txt_file.write('  bottom: "%s"\n'     % info['bottom'][0])
  txt_file.write('  top: "%s"\n'        % info['top'])
  txt_file.write('  name: "%s"\n'       % info['top'])
  txt_file.write('  type: "InnerProduct"\n')
  txt_file.write('  inner_product_param {\n')
  txt_file.write('    num_output: %s\n' % info['attrs']['num_hidden'])
  txt_file.write('  }\n')
  txt_file.write('}\n')
  txt_file.write('\n')
  pass

def Flatten(txt_file, info):
  txt_file.write('layer {\n')
  txt_file.write('  bottom: "%s"\n'     % info['bottom'][0])
  txt_file.write('  top: "%s"\n'        % info['top'])
  txt_file.write('  name: "%s"\n'       % info['top'])
  txt_file.write('  type: "Flatten"\n')
  txt_file.write('}\n')
  txt_file.write('\n')
  
def SoftmaxOutput(txt_file, info):
  txt_file.write('layer {\n')
  txt_file.write('  bottom: "%s"\n'     % info['bottom'][0])
  txt_file.write('  top: "%s"\n'        % info['top'])
  txt_file.write('  name: "%s"\n'       % info['top'])
  txt_file.write('  type: "Softmax"\n')
  txt_file.write('}\n')
  txt_file.write('\n')

def Reshape(txt_file,info):
  txt_file.write('layer {\n')
  txt_file.write('  bottom: "%s"\n'       % info['bottom'][0])
  txt_file.write('  top: "%s"\n'          % info['top'])
  txt_file.write('  name: "%s"\n'         % info['top'])
  txt_file.write('  type: "Reshape"\n')
  txt_file.write('  reshape_param {\n')
  txt_file.write('    shape {\n')
  for dim in info['attrs']['shape'].split('(')[1].split(')')[0].split(','):
    txt_file.write('      dim: %s\n'      % dim)
  #txt_file.write('      dim: %s\n'        % info['attrs']['shape'].split('(')[1].split(',')[1])
  #txt_file.write('      dim: %s\n'        % info['attrs']['shape'].split('(')[1].split(',')[2])
  #txt_file.write('      dim: %s\n'        % info['attrs']['shape'].split(')')[0].split(',')[3])
  txt_file.write('    }\n')
  txt_file.write('    axis: 1  \n')
  txt_file.write('  }\n')
  txt_file.write('}\n')
  txt_file.write('\n')


def broadcast_mul(txt_file,info):
  txt_file.write('layer {\n')
  txt_file.write('  name: "%s"\n'         % info['top'])
  txt_file.write('  type: "BroadcastMul"\n')
  for bottom_i in info['bottom']:
    txt_file.write('  bottom: "%s"\n'     % bottom_i)
  txt_file.write('  top: "%s"\n'          % info['top'])
  txt_file.write('}\n')
  txt_file.write('\n')

def broadcast_add(txt_file,info):
  txt_file.write('layer {\n')
  txt_file.write('  name: "%s"\n'         % info['top'])
  txt_file.write('  type: "Eltwise"\n')
  for bottom_i in info['bottom']:
    txt_file.write('  bottom: "%s"\n'     % bottom_i)
  txt_file.write('  top: "%s"\n'          % info['top'])
  txt_file.write('  eltwise_param {\n')
  txt_file.write('    operation: SUM\n')
  txt_file.write('  }\n')
  txt_file.write('}\n')
  txt_file.write('\n')

def mulscalar(txt_file,info):
  '''
  txt_file.write('layer {\n')
  txt_file.write('  top: "%s"\n'          % (info['top'] + "_second"))
  txt_file.write('  name: "%s"\n'         % (info['top'] + "_second"))
  txt_file.write('  type: "DummyData"\n')
  txt_file.write('  dummy_data_param {\n')
  txt_file.write('    data_filler {\n')
  txt_file.write('        type: "constant"\n')
  txt_file.write('        value: %s\n'    % info['attrs']['scalar'])
  txt_file.write('    }\n')
  txt_file.write('    shape: { dim: 1 dim: 3 dim: 224 dim: 224 }\n') # TODO
  txt_file.write('  }\n')
  txt_file.write('}\n')

  txt_file.write('layer {\n')
  txt_file.write('  name: "%s"\n'         % info['top'])
  txt_file.write('  type: "Eltwise"\n')
  for bottom_i in info['bottom']:
    txt_file.write('  bottom: "%s"\n'     % bottom_i)
  txt_file.write('  bottom: "%s"\n'       % (info['top'] + "_second"))
  txt_file.write('  top: "%s"\n'          % info['top'])
  txt_file.write('  eltwise_param {\n')
  txt_file.write('    operation: PROD\n')
  txt_file.write('  }\n')
  txt_file.write('}\n')
  txt_file.write('\n')
  '''
  txt_file.write('layer {\n')
  txt_file.write('  bottom: "%s"\n'        % info['bottom'][0])
  txt_file.write('  top:  "%s"\n'          % info['top'])
  txt_file.write('  name: "%s"\n'          % info['top'])
  txt_file.write('  type: "Scale"\n')
  txt_file.write('  scale_param {\n')
  txt_file.write('    bias_term: false\n')
  txt_file.write('    filler {\n')
  txt_file.write('      type:  "constant"\n') 
  txt_file.write('      value: 0.017\n') 
  txt_file.write('    }\n')
  txt_file.write('  }\n')
  txt_file.write('}\n')
  txt_file.write('\n')
  pass
# ----------------------------------------------------------------
def write_node(txt_file, info):
    if 'label' in info['name']:
        return        
    if info['op'] == 'null' and info['name'] == 'data':
        data(txt_file, info)
    elif info['op'] == 'Convolution':
        Convolution(txt_file, info)
    elif info['op'] == 'ChannelwiseConvolution':
        ChannelwiseConvolution(txt_file, info)
    elif info['op'] == 'BatchNorm':
        BatchNorm(txt_file, info)
    elif info['op'] == 'Activation':
        Activation(txt_file, info)
#    elif info['op'] == 'ElementWiseSum':
    elif info['op'] == 'elemwise_add':
        ElementWiseSum(txt_file, info)
    elif info['op'] == '_Plus':
        ElementWiseSum(txt_file, info)
    elif info['op'] == 'Concat':
        Concat(txt_file, info)
    elif info['op'] == 'Pooling':
        #Pooling(txt_file, info)
        Pooling_global(txt_file, info)
    elif info['op'] == 'Flatten':
        Flatten(txt_file, info)
    elif info['op'] == 'FullyConnected':
        FullyConnected(txt_file, info)
    elif info['op'] == 'SoftmaxActivation' or info['op'] == 'SoftmaxOutput':
        SoftmaxOutput(txt_file, info)
###
    elif info['op'] == 'Cast':
        Cast(txt_file, info)
    elif info['op'] == 'SliceChannel':
        SliceChannel(txt_file, info)
    elif info['op'] == 'L2Normalization':
        L2Normalization(txt_file, info)
    elif info['op'] == 'Reshape':
        #Reshape(txt_file,info)
        Flatten(txt_file,info)
    elif info['op'] == 'Dropout':
        DropOut(txt_file,info)
    elif info['op'] == "broadcast_add":
        broadcast_add(txt_file,info)
    elif info['op'] == '_mul_scalar':
        mulscalar(txt_file,info)
    else:
        #logging.warn("Unknown mxnet op: %s" %info['op'])
        sys.exit("Warning! Unknown mxnet op:{}".format(info['op']))





