
import torch
import caffe
from torch import nn
from torch.nn import functional as F

from utils import *
from prototxt_basic import *

def caffe_active(ofs,top,bottom,type):
  attr = {}
  attr["top"] = top
  attr["bottom"] = [bottom]
  attr["attrs"] = {}
  attr["attrs"]["act_type"] = type
  Activation(ofs,attr)

def caffe_normalization(ofs,top,bottom,eps):
  attr = {}
  attr["top"] = top
  attr["bottom"] = [bottom]
  attr["attrs"] = {}
  attr["attrs"]['eps'] = str(eps)
  BatchNorm(ofs,attr)

def caffe_global_avg_pool(ofs,top,bottom):
  attr = {}
  attr["top"] = top
  attr["bottom"] = [bottom]
  attr["attrs"] = {}
  attr["attrs"]["pool_type"] = "avg"
  attr["attrs"]["global_pool"] = "True"
  Pooling_global(ofs,attr)

def caffe_convolution(ofs,top,bottom,num_filter,kernel = 1,pad = 0,stride=1,bias=False,group = 1):
  attr = {}
  attr["top"] = top
  attr["bottom"] = [bottom]
  attr["attrs"] = {}
  attr["attrs"]["num_filter"] = str(num_filter)
  attr["attrs"]["kernel"] = "(" + str(kernel) + ",)"
  attr["attrs"]["pad"] = "(" + str(pad) + ",)"
  attr["attrs"]["stride"] = "(" + str(stride) + ",)"
  attr["attrs"]["num_group"] = str(group)
  attr["attrs"]["no_bias"] = str(not bias)
  Convolution(ofs,attr)

def caffe_BroadcastMul(ofs,top,bottom):
  attr = {}
  attr["top"] = top
  attr["bottom"] = bottom
  broadcast_mul(ofs,attr)

def caffe_dropout(ofs,top,bottom,drop_rate):
  attr = {}
  attr["top"] = top
  attr["bottom"] = [bottom]
  attr["attrs"] = {}
  attr["attrs"]["p"] = str(drop_rate)
  DropOut(ofs,attr)

def caffe_elementwise(ofs,top,bottom):
  attr = {}
  attr["top"] = top
  attr["bottom"] = bottom
  ElementWiseSum(ofs,attr)

def caffe_fc(ofs,top,bottom,num_hidden):
  attr = {}
  attr["top"] = top
  attr["bottom"] = [bottom]
  attr["attrs"] = {}
  attr["attrs"]["num_hidden"] = str(num_hidden)
  FullyConnected(ofs,attr)

def caffe_flatten(ofs,top,bottom):
  attr = {}
  attr["top"] = top
  attr["bottom"] = [bottom]
  Flatten(ofs,attr)

class MBConvBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block

    Args:
        block_args (namedtuple): BlockArgs, see above
        global_params (namedtuple): GlobalParam, see above

    Attributes:
        has_se (bool): Whether the block contains a Squeeze and Excitation layer.
    """
    #caffe_attr {layerId,bottom_name,ofs}
    def __init__(self, block_args, global_params,caffe_attr,insize,drop_rate = None):
        super(MBConvBlock,self).__init__()
        print(block_args)
        #print(caffe_attr)
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # skip connection and drop connect

        # Expansion phase
        outsize = insize
        inp = self._block_args.input_filters
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
        bsize = 6.0
        if self._block_args.expand_ratio != 1:
            caffe_convolution(caffe_attr[2],caffe_attr[0] + "._expand_conv",caffe_attr[1],oup)
            self._expand_conv = getConv2d(in_size = insize, in_channels=inp,out_size = outsize, out_channels=oup,kernel_size = 1,bias = False)
            #self._expand_conv = Conv2dSamePadding(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            caffe_normalization(caffe_attr[2],caffe_attr[0] + "._bn0",caffe_attr[0] + "._expand_conv",self._bn_eps)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
            caffe_active(caffe_attr[2],caffe_attr[0] + "._swish0",caffe_attr[0] + "._bn0","swish")
            self._swish0 = Swish()
            bsize = 8.0

        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        outsize = insize/s[0]

        if self._block_args.expand_ratio == 1:
            bottom = caffe_attr[1]
        else:
            bottom = caffe_attr[0] + "._swish0"

        caffe_convolution(caffe_attr[2],caffe_attr[0] + "._depthwise_conv",bottom,oup,k,pad = k/2,stride = s[0],group = oup)
        self._depthwise_conv = getConv2d(in_size = insize,in_channels=oup,out_size = outsize,out_channels=oup,groups=oup,kernel_size=k, stride=s[0], bias=False)
        caffe_normalization(caffe_attr[2],caffe_attr[0] + "._bn1",caffe_attr[0] + "._depthwise_conv",self._bn_eps)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        caffe_active(caffe_attr[2],caffe_attr[0] + "._swish1",caffe_attr[0] + "._bn1","swish")
        self._swish1 = Swish()
        # Squeeze and Excitation layer, if desired
        if self.has_se:
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            caffe_global_avg_pool(caffe_attr[2],caffe_attr[0] + "._global_avg_pool",caffe_attr[0] + "._swish1")
            self._global_avg_pool = torch.nn.AdaptiveAvgPool2d(1)
            caffe_convolution(caffe_attr[2],caffe_attr[0] + "._se_reduce",caffe_attr[0] + "._global_avg_pool",num_squeezed_channels,bias=True)
            self._se_reduce = getConv2d(in_size = 1,in_channels=oup,out_size = 1,out_channels=num_squeezed_channels, kernel_size=1)
            caffe_active(caffe_attr[2],caffe_attr[0] + "._swish2",caffe_attr[0] + "._se_reduce","swish")
            self._swish2 = Swish()
            caffe_convolution(caffe_attr[2],caffe_attr[0] + "._se_expand",caffe_attr[0] + "._swish2",oup,bias=True)
            self._se_expand = getConv2d(in_size = 1,in_channels=num_squeezed_channels,out_size = 1,out_channels=oup, kernel_size=1) 
            caffe_active(caffe_attr[2],caffe_attr[0] + "._sigmoid",caffe_attr[0] + "._se_expand","sigmoid")
            self._sigmoid = torch.nn.Sigmoid()
			
            caffe_BroadcastMul(caffe_attr[2],caffe_attr[0] + "._broadcast_mul",[caffe_attr[0] + "._swish1",caffe_attr[0] + "._sigmoid"])
            self._mult = BroadcastMul()
			#self._se_reduce = Conv2dSamePadding(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            #self._se_expand = Conv2dSamePadding(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Output phase
        final_oup = self._block_args.output_filters
        if self.has_se:
		    bottom = caffe_attr[0] + "._broadcast_mul"
        else:
            bottom = caffe_attr[0] + "._swish1"

        caffe_convolution(caffe_attr[2],caffe_attr[0] + "._project_conv",bottom,final_oup)
        self._project_conv = getConv2d(in_size = outsize,in_channels=oup,out_size = outsize,out_channels=final_oup, kernel_size=1, bias = False)
        #self._project_conv = Conv2dSamePadding(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        caffe_normalization(caffe_attr[2],caffe_attr[0] + "._bn2",caffe_attr[0] + "._project_conv",self._bn_eps)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)

        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        self.drop = drop_rate != None
        self.id_skip = self.id_skip and self._block_args.stride[0] == 1 and input_filters == output_filters
        print(self.drop,self.id_skip)
        bottom = caffe_attr[0] + "._bn2"
        self.bottom_name = bottom
        if self.drop and self.id_skip:
            caffe_dropout(caffe_attr[2],caffe_attr[0] + "._dropout",caffe_attr[0] + "._bn2",drop_rate/bsize)
            self._dropout = torch.nn.Dropout2d(drop_rate/bsize)
            bottom = caffe_attr[0] + "._dropout"

        if self.id_skip:
            caffe_elementwise(caffe_attr[2],caffe_attr[0] + "._shortcut",[caffe_attr[1],bottom])
            self._shortcut = AddTensor()
            self.bottom_name = caffe_attr[0] + "._shortcut"

        self.outsize = outsize

    def getOutSize(self):
        return self.outsize
    def getOutputName(self):
        return self.bottom_name

    def forward(self, inputs):
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._swish0(self._bn0(self._expand_conv(inputs)))
        x = self._swish1(self._bn1(self._depthwise_conv(x)))

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = self._global_avg_pool(x)#F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_expand(self._swish2(self._se_reduce(x_squeezed)))
            x_squeezed = self._sigmoid(x_squeezed)
            x = self._mult(x,x_squeezed)

        x = self._bn2(self._project_conv(x))
        if self.id_skip:
            if self.drop:
                x = self._dropout(x)
            x = self._shortcut(x,inputs)
        return x

class EfficientNet(nn.Module):
    """
    An EfficientNet model. Most easily loaded with the .from_name or .from_pretrained methods

    Args:
        blocks_args (list): A list of BlockArgs to construct blocks
        global_params (namedtuple): A set of GlobalParams shared between blocks

    Example:
        model = EfficientNet.from_pretrained('efficientnet-b0')

    """
    def __init__(self, blocks_args=None, global_params=None, model_name = "efficientnet-b0"):
        super(EfficientNet,self).__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        #print(blocks_args)
        self._global_params = global_params
        self._blocks_args = blocks_args

        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Stem
        insize = 224
        in_channels = 3  # rgb
        out_channels = round_filters(32, self._global_params)  # number of output channels

        file = open("caffemodel/" + model_name + ".prototxt","w")
        data(file,{})
        caffe_convolution(file,"_conv_stem","data",out_channels, 3, pad = 1 , stride = 2)
        self._conv_stem = getConv2d(insize,in_channels,insize/2,out_channels,kernel_size=3, stride=2, bias=False)#Conv2dSamePadding(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        caffe_normalization(file,"_bn0","_conv_stem",bn_eps)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        caffe_active(file,"_swish0","_bn0","swish")
        self._swish0 = Swish()
        # Build blocks
        insize = insize/2
        self._blocks = nn.ModuleList([])
        #print self._blocks_args
        last = "_swish0"
        for block_args in self._blocks_args:
            #print(block_args)
            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )

            # The first block needs to take care of stride and filter size increase.
            drop_rate = None
            if self._global_params.drop_connect_rate:
                drop_rate = self._global_params.drop_connect_rate*len(self._blocks)
            self._blocks.append(MBConvBlock(block_args, self._global_params,("_blocks." + str(len(self._blocks)),last,file),insize,drop_rate)) ####
            insize = self._blocks[-1].getOutSize()
            last = self._blocks[-1].getOutputName()
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=[1])
                #print(block_args)
            for _ in range(block_args.num_repeat - 1):
                drop_rate = None
                if self._global_params.drop_connect_rate:
                    drop_rate = self._global_params.drop_connect_rate * len(self._blocks)
                self._blocks.append(MBConvBlock(block_args, self._global_params,("_blocks." + str(len(self._blocks)),last,file),insize,drop_rate)) ###
                insize = self._blocks[-1].getOutSize()
                last = self._blocks[-1].getOutputName()
        # Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(1280, self._global_params)
        caffe_convolution(file,"_conv_head",last,out_channels)
        self._conv_head = getConv2d(insize,in_channels,insize,out_channels,kernel_size=1, bias=False)#Conv2dSamePadding(in_channels, out_channels, kernel_size=1, bias=False)
        caffe_normalization(file,"_bn1","_conv_head",bn_eps)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        caffe_active(file,"_swish1","_bn1","swish")
        self._swish1 = Swish()
        # Final linear layer
        caffe_global_avg_pool(file,"_global_avg_pool","_swish1")
        self._global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self._dropout = self._global_params.dropout_rate
        if self._dropout:
            caffe_dropout(file,"_dropout","_global_avg_pool",self._dropout)
            self._dropout = torch.nn.Dropout2d(self._dropout)
        caffe_flatten(file,"_flatten","_dropout")
        caffe_fc(file,"_fc","_flatten",self._global_params.num_classes)
        self._fc = nn.Linear(out_channels, self._global_params.num_classes)
        file.close()

    def extract_features(self, inputs):
        """ Returns output of the final convolution layer """
        # Stem
        x = self._swish0(self._bn0(self._conv_stem(inputs)))

        # Blocks
        for idx, block in enumerate(self._blocks):
            x = block(x)
        return x

    def forward(self, inputs):
        """ Calls extract_features to extract features, applies final linear layer, and returns logits. """

        # Convolution layers
        x = self.extract_features(inputs)

        # Head
        x = self._swish1(self._bn1(self._conv_head(x)))
        #x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        x = self._global_avg_pool(x).squeeze(-1).squeeze(-1)
        if self._dropout:
            x = self.dropout(x)
            #x = F.dropout(x, p=self._dropout, training=self.training)
        x = self._fc(x)
        return x

def get_from_name(model_name, override_params=None):
    #cls._check_model_name_is_valid(model_name)
    blocks_args, global_params = get_model_params(model_name, override_params)
    return EfficientNet(blocks_args, global_params,model_name)

def get_from_pretrained(model_name):
    model = EfficientNet.from_name(model_name)
    load_pretrained_weights(model, model_name)
    return model

def get_image_size(cls, model_name):
    cls._check_model_name_is_valid(model_name)
    _, _, res, _ = efficientnet_params(model_name)
    return res

def _check_model_name_is_valid(cls, model_name, also_need_pretrained_weights=False):
    """ Validates model name. None that pretrained weights are only available for
    the first four models (efficientnet-b{i} for i in 0,1,2,3) at the moment. """
    num_models = 4 if also_need_pretrained_weights else 8
    valid_models = ['efficientnet_b'+str(i) for i in range(num_models)]
    if model_name.replace('-','_') not in valid_models:
        raise ValueError('model_name should be one of: ' + ', '.join(valid_models))
