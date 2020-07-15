
import torch
import numpy as np
#权重读取类
class WeightReader():
    def __init__(self, weight_file):
        with open(weight_file, 'r') as fp:
            header = np.fromfile(fp, dtype = np.int32, count = 5)
            self.header = torch.from_numpy(header)
            self.seen = self.header[3]
            #The rest of the values are the weights
            #load them up
            self.weights = np.fromfile(fp, dtype = np.float32)
    #加载权重参数
    def load_weights(self, model):
        ptr = 0
        for _, block in model.blocks.items():
            for _, layer in block.layers.items():
                bn = layer.bnorm
                conv = layer.conv
                if bn is not None:
                    #Get the number of weights of Batch Norm Layer
                    num_bn_biases = bn.bias.numel()
                    #Load the data
                    #偏差
                    bn_biases = torch.from_numpy(self.weights[ptr:ptr + num_bn_biases])
                    ptr += num_bn_biases
                    #权重
                    bn_weights = torch.from_numpy(self.weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
                    #均值
                    bn_running_mean = torch.from_numpy(self.weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
                    #方差
                    bn_running_var = torch.from_numpy(self.weights[ptr: ptr + num_bn_biases])
                    ptr  += num_bn_biases
                    #Cast the loaded weights into dims of model weights.
                    bn_biases = bn_biases.view_as(bn.bias.data)
                    bn_weights = bn_weights.view_as(bn.weight.data)
                    bn_running_mean = bn_running_mean.view_as(bn.running_mean)
                    bn_running_var = bn_running_var.view_as(bn.running_var)
                    #Copy the data to model
                    bn.bias.data.copy_(bn_biases)
                    bn.weight.data.copy_(bn_weights)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)
                else:
                    #Number of biases
                    num_biases = conv.bias.numel()
                    #Load the biases
                    conv_biases = torch.from_numpy(self.weights[ptr: ptr + num_biases])
                    ptr = ptr + num_biases
                    #reshape the loaded weights according to the dims of the model weights
                    conv_biases = conv_biases.view_as(conv.bias.data)
                    #Finally copy the data
                    conv.bias.data.copy_(conv_biases)
                #load the weights for the Convolutional layers
                num_weights = conv.weight.numel()
                #Do the same as above for weights
                conv_weights = torch.from_numpy(self.weights[ptr:ptr+num_weights])
                ptr = ptr + num_weights
                conv_weights = conv_weights.view_as(conv.weight.data)
                conv.weight.data.copy_(conv_weights)
    #查看网络参数
    def weight_summary(self, model):
        train_able, train_disable = 0, 0
        for _, block in model.blocks.items():
            for _, layer in block.layers.items():
                bn = layer.bnorm
                conv = layer.conv
                if bn is not None:
                    train_able += (bn.bias.numel() + bn.weight.numel())
                    train_disable += (bn.running_mean.numel() + bn.running_var.numel())
                else:
                    train_able += conv.bias.numel()
                train_able += conv.weight.numel()
        print("total = %d"%(train_able + train_disable))
        print("count of train_able = %d"%train_able)
        print("count of train_disable = %d"%train_disable)
