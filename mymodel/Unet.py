import tensorflow as tf
import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM
import tensorflow.keras.backend as K
from tensorflow.keras.regularizers import l2
import cfg

def Convolution(x, kernal, out_dim, strides=1, reg=1e-4, name=''):
    padding = (kernal - 1) // 2
    x = KL.ZeroPadding2D(padding=padding, name=name+'.pad')(x)
    x = KL.Conv2D(out_dim, kernal, strides=strides, use_bias=False, kernel_regularizer=l2(reg), name=name+'.conv')(x)
    x = KL.BatchNormalization(epsilon=1e-5, name=name+'.bn')(x)
    x = KL.Activation('relu', name=name+'.relu')(x)
    return x

def Residual(x, dim_out, strides=1, reg=1e-4, name=''):
    shortcut = x
    dim_in = K.int_shape(shortcut)[-1]
    x = KL.ZeroPadding2D(padding=1, name=name + '.pad1')(x)
    x = KL.Conv2D(dim_out, 3, strides=strides, use_bias=False, kernel_regularizer=l2(reg), name=name+'.conv1')(x)
    x = KL.BatchNormalization(epsilon=1e-5, name=name + '.bn1')(x)
    x = KL.Activation('relu', name=name + '.relu1')(x)

    x = KL.Conv2D(dim_out, 3, padding='same', use_bias=False, kernel_regularizer=l2(reg), name=name+'.conv2')(x)
    x = KL.BatchNormalization(epsilon=1e-5, name=name + '.bn2')(x)

    if dim_in != dim_out or strides != 1:
        shortcut = KL.Conv2D(dim_out, 1, strides=strides, use_bias=False, kernel_regularizer=l2(reg), name=name+'.skip_conv')(shortcut)
        shortcut = KL.BatchNormalization(epsilon=1e-5, name=name + '.skip_bn')(shortcut)

    x = KL.Add(name=name + '.add')([x, shortcut])
    x = KL.Activation('relu', name=name + '.relu')(x)
    return x

def LeftFeatures(bottom):
    # create left half blocks for unet module
    features = [bottom]
    for count, dim in enumerate(cfg.dims):
        name = 'down'+str(count)       
        x = Residual(features[-1], dim, strides=2, reg=cfg.reg, name=name+'.res0')
        if not cfg.is_tiny:
            x = Residual(x, dim, reg=cfg.reg, name=name+'.res1')
        features.append(x)
    return features

def RightFeatures(leftfeatures):
    rf = BottleneckLayer(leftfeatures[-1], cfg.dims[-1])
    for idx in reversed(range(len(cfg.dims))):
        name = 'up'+str(idx) 
        rf = ConnectLeftRight(leftfeatures[idx], rf, cfg.dims[idx], cfg.dims[max(idx - 1, 0)], name=name)
    return rf

def ConnectLeftRight(left, right, num_channels, num_channels_next, name):
    
    left = Residual(left, num_channels_next, reg=cfg.reg, name=name+'.left_res0')
    # left = Residual(left, num_channels_next, reg=cfg.reg, name=name+'.left_res1')
    right = Residual(right, num_channels_next, reg=cfg.reg, name=name+'.right_res0')
    # right = Residual(right, num_channels_next, reg=cfg.reg, name=name+'.right_res1')
    right = KL.UpSampling2D(name=name+'.right_upsample')(right)
    out = KL.Add(name=name + '.add')([left, right])
    return out

def BottleneckLayer(x, num_channels):
    # 4 residual blocks with 512 channels in the middle
    name = 'bottleneck_'
    x = Residual(x, num_channels, reg=cfg.reg, name=name+str(0))
    if not cfg.is_tiny:
        x = Residual(x, num_channels, reg=cfg.reg, name=name+str(1))
        # x = Residual(x, num_channels, reg=cfg.reg, name=name+str(2))
        # x = Residual(x, num_channels, reg=cfg.reg, name=name+str(3))
    return x

def Head(x):
    name = 'head'
    shape = x.get_shape().as_list()
    x = KL.Conv2D(shape[-1], 3, use_bias=True, padding='same', name=name+'.conv', kernel_regularizer=l2(cfg.reg))(x)
    x = KL.Activation('relu', name=name+'.relu')(x)
    
    bias_init = tf.constant_initializer(value=-2.19)
    x = KL.Conv2D(360, 1, activation='sigmoid', bias_initializer=bias_init, name='prediction')(x)
    return x

def Unet(image):
    x = Convolution(image, 7, max(cfg.dims[0]//2, 64), strides=2, reg=cfg.reg, name='pre0')
    bottom = Residual(x, cfg.dims[0], strides=2, reg=cfg.reg, name='pre1')
    
    # create left features , f1, f2, f4, f8, f16 and f32
    lfs = LeftFeatures(bottom)
    # create right features, connect with left features
    x = RightFeatures(lfs)
    x = Convolution(x, 3, cfg.dims[0], strides=1, reg=cfg.reg, name='last_conv')

    pred = Head(x)
    return pred
 
def GetModel():
    image = KL.Input(shape=cfg.shape_info['image'], name='input_image')
    pred = Unet(image)
    label = KL.Input(shape=cfg.shape_info['label'], name='input_label')
    weight = KL.Input(shape=cfg.shape_info['weight'], name='input_weight')
    loss = KL.Lambda(lambda x: Loss(*x), name='loss')([label, pred, weight])
    inputs = {}
    inputs['image'] = image
    inputs['label'] = label
    inputs['weight'] = weight
    
    outputs = {}
    outputs['prediction'] = pred
    outputs['loss'] = loss

    model = KM.Model(inputs, outputs, name='Unet')
    return model
    
class WrapperModel(tf.keras.Model):
    # add customized losses and metrics
    def __init__(self, model, is_training=True, **kwargs):
        super(WrapperModel, self).__init__(**kwargs)
        self.is_training = is_training
        self.m = model

    def call(self, inputs):
        outputs = self.m(inputs)
        if self.is_training:
            # add losses and metrics
            loss = outputs['loss']
            self.add_loss(tf.reduce_mean(loss, name='loss'))
            # self.add_metric(tf.reduce_mean(loss), name='loss', aggregation='mean')
        return outputs
    
def Loss(label, pred, weight):
    '''
    params:
        label: (B, h, w, 360)
        pred: (B, h, w, 360)
        weight: (B, h, w, 360)
    '''
    alpha = 2.0
    beta = 4.0
    
    pos_inds = label
    neg_inds = 1.0 - label
    
    pos_loss = (-1) * K.log(K.clip(pred, 1e-4, 1)) * K.pow(1 - pred, alpha) * pos_inds
    neg_loss = (-1) * K.log(K.clip(1 - pred, 1e-4, 1)) * K.pow(pred, alpha) * neg_inds
    loss = (pos_loss + neg_loss) * K.pow(weight, beta) # (B, h, w, 360)

    axis = (1,2,3)
    loss = K.sum(loss, axis=axis) # (B, )
    count = K.sum(label, axis=axis) # (B, )
    count = tf.where(tf.less_equal(count, 1.0), 1.0, count) # (B, )
    return loss / count

    # return tf.where(tf.equal(count, 0.0), 0.0, total_loss/(count+1e-4)) # (B, )

if __name__=='__main__':
    # To Do: 维度问题
    import argparse
    opt = argparse.Namespace()
    opt.shape_image = (128, 128, 3)
    opt.shape_label = (32, 32, 360)
    opt.shape_weight = (32, 32, 360)
    opt.reg = 1e-2
    m = GetModel()
    w = WrapperModel(m)
    
