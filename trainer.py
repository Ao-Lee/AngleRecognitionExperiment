import os
import tensorflow as tf
from tensorflow.keras import callbacks as KC
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam

from mymodel.Unet import GetModel, WrapperModel
from mydata.dataset import MyDataset
from utils import InitDirs, ConfigDevices, GetSchedular, MyLogger
import cfg

def Train(devices='1, 2, 3'):
    num_gpus = len(devices.split(','))
    batch = num_gpus * cfg.batch_per_replica
    ConfigDevices(devices)
    
    '''data'''
    ds = MyDataset(batch)
    ds_tr, ds_val = ds.GetDataset()
    '''model'''
    is_training = True
    
    if num_gpus > 1:
        strategy = tf.distribute.MirroredStrategy()
        assert num_gpus == strategy.num_replicas_in_sync
        with strategy.scope():
            m = GetModel()
            model = WrapperModel(m, is_training=is_training)
            model.compile(optimizer=Adam(lr=cfg.lr))
    
    else:
        m = GetModel()
        model = WrapperModel(m, is_training=is_training)
        model.compile(optimizer=Adam(lr=cfg.lr))
        
    for data in ds_tr.take(1): # build model
        _ = model(data)
    
    metrics_names = ['val_' + name for name in model.metrics_names]
    metrics_names += model.metrics_names
    
    nickname = 'Card' + ''.join(c for c in devices if c.isdigit())
    dir_ckpt, dir_pics, dir_summary, dir_log = InitDirs(dir_results=cfg.dir_results, nickname=nickname)

    Schedular = GetSchedular(lr_base=cfg.lr, epoch_total=cfg.num_epoches)
    reduce_lr = KC.LearningRateScheduler(Schedular)
    early_stopping = KC.EarlyStopping(monitor='loss', patience=30, verbose=1)
    tb = KC.TensorBoard(log_dir=dir_log, histogram_freq=1)
    logger = MyLogger(path_root=dir_summary, opt=cfg, metrics_names=metrics_names)
    ckpt = KC.ModelCheckpoint(monitor='val_loss', 
                              filepath=os.path.join(dir_ckpt, 'weights.h5'), 
                              save_weights_only=True, save_best_only=True, mode='min')
    
    callbacks = [reduce_lr, early_stopping, tb, logger, ckpt]
    
    result = model.fit(ds_tr,
                       steps_per_epoch = cfg.size_train // batch,
                       epochs = cfg.num_epoches,
                       verbose = 1,
                       callbacks = callbacks,
                       validation_data = ds_val,
                       validation_steps = cfg.size_validation // batch)

    return model, result.history

if __name__=='__main__':
    print('tf version:\t{}'.format(tf.__version__))
    K.clear_session()
    
    devices='1, 2, 3'
    Train(devices)
        
        
   