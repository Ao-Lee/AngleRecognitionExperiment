import tensorflow as tf
import numpy as np
import datetime
import os
from functools import partial

class MyLogger(tf.keras.callbacks.Callback):
    def __init__(self, path_root, opt, metrics_names, **kwargs):
        super().__init__(**kwargs)
        self.path_logfile = os.path.join(path_root, 'history.txt')
        metrics = ['lr', 'loss', 'val_loss']
        metrics += metrics_names
        self.history = {metric:[] for metric in metrics}
        self.opt = opt
        
    def on_epoch_end(self, epoch, logs=None):
        for metric in self.history.keys():
            if metric not in logs.keys(): continue
            cut_digit = 7 if metric == 'lr' else 3
            result = round(logs[metric], cut_digit)
            self.history[metric].append(result)
            
        
        with open(self.path_logfile, 'w') as file:
            self.log_options(file)
        
            
        with open(self.path_logfile, 'a') as file:
            self.log_training(file)
           
    def log_options(self, file):
        print('--------------- train params ---------------', file=file)
        for name in vars(self.opt):
            if name.startswith('__'): continue
            print('{}:{}'.format(name, getattr(self.opt, name)), file=file)
            
    def log_training(self, file):
        print('--------------- training log ---------------', file=file)
        for metric in self.history.keys():
            print('{}: {}'.format(metric, self.history[metric]), file=file)    
            
def PreprocessInput(batch_image):   
	return (batch_image / 127.5) - 1

def DepreprocessInput(batch_image):
    return ((batch_image + 1) * 127.5).astype(np.uint8)

def GetSchedular(lr_base=0.001, epoch_total=200):
    def _Scheduler(epoch_current, lr_current, plan):
        return plan[epoch_current]
    current_epoch = np.arange(epoch_total)
    plan = lr_base * (1 - current_epoch / epoch_total) ** 0.9
    Schedular = partial(_Scheduler, plan=plan)
    return Schedular

def ConfigDevices(devices='0, 1'):
    os.environ["CUDA_VISIBLE_DEVICES"] = devices
    tf.config.set_soft_device_placement(True)
    devices = tf.config.list_physical_devices('GPU')
    for device in devices:
        # print(device)
        try: 
            tf.config.experimental.set_memory_growth(device, True)
        except:
            print('Invalid device or cannot modify virtual devices once initialized')

def InitDirs(dir_results='results', nickname=''):
    '''
    Initialize all the folders for recording results
    the structure of the folders:
    --dir_results
    ----dir_models 	 	# dir for saving model check points
    ----dir_pics 	 	# dir for saving generated figures during training
    ----dir_summary 	# dir of tf.summary for tracing loss values during training
    '''
    time = datetime.datetime.now().strftime('%Y-%m-%d-%H%M')
    time = time + '-' + nickname
    dir_results = os.path.join(dir_results, time)
    dir_ckpt = os.path.join(dir_results, 'ckpt')
    dir_pics = os.path.join(dir_results, 'pics')
    dir_summary = os.path.join(dir_results, 'summary')
    dir_log = os.path.join(dir_results, 'log')
    
    def _CreatePath(path):
        if not os.path.exists(path): os.makedirs(path)

    _CreatePath(dir_ckpt)
    _CreatePath(dir_pics)
    _CreatePath(dir_summary)
    _CreatePath(dir_log)
    return dir_ckpt, dir_pics, dir_summary, dir_log