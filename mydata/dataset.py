import numpy as np
import tensorflow as tf
from functools import partialmethod
from mydata.utils import PreprocessInput, AngleManager
from mydata.single_gen import GenImageMaskLabel
import cfg

class MyDataset(object):
    def __init__(self, batch_size,
                 number_batch_tr = None,
                 number_batch_val = None):
        self.batch_size = batch_size
        self.number_batch_tr = number_batch_tr
        self.number_batch_val = number_batch_val
        self.manager = AngleManager(cfg.sigma)
         
    def _GetGenerator(self):
        while True:
            image, mask, angles = GenImageMaskLabel(size = cfg.img_size,
                                                    thickness = cfg.angle_line_thickness,
                                                    min_angle_diff = cfg.min_angle_diff,
                                                    max_angle_num = cfg.max_angle_num,
                                                    prob_no_angle = cfg.prob_no_angle,
                                                    )
            
            image = PreprocessInput(image) # image is ndarray, not PIL image
            label, weight = self.manager.ComputeLabelAndWeigh(angles)
            
            label = label.reshape(1, 1, -1) # (1, 1, 36)
            label = label * mask # (size//4, size//4, 36)
            
            weight = weight.reshape(1, 1, -1) # (1, 1, 36)
            weight = weight * mask # (size//4, size//4, 36)
            
            result = {}
            result['image'] = image # (size, size, 3)
            result['label'] = label # (size//4, size//4, 36)
            result['weight'] = weight # (size//4, size//4, 36)
            yield result
            
    _GetG = partialmethod(_GetGenerator)
    
    def _FnGenTr(self):
        return self._GetG()
    
    def _FnGenVal(self):
        return self._GetG()
    
    def GetDataset(self, infinite=True):
        '''
        if infinite is true, the virtual datasize is infinite, the dataset 
        iterates forever. Set to True if you are using keras default training 
        API, otherwize the iteration will run out eventually. In user-defined 
        training process, it is convenient to set to False.
        '''
        dict_shape = cfg.shape_info
        dict_type = {tag:tf.float32 for tag in dict_shape}
        dict_tensorshape = {tag:tf.TensorShape(dict_shape[tag]) for tag in dict_shape}

        # note _FnGenTr() is not a generator, instead, this is a function which returns a generator
        ds_tr = tf.data.Dataset.from_generator(self._FnGenTr, dict_type, dict_tensorshape)
        ds_val = tf.data.Dataset.from_generator(self._FnGenVal, dict_type, dict_tensorshape)
    
        ds_tr = ds_tr.batch(self.batch_size)
        ds_val = ds_val.batch(self.batch_size)
        
        if self.number_batch_tr is None or self.number_batch_val is None:
            infinite = True

        if not infinite:
            ds_tr = ds_tr.take(self.number_batch_tr)
            ds_val = ds_val.take(self.number_batch_val)
        
        # prefetch batches
        ds_tr = ds_tr.prefetch(10)
        ds_val = ds_val.prefetch(10)
        
        return ds_tr, ds_val
    
if __name__=='__main__':
    from viz import VizImage, Shower, VizAngle
    # from mydata.utils import DepreprocessInput
    
    batch = 1
    ds = MyDataset(batch_size = batch)
    ds_tr, ds_val = ds.GetDataset()  
    
    for data in ds_tr.take(1):
        for idx in range(batch):
            image = data['image'][idx, ...].numpy() 
            weight = data['weight'][idx, ...].numpy()
            label = data['label'][idx, ...].numpy()
            pts = np.sum(weight, axis=-1)
            pts[pts!=0] = 1
            
            ys, xs = np.nonzero(pts)
            weight = weight[ys[0], xs[0], :]
            weight = 1 - weight
            label = label[ys[0], xs[0], :]
            
            show_list = []
            show_list.append(VizImage(image, title='image'))
            # show_list.append(VizImage(pts, title='points'))
            show_list.append(VizAngle(weight, title='weight'))
            show_list.append(VizAngle(label, title='label'))
            
            shower = Shower(show_list)
            shower.Show()
            

            
            
           