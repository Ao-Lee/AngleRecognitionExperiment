'''
provide functionality for graphic & visualization
'''
import matplotlib.pyplot as plt
import numpy as np
import cv2

class Shower(object):
    def __init__(self, visualizers, figsize=4):
        self.visualizers = visualizers
        self.figsize = (figsize * len(visualizers), figsize)
    
    def Show(self):
        fig, axs = plt.subplots(1, len(self.visualizers), figsize=self.figsize)
        for idx in range(len(self.visualizers)):
            visualizer = self.visualizers[idx]
            axs[idx] = visualizer.Plot(axs[idx])
            axs[idx].set_title(visualizer.title)
            axs[idx].axis('off')

        
class VizBase(object):
    def __init__(self, title=''):
        self.title = title
        
    def Plot(self, ax):
        ax.set_title(self.title)
        return ax

class VizImage(VizBase):
    def __init__(self, image, **kwargs):
        super().__init__(**kwargs)
        self.image = image.astype(np.uint8)
        
    def Plot(self, ax):
        if len(self.image.shape) == 3:
            ax.imshow(self.image)
        else:
            ax.imshow(self.image, cmap=plt.cm.gray)
        return ax
    
class VizAngle(VizBase):
    def __init__(self, probs, output_stride=1, threshold=0.5, **kwargs):
        super().__init__(**kwargs)
        
        probs[probs < threshold] = 0 
        self.probs = probs
        self.interval = 360 // len(probs)
        self.degrees = np.arange(len(probs)) * self.interval + self.interval // 2
        
    def Plot(self, ax):
        length = 0.4
        head_width = 0.04   
        head_length = 0.04
        
        for degree, prob in zip(self.degrees, self.probs):
            if prob == 0: continue
            radian = (degree/180)*np.pi
            delta_x = np.cos(radian) * length * prob
            # delta_y = np.sin(radian) * length * (-1) * prob
            delta_y = np.sin(radian) * length * prob
            ax.arrow(0.5, 0.5, delta_x, delta_y, head_width=head_width, head_length=head_length, fc='c', ec='c')
        return ax
    

# if u got a list of imgs with the same size, and u wanna show them together in one shot, here is what u got
def MergeImage(imgs, how='auto', color=(40,40,40), margin='auto', min_size=600):

    assert how in ['vertical', 'horizontal', 'auto']
    num = len(imgs)
    assert num >= 1
    h = imgs[0].shape[0]
    w = imgs[0].shape[1]

    for img in imgs:
        assert img.shape == (h, w, 3)

    if how == 'auto':
        how = 'horizontal' if h < w else 'vertical'
    color = np.array(color,dtype=np.uint8)
    if margin == 'auto':
        margin = min(h, w)//20

    
    new_h = h + margin*2 if how=='horizontal' else h*num + margin*(num+1)
    new_w = w + margin*2 if how=='vertical' else w*num + margin*(num+1)
    
    new_img = np.zeros([new_h, new_w, 3], dtype=np.uint8)
    new_img[:,:,:] = color

    for i, img in enumerate(imgs):
        if how == 'horizontal':
            start = margin*(i+1) + w*i
            end = margin*(i+1) + w*(i+1)
            new_img[margin:margin+h, start:end, :] = img

        if how == 'vertical':
            start = margin*(i+1) + h*i
            end = margin*(i+1) + h*(i+1)
            new_img[start:end, margin:margin+w, :] = img
        
    size = min(new_w, new_h)
    ratio = 1 if size<= min_size else min_size/size
    new_w = int(new_w*ratio)
    new_h = int(new_h*ratio)
    new_img = cv2.resize(new_img, (new_w, new_h))
    return new_img

def VisualizeLabelPoints(label, image=None, figsize=(20,20)):
    points = np.argwhere(label==1)
    plt.figure(figsize=figsize)
    if image is None:
        plt.imshow(label)
    else:
        plt.imshow(image.astype('uint8'))
        
    plt.scatter(x=points[:, 1], y=points[:, 0], c='#1ec8ea', s=60)
    plt.axis('off')
    plt.show()

'''
inputs:
	list_imgs:
		a list of batch of images, for example [img1, img2, ... ,imgn]
		each img in the list has shape of (batch, h, w, c)
	list_names:
		a list of strings [img1_title, img2_title, ... ,imgn_title]
	return:
		plot a figure with following structure:
		-------------------------------------------------
		|img1_title     img2_title      imgn_title      |
		|-----------------------------------------------|
		|img1[batch0]   img2[batch0]    imgn[batch0]    |
		|-----------------------------------------------|
		|img1[batch1]   img2[batch1]    imgn[batch1]    |
		-------------------------------------------------
'''
def PlotResults(list_imgs, list_names, figsize=8, file_path=None):
    assert len(list_imgs)==len(list_names)
    num_imgs = len(list_imgs)
    num_batch = list_imgs[0].shape[0]
    fig, axs = plt.subplots(num_batch, num_imgs, figsize=(figsize*num_imgs,figsize*num_batch))
    for batch_id in range(num_batch):
        for img_id in range(num_imgs):
            axs[batch_id, img_id].imshow(list_imgs[img_id][batch_id])
            axs[batch_id, img_id].set_title(list_names[img_id])
            axs[batch_id, img_id].axis('off')
	# return fig
    if file_path is not None: fig.savefig(file_path)
    plt.close()
    
if __name__=='__main__':
    pass
    
    
    
    