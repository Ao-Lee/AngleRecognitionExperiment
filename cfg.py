'''training options'''
lr = 1e-4
num_epoches = 300 # epoches to train
batch_per_replica = 24 # batch size per single GPU
size_train = 8000 # num of training data per epoch
size_validation = 100 # num of validation data per epoch
dir_results = '.' # root directory for saving results

'''data generation options'''
img_size = 128 # size of the generated image
output_stride = 4 # keep this value fixed to use the Unet 
assert img_size % output_stride == 0
min_angle_diff = 45 # minimum anglar difference between each two angles for a single point
angle_line_thickness = 2 # thickness of generated angle lines 
sigma = 4.5 # affect weight distribution for each angle
max_angle_num = 5 # maximum number of angles for a single point
prob_no_angle = 0.03 # probability of a generated point with no angle


'''model options'''
reg = 1e-5 # weight decay
is_tiny = True
factor = 0.125
dims = [256, 384, 384, 512]
dims = [int(dim * factor) for dim in dims]

'''shapes''' 
# shapes are infered from other options, this information is shared across the whole program
shape_info = {}
shape_info['image'] = (img_size, img_size, 3)
shape_info['weight'] = (img_size//output_stride, img_size//output_stride, 360)
shape_info['label'] = (img_size//output_stride, img_size//output_stride, 360)


