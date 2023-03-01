import numpy as np
import cv2
import math
  
def GenImageMaskLabel(size = 128, 
            background = (0, 0, 0),
            color = (255, 255, 255),
            thickness = 2,
            min_angle_diff = 60,
            output_stride = 4,
            max_angle_num = 4,
            prob_no_angle = 0.05
            ):
    
    assert size % output_stride == 0
    image = np.zeros(shape=(size, size, 3))
    mask = np.zeros(shape=(size//output_stride, size//output_stride, 1))
    
    cx, cy = _GetCenter(size)
    mask[cy//output_stride, cx//output_stride, :] = 1.0
    cv2.line(image, (cx, cy), (cx, cy), color, thickness=thickness)
    angles = _RandomAngles(min_angle_diff=min_angle_diff, max_angle_num=max_angle_num, prob_no_angle=prob_no_angle)
    for angle in angles:
        radian = angle * np.pi / 180
        radius = math.floor(size/2)
        radius = _RandomRadius(radius)
        tx = int(radius * np.cos(radian) + cx)
        ty = int(radius * (-1) * np.sin(radian) + cy)
        cv2.line(image, (cx, cy), (tx, ty), color, thickness=thickness)

    return image, mask, angles

def _GetCenter(image_size):
    c = math.ceil(image_size/2)
    return c, c

def _RandomRadius(radius, low_ratio=0.2, high_ratio=1.0):
    rand = np.random.random()
    scale = rand * (high_ratio - low_ratio) + low_ratio
    return radius * scale
    
def _RandomAngles(min_angle_diff=60, max_angle_num=3, prob_no_angle=0.05, degrees=360):
    pool = np.arange(degrees)
    cases = np.arange(max_angle_num + 1)
    p = 1 / np.arange(max_angle_num, 0, -1)
    p = p * (1 - prob_no_angle) / np.sum(p)
    p = [prob_no_angle] + list(p)
    num_angles = np.random.choice(cases, p=p)

    angles = []
    for _ in range(num_angles):
        if len(pool) == 0: break
        angle = np.random.choice(pool)
        angles.append(angle)
        interval = np.arange(angle-min_angle_diff, angle+min_angle_diff)
        interval[interval<0]+=degrees
        interval[interval>=degrees]-=degrees
        pool = np.setdiff1d(pool, interval, assume_unique=True)
        
    return np.array(angles)

if __name__=='__main__':
    from viz import VizImage, Shower
    image, mask, angles = GenImageMaskLabel()
    viz1 = VizImage(image)
    viz2 = VizImage(mask[..., 0])
    s1 = Shower([viz1])
    s1.Show()
    s2 = Shower([viz2])
    s2.Show()
    print(angles)
    

