import numpy as np

def PreprocessInput(batch_image):   
	return (batch_image / 127.5) - 1

def DepreprocessInput(batch_image):
    return ((batch_image + 1) * 127.5).astype(np.uint8)

def AdvancedRoll(A, r):
    '''
    roll each row of matrix A independently accroading to r
    example:
        A = ndarray([[1, 2, 3, 4, 5],
                     [1, 2, 3, 4, 5],
                     [1, 2, 3 ,4 ,5]])
        r = ndarray([0, 2, -1])
        output = ndarray([[1, 2, 3, 4, 5],      # 1st row is right shifted by 0
                          [4, 5, 1, 2, 3],      # 2nd row is right shifted by 2
                          [2, 3, 4, 5, 1]])     # 3rd row is right shifted by -1
    '''
    rows, column_indices = np.ogrid[:A.shape[0], :A.shape[1]]    
    r[r < 0] += A.shape[1]
    column_indices = column_indices - r[:,np.newaxis]
    return A[rows, column_indices]
    
def GetStaticAngleWeight(sigma=4, size=360):
    '''
    returns a gaussian-like distribution, ndarray of shape (size, )
    '''
    locs = np.linspace(0, size-1, size)
    mask = locs>size//2
    locs[mask] = size - locs[mask]
    d = locs * locs
    e = 2 * sigma ** 2
    exponent = d / e
    weight = np.exp(-exponent)
    return weight

def MergeAngleWeights(angles, static_weight):
    '''
    inputs: 
        angles: ndarray of shape (n, ), n is in the range of [0, 3], ie. could be zero
        static_weight: a weight distribution centered at 0, ndarray of shape (360,) 
        Weight distribution is infinitesimal when approached to each angle, but 
        has value 1 at or far away from each angle
        
    outputs: a weight map centered at each angle, ndarray of shape (360, )
    '''
    if len(angles) == 0:
        return np.ones(shape=(360))
    
    weight_map = np.tile(static_weight, (len(angles), 1)) # (n, 360)
    weight_map = AdvancedRoll(weight_map, angles) # (n, 360)
    weight_map =  np.amax(weight_map, axis=0) # (360, )
    weight_map = 1.0 - weight_map
    weight_map[angles] = 1.0
    return weight_map


def _Test():
    w = GetStaticAngleWeight(4)
    angles = np.array([7, 92, 218, 345])
    p = MergeAngleWeights(angles, w)
    print(p)
    
class AngleManager(object):
    
    def __init__(self, sigma):
        self.static_angle_weight = GetStaticAngleWeight(sigma)
        
    def ComputeLabelAndWeigh(self, angle):
        '''
        inputs: ndarray of shape (n, ), n is in the range of [0, 3], ie. could be zero
        returns:
            weight: ndarray of shape (360, )
            label: ndarray of shape (360, )
        '''
        weight = MergeAngleWeights(angle, self.static_angle_weight)
        label = np.zeros((360))
        if len(angle) != 0: label[angle] = 1.0
        return label, weight
        
if __name__=='__main__':
    _Test()