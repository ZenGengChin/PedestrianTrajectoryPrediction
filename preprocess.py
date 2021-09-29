import numpy as np
import pandas as pd
from torch import float64

from coordinate import coordConverter

class PTPSequenceParser(object):
    def __init__(self, filename, homo_path = None,isMeter = True) -> None:
        self.data = pd.read_csv(filename, sep='\s+', header=None).values
        self.data = np.array(sorted(self.data,key=lambda x:(x[1],x[0])),dtype='float')
        if isMeter:
            converter = coordConverter(homo_path)
            for i in self.data:
                i[2], i[4] = converter.meter2pix(i[2],i[4])
    def train_test_split(self, seed=0, seq_size = 10):
        batch_size = len(set(self.data[:,1]))
        vdata = np.zeros((batch_size, seq_size, 6))
        for i in range(0,batch_size):
            result = self.data[np.where(self.data[:,1] == list(set(self.data[:,1]))[i])]
            vdata[i,0:min(seq_size, len(result)),:] = result[0:min(seq_size, len(result)),[0,1,2,4,5,7]]
        np.random.seed(seed=seed)
        np.random.shuffle(vdata)
        train = vdata[0:(batch_size//2)]
        test = vdata[(batch_size//2):]
        return [train, test]