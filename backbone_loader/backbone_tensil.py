
import numpy as np
from pynq import Overlay
from tcu_pynq.driver import Driver
from tcu_pynq.architecture import pynqz1

class backbone_tensil_wrapper:

    def __init__(self,overlay,path_tmodel):
        """
        Args :
            - path_bit : path qui mêne au bitstream, e.g : home/xilinx/bitstream.bit
            - path_tmodel : path qui mène aui tmodel, e.g : home/xilinx/model.tmodel
        """
        print(f"dma 0 : {overlay.axi_dma_0}")
        self.tcu = Driver(pynqz1, overlay.axi_dma_0)
        print("tcu succefullt loaded")
        self.tcu.load_model(path_tmodel)

    def __call__(self,img):
        assert img.shape[0]==1
        assert len(img.shape)==4
        assert img.shape[-1]==3

        img=img[0]
        c,h,w=img.shape

        
        #img=np.transpose(img.reshape((c, h, w)), axes=[1, 2, 0])
        img=np.pad(img, [(0, 0), (0, 0), (0, self.tcu.arch.array_size - 3)], 'constant', constant_values=0)
        img=img.reshape((-1, self.tcu.arch.array_size))

        inputs = {'input.1': img}
        outputs = self.tcu.run(inputs)
        return  outputs['Output'][None,:]
    