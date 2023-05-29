import sys
import os
import json
import numpy as np
from typing import Union
from pynq import Overlay

file_dir = os.path.dirname(__file__)
print(file_dir)
sys.path.append(
    file_dir
)  # add current path to python path because import of tcu_pynq is relative

# import using the same syntax as in the driver in order to avoid mixing repo if sevral folders are present
from tcu_pynq.driver import Driver
from tcu_pynq.architecture import Architecture
from tcu_pynq.data_type import DataType


class BackboneTensilWrapper:
    def __init__(
        self,
        overlay: Overlay,
        path_tmodel: Union[str, os.PathLike],
        onnx_output_name: str,
        debug=False,
    ):
        """
        Args :
            - path_bit : path qui mêne au bitstream, e.g : home/xilinx/bitstream.bit
            - path_tmodel : path qui mène aui tmodel, e.g : home/xilinx/model.tmodel
        """
        print(f"dma 0 : {overlay.axi_dma_0}")

        if not hasattr(overlay, "axi_dma_0"):
            raise RuntimeError("DMA was not found in overlay")
        with open(path_tmodel) as f:
            js = json.load(f)
            arch = js["arch"]
            arch["data_type"] = DataType[arch["data_type"]]
            self.tarch = Architecture(**arch)

        self.tcu = Driver(self.tarch, overlay.axi_dma_0, debug=debug)
        print("tcu succefullt loaded")
        self.tcu.load_model(path_tmodel)
        assert self.tcu.arch.array_size >= 3, "array size must be >=3"
        self.onnx_output_name = onnx_output_name

    def __call__(self, single_image_batch: np.ndarray):
        assert len(single_image_batch.shape) == 4, "single image is not a batch"
        assert (
            single_image_batch.shape[0] == 1
        ), "img is supposed to be a batch of size one"
        assert single_image_batch.shape[-1] == 3, "last channel is not a rgb image"

        img = single_image_batch[0]

        img = np.pad(
            img,
            [(0, 0), (0, 0), (0, self.tcu.arch.array_size - 3)],  # asuming rgb
            "constant",
            constant_values=0,
        )
        img = img.reshape((-1, self.tcu.arch.array_size))

        inputs = {"input.1": img}
        outputs = self.tcu.run(inputs)

        return outputs[self.onnx_output_name][None, :]
