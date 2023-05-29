import numpy as np
import onnxruntime as ort


class backbone_onnx_wrapper:
    def __init__(self, model_path):
        """
        Args :

            model_path : path to the onnx file

        """
        print(f"path to model : {model_path}")
        self.ort_session = ort.InferenceSession(model_path)

    def __call__(self, img):
        """
        img : batchified numpy img with channel last convention
        """

        assert img.shape[0] == 1
        assert len(img.shape) == 4
        assert img.shape[-1] == 3

        img = np.transpose(img, (0, 3, 1, 2))  # onnx channel first convention

        outputs = self.ort_session.run(
            None,
            {"input.1": img.astype(np.float32)},
        )

        if len(outputs) > 1:
            print("warning : more than one output")
            return outputs[
                1
            ]  # return only the feature part (second part of the tuple output)
        return outputs[0]
