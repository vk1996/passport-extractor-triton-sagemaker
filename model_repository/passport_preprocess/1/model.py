import triton_python_backend_utils as pb_utils
import numpy as np
import cv2

class TritonPythonModel:
    def initialize(self, args):
        self.image_size=(416,416)


    def __preprocess_image(self, img, swap=(2, 0, 1)):
        padded_img = np.ones((self.image_size[0], self.image_size[1], 3), dtype=np.uint8) * 114
        r = min(self.image_size[0] / img.shape[0], self.image_size[1] / img.shape[1])
        resized_img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)), interpolation=cv2.INTER_LINEAR).astype(np.uint8)
        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
        padded_img = padded_img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img, r

    def execute(self, requests):
        responses = []
        for request in requests:
            raw_image = pb_utils.get_input_tensor_by_name(request, "raw_image")
            raw_image = raw_image.as_numpy()

            preprocessed_image, resize_ratio = self.__preprocess_image(raw_image)
            preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
            # print("resize ratio:",resize_ratio,type(resize_ratio))
            preprocessed_image = pb_utils.Tensor("preprocessed_image",preprocessed_image)
            resize_ratio = pb_utils.Tensor("resize_ratio",np.array(resize_ratio,dtype=np.float32))
            responses.append(pb_utils.InferenceResponse(output_tensors=[preprocessed_image,resize_ratio]))
        return responses
