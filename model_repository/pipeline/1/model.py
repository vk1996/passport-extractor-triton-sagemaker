import triton_python_backend_utils as pb_utils
import tritonclient.http as httpclient
import numpy as np
import cv2





class TritonPythonModel:
    def initialize(self, args):
        self.client = httpclient.InferenceServerClient(url="localhost:8080")

    def set_input(self,input_name, obj, datatype="FP32"):
        input_holder = httpclient.InferInput(input_name, obj.shape, datatype=datatype)
        input_holder.set_data_from_numpy(obj, binary_data=True)
        return input_holder
    @staticmethod
    def split_mrz(mrz_roi):
        h, w, _ = mrz_roi.shape
        split_h = h // 2
        return [mrz_roi[:split_h, :, :], mrz_roi[split_h:, :, :]]

    def detect_passport(self,raw_image):
        preprocess_input = self.set_input("raw_image", raw_image)
        preprocess_response = self.client.infer(model_name="passport_preprocess", inputs=[preprocess_input])
        preprocessed_image = preprocess_response.as_numpy("preprocessed_image")
        resize_ratio = np.array([preprocess_response.as_numpy("resize_ratio")])

        infer_input = self.set_input("images", preprocessed_image)
        infer_response = self.client.infer(model_name="passport_inference", inputs=[infer_input])
        predictions = infer_response.as_numpy("output")

        predictions_input = self.set_input("predictions", predictions)
        resize_ratio_input = self.set_input("resize_ratio", resize_ratio)
        postprocess_response = self.client.infer(model_name="passport_postprocess",
                                            inputs=[predictions_input, preprocess_input, resize_ratio_input])

        out_img = postprocess_response.as_numpy("output_image").astype(np.uint8)
        boxes = postprocess_response.as_numpy("boxes")
        scores = postprocess_response.as_numpy("scores")
        classes = postprocess_response.as_numpy("classes")
        classes_names = postprocess_response.as_numpy("classes_names")

        return out_img, boxes, scores, classes, classes_names

    def detect_ocr(self,roi):
        preprocess_input = self.set_input("roi", roi)
        preprocess_response = self.client.infer(model_name="ocr_preprocess", inputs=[preprocess_input])
        preprocessed_image = preprocess_response.as_numpy("preprocessed_image")

        infer_input = self.set_input("x", preprocessed_image)
        infer_response = self.client.infer(model_name="ocr_inference", inputs=[infer_input])
        predictions = infer_response.as_numpy("softmax_2.tmp_0")[0]
        # print(predictions.shape)
        predictions_input = self.set_input("predictions", predictions)
        postprocess_response = self.client.infer(model_name="ocr_postprocess", inputs=[predictions_input])
        output_text = postprocess_response.as_numpy("output_text").tolist().decode()
        return output_text

    def execute(self, requests):
        responses = []
        for request in requests:
            raw_image = pb_utils.get_input_tensor_by_name(request, "raw_image")
            raw_image = raw_image.as_numpy()
            src_img = raw_image.copy()

            _, boxes, scores, classes, classes_names = self.detect_passport(raw_image)

            result = {}
            for box, score, label, class_name in zip(boxes, scores, classes, classes_names):
                box = [int(i) for i in box]
                roi = src_img[box[1]:box[3], box[0]:box[2]]
                class_name = class_name.decode()
                #print(class_name)
                if class_name == "photo":
                    result[class_name] = roi.tolist()
                if class_name == "mrz":
                    texts = []
                    for split_roi in TritonPythonModel.split_mrz(roi):
                        texts.append(self.detect_ocr(split_roi))
                    texts = "\n".join(texts)
                    result[class_name] = texts

            out_tensors = [
                pb_utils.Tensor("photo", np.array(result["photo"]).astype(np.uint8)),
                pb_utils.Tensor("mrz", np.array(result["mrz"]).astype(object))  # STRING type
            ]

            responses.append(pb_utils.InferenceResponse(output_tensors=out_tensors))

        return responses











# import tritonclient.http as httpclient
# client = httpclient.InferenceServerClient(url="localhost:8000")
# import numpy as np
# import cv2
# from matplotlib import pyplot as plt
#
#
# def set_input(input_name,obj,datatype="FP32"):
#     input_holder = httpclient.InferInput(input_name, obj.shape, datatype=datatype)
#     input_holder.set_data_from_numpy(obj, binary_data=True)
#     return input_holder
#
# raw_image=cv2.imread("/models/test.jpg").astype(np.float32)
# src_img=raw_image.copy()
# inputs=set_input("raw_image",raw_image)
# response = client.infer(model_name="pipeline", inputs=[inputs])
# print(response.as_numpy("mrz").tolist().decode())
# print(response.as_numpy("photo").shape)

# curl -s -X POST http://localhost:8080/v2/models/pipeline/infer -H "Content-Type: application/json" -d @input.json | jq -r '.outputs[] | select(.name=="mrz") | .data[0]'