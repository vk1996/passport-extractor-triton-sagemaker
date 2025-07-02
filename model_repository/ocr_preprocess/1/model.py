'''
Copyright 2025 Vignesh(VK)Kotteeswaran <iamvk888@gmail.com>
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''


import triton_python_backend_utils as pb_utils
import numpy as np
import cv2
import math



def img_decode(img):

    '''

      Converts byte array to numpy array

        Args:
            img(byte array)

        Returns:
            img (numpy array)



    '''
    img = np.frombuffer(img, dtype='uint8')
    img = cv2.imdecode(img, 1)
    # print(img.shape)

    return img






class TritonPythonModel:
    def initialize(self, args):
        pass



    def resize_norm_img(self,img):

        '''

            Args:

                img : numpy array


            Returns:
                returns preprocessed & normalized numpy array of image
        '''


        self.image_shape=[3,48,int(img.shape[1]*2)]


        imgC,imgH,imgW=self.image_shape


        max_wh_ratio = imgW * 1.0 / imgH
        h, w = img.shape[0], img.shape[1]
        ratio = w * 1.0 / h
        max_wh_ratio = min(max(max_wh_ratio, ratio), max_wh_ratio)
        imgW = int(imgH * max_wh_ratio)

        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = cv2.resize(img, (resized_w, imgH))

 
        self.show_frame=resized_image
        resized_image = resized_image.astype('float32')
        
        

        if self.image_shape[0] == 1:
            resized_image = resized_image / 255
            
            resized_image = resized_image[np.newaxis, :]
        else:
            
            resized_image = resized_image.transpose((2, 0, 1)) / 255

        
        
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image

        return padding_im



    def execute(self, requests):
        responses = []
        for request in requests:
            roi = pb_utils.get_input_tensor_by_name(request, "roi")
            roi = roi.as_numpy()
            preprocessed_image=np.expand_dims(self.resize_norm_img(roi),axis=0)
            preprocessed_image = pb_utils.Tensor("preprocessed_image", preprocessed_image)
            responses.append(pb_utils.InferenceResponse(output_tensors=[preprocessed_image]))
        return responses

    


    
        



