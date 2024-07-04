#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf


class HandwrittenClassifier(object):
    def __init__(
        self,
        model_path='model/handwritten_classifier/handwritten_classifier_v2.tflite',
        num_threads=1,
    ):
        self.interpreter = tf.lite.Interpreter(model_path=model_path,
                                               num_threads=num_threads)

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape']
    
    def __call__(
        self,
        image,
    ):
        assert image.shape == (self.input_shape[1], self.input_shape[2]), \
            f"Input image shape {image.shape} does not match expected shape {self.input_shape[1:3]}"
        input_details_tensor_index = self.input_details[0]['index']
        input_data = image.reshape(self.input_shape).astype(np.float32)

        self.interpreter.set_tensor(input_details_tensor_index, input_data)
        # self.interpreter.set_tensor(
        #     input_details_tensor_index,
        #     np.array([image], dtype=np.float32))
        self.interpreter.invoke()

        output_details_tensor_index = self.output_details[0]['index']

        result = self.interpreter.get_tensor(output_details_tensor_index)

        result_index = np.argmax(np.squeeze(result))

        return result_index
