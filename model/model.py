"""
Copyright (c) 2019-present NAVER Corp.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch.nn as nn
from base.base_model import BaseModel

from model.transformation import TPS_SpatialTransformerNetwork
from model.feature_extraction import VGG_FeatureExtractor, RCNN_FeatureExtractor, ResNet_FeatureExtractor
from model.sequence_modeling import BidirectionalLSTM
from model.prediction import Attention
from base.base_model import BaseModel
# class CRNN2(BaseModel):
#
#     def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False):
#         self.FeatureExtraction = ResNet_FeatureExtractor(1, 256)
#         self.FeatureExtraction_output = 512  # int(imgH/16-1) * 512
#         self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1
#         self.SequenceModeling = nn.Sequential(
#             BidirectionalLSTM(512, nh, nh),
#             BidirectionalLSTM(nh, nh, nclass))
#         self.Prediction = Attention(nclass, nh, nclass)
#
#     def forward(self, input, text, is_train=True):
#         """ Feature extraction stage """
#         visual_feature = self.FeatureExtraction(input)
#         visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
#         visual_feature = visual_feature.squeeze(3)
#
#         contextual_feature = self.SequenceModeling(visual_feature)
#         prediction = self.Prediction(contextual_feature.contiguous(), text, is_train,
#                                      batch_max_length=self.opt.batch_max_length)
#
#         return prediction


class Model(BaseModel):

    def __init__(self, imgH, imgW, input_channel, num_fiducial, output_channel, hidden_size, num_class, batch_max_length):
        super(Model, self).__init__()

        # maximum-label-length
        self.batch_max_length = batch_max_length

        self.Transformation = TPS_SpatialTransformerNetwork(
            F=num_fiducial, I_size=(imgH, imgW), I_r_size=(imgH, imgW), I_channel_num=input_channel)

        self.FeatureExtraction = ResNet_FeatureExtractor(input_channel, output_channel)
        self.FeatureExtraction_output = output_channel  # int(imgH/16-1) * 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1

        """ Sequence modeling"""
        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(self.FeatureExtraction_output, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, num_class))
        # self.SequenceModeling_output = hidden_size
        #
        # self.Prediction = Attention(self.SequenceModeling_output, hidden_size, num_class)

    def forward(self, input, text, is_train=True):

        input = self.Transformation(input)
        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)

        contextual_feature = self.SequenceModeling(visual_feature)

        return contextual_feature
        # prediction = self.Prediction(contextual_feature.contiguous(), text, is_train, batch_max_length=self.batch_max_length)

        # return prediction