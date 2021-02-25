# Copyright (c) Facebook, Inc. and its affiliates.

import argparse

import numpy as np
import requests
import torch
from mmf.common.registry import registry
from mmf.common.sample import Sample, SampleList
from mmf.utils.build import build_encoder, build_model, build_processors
from mmf.utils.checkpoint import load_pretrained_model
from mmf.utils.general import get_current_device
from omegaconf import OmegaConf
from PIL import Image


class Inference:
    def __init__(self):
        self.args = self.get_parser().parse_args()
        self.processor, self.feature_extractor, self.model = self._build_model()

    def get_parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--config", default=None, type=str, help="Inference config file"
        )
        return parser

    def _build_model(self):
        self.config = OmegaConf.load(self.args.config)
        processor = build_processors(self.config.processors)
        feature_extractor = build_encoder(self.config.features.image_feature_encodings)
        if self.config.model_config.pretrained:
            state_dict = load_pretrained_model(self.config.model_config.pretrained_path)
            # adding this line breaks the visualbert use case
            registry.register("config", state_dict["full_config"])
            model = build_model(self.config.model_config)
            model.load_state_dict(state_dict["checkpoint"])
        else:
            model = build_model(self.config.model_config)

        return processor, feature_extractor, model

    def forward(self, image_path, text, image_format="path"):
        text_output = self.processor["text_processor"](text)
        if image_format == "path":
            img = np.array(Image.open(image_path))
        elif image_format == "url":
            img = np.array(Image.open(requests.get(image_path, stream=True).raw))
        img = torch.as_tensor(img)
        max_detect = self.config.features.image_feature_encodings.params.max_detections

        if self.config.features.image_feature_encodings.type == "frcnn":

            image_preprocessed, sizes, scales_yx = self.processor["image_processor"](
                img
            )

            image_output = self.feature_extractor(
                image_preprocessed,
                sizes=sizes,
                scales_yx=scales_yx,
                padding=None,
                max_detections=max_detect,
                return_tensors="pt",
            )

            image_output = image_output["roi_features"][0]
        else:
            image_preprocessed = self.processor["image_processor"](img)
            image_output = self.feature_extractor(image_preprocessed)

        sample = Sample(text_output)
        sample.image_feature_0 = image_output

        sample_list = SampleList([sample])
        sample_list = sample_list.to(get_current_device())
        self.model = self.model.to(get_current_device())

        output = self.model(sample_list)

        answers = output["scores"].argmax(dim=1)

        answer = self.processor["answer_processor"].idx2word(answers[0])

        return answer


if __name__ == "__main__":
    inference = Inference()
    answer = inference.forward(
        "http://i.imgur.com/1IWZX69.jpg",
        {"text": "what type of shoes is the woman in pink wearing"},
        image_format="url",
    )
    print(answer)
