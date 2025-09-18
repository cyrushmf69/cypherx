from vllm.multimodal.processing import (
    BaseMultiModalProcessor,
    BaseProcessingInfo,
    BaseDummyInputsBuilder,
    MULTIMODAL_REGISTRY
)
from transformers import AutoProcessor, AutoTokenizer


class BeastProcessingInfo(BaseProcessingInfo):
    IMAGE_KEY = "image"
    TEXT_KEY = "text"


class BeastDummyInputsBuilder(BaseDummyInputsBuilder):
    def build(self, processing_info: BeastProcessingInfo):
        import torch
        dummy_image = torch.zeros((1, 3, 224, 224), dtype=torch.float32)
        dummy_text = "dummy text"
        return {processing_info.IMAGE_KEY: dummy_image, processing_info.TEXT_KEY: dummy_text}


@MULTIMODAL_REGISTRY.register_processor(name="beast_processor")
class BeastProcessor(BaseMultiModalProcessor):
    def __init__(self, model_name="Salesforce/blip-image-captioning-base"):
        super().__init__()
        self.hf_processor = AutoProcessor.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def process(self, processing_info: BeastProcessingInfo, inputs: dict):
        return self.hf_processor(
            text=inputs.get(processing_info.TEXT_KEY),
            images=inputs.get(processing_info.IMAGE_KEY),
            return_tensors="pt",
            padding=True
        )

    def process_dummy_inputs(self, processing_info: BeastProcessingInfo):
        return BeastDummyInputsBuilder().build(processing_info)
