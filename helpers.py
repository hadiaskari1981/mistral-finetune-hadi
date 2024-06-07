import json
import logging
import time

logger = logging.getLogger(__name__)


def get_quantization_config():
    try:
        # import bitsandbytes
        import torch
        from transformers import BitsAndBytesConfig

        assert bitsandbytes

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    except Exception:
        logger.warning('Failed to initialize bitsandbytes config, not quantizing', exc_info=True)
        return None
