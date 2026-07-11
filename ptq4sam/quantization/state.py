import logging
from .fake_quant import QuantizeBase
logger = logging.getLogger("ptq4sam")


def enable_calibration_woquantization(model, quantizer_type='fake_quant'):
    logger.info('Enable observer and Disable quantize for {}'.format(quantizer_type))
    for name, submodule in model.named_modules():
        if isinstance(submodule, QuantizeBase):
            if quantizer_type not in name:
                logger.debug('The except_quantizer is {}'.format(name))
                submodule.disable_observer()
                submodule.disable_fake_quant()
                continue
            logger.debug('Enable observer and Disable quant: {}'.format(name))
            submodule.enable_observer()        # 启用模型中指定类型的量化器的观察器
            submodule.disable_fake_quant()     # 禁用实际的量化操作


def enable_quantization(model, quantizer_type='fake_quant'):
    logger.info('Disable observer and Enable quantize.')
    for name, submodule in model.named_modules():
        if isinstance(submodule, QuantizeBase):
            if quantizer_type not in name:
                logger.debug('The except_quantizer is {}'.format(name))
                submodule.disable_observer()
                submodule.disable_fake_quant()
                continue
            logger.debug('Disable observer and Enable quant: {}'.format(name))
            submodule.disable_observer()
            submodule.enable_fake_quant()


def disable_all(model):
    logger.info('Disable observer and disable quantize.')
    for name, submodule in model.named_modules():
        if isinstance(submodule, QuantizeBase):
            logger.debug('Disable observer and disable quant: {}'.format(name))
            submodule.disable_observer()
            submodule.disable_fake_quant()
