from .ffc import FFCResNetGenerator


def build_model(config):
    model_type = config.MODEL.NAME
    print(f"Creating model: {model_type}")
    if model_type == "FFCResNetGenerator":
        model = eval(model_type)(
            n_blocks=config.MODEL.BLOCK,
            num_class=config.MODEL.NUM_CLASSES
        )
    return model
