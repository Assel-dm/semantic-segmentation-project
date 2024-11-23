from segmentation_models_pytorch import Unet

def build_unet():
    return Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    )

