from .torch_model import BWResNet18Wrapper


def get_object(
    num_classes: int = 10, pretrained: bool = True, hid_lay_size: int = 100,
    dropout: float = 0.1, weights_path: str = "best_model.pth"
):
    return BWResNet18Wrapper(num_classes, pretrained, hid_lay_size, dropout, weights_path)
