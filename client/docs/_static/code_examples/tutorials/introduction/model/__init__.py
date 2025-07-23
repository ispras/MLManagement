from .model import MyModelWrapper


def get_object(param1: int, param2: str, ..., paramN: bool, weight_path: str):
    return MyModelWrapper(param1, param2, ..., paramN, weight_path)
