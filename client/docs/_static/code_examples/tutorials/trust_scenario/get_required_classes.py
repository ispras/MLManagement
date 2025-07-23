from ML_management.sdk import get_required_classes_by_executor


req_classes = get_required_classes_by_executor("ProjectedGradientDescent_Attack")
print(req_classes)