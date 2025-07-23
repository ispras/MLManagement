from .names import new_model_name
from ML_management.mlmanagement import load_model
from sklearn import datasets


model = load_model(new_model_name).loaded_object
iris = datasets.load_iris()

assert model.predict_function(iris.data[0].reshape(1, -1)) == iris.target[0]
