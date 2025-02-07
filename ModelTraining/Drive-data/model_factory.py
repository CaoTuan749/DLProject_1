# model_factory.py
import importlib

def get_model(model_name, input_dim, hidden_dim, output_dim=1):
    """
    Factory function to return a model instance.
    Expects a module named <model_name>.py with a class 'Model' defined.
    """
    module = importlib.import_module(model_name)
    model_class = getattr(module, "Model")
    return model_class(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
