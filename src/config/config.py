import yaml
import types

class Config(types.SimpleNamespace):
    """
    Config loader: loads a YAML and turns nested dicts into attribute-accessible namespaces.
    """

    def __init__(self, **params):
        super().__init__(**params)

    @classmethod
    def from_yaml(cls, config_file: str):
        """Load configuration from a YAML file."""
        with open(config_file, "r") as f:
            params = yaml.safe_load(f)

        def convert(val):
            if isinstance(val, dict):
                return types.SimpleNamespace(**{k: convert(v) for k, v in val.items()})
            elif isinstance(val, (list, tuple)):
                t = type(val)
                return t(convert(v) for v in val)
            else:
                return val

        params_converted = {key: convert(value) for key, value in params.items()}
        return cls(**params_converted)
