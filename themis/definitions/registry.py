EVAL_REGISTRY = {}


def register_eval(name):
    def decorate(cls):
        EVAL_REGISTRY[name] = cls
        return cls

    return decorate


def get_eval(eval_name):
    """Factory function"""
    try:
        return EVAL_REGISTRY[eval_name]
    except KeyError as e:
        raise NotImplementedError(f"Evaluation {eval_name} not found") from e
