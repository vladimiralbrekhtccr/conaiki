# count the amount of params in module:

def count_params(model_module):
    """
    Args:
        model_module: ()
            provide module for which you want to calculate the amount of params

    Example:

    ```python
    >>> count_params(model_module=model.lm_head)
    ```
    """
    module_params = sum(p.numel() for p in model_module.parameters() if p.requires_grad)
    print(module_params)


def temp_freeze_all(model):
    for p in model.audio_tower.parameters():
        p.requires_grad = False
    for p in model.visual.parameters():
        p.requires_grad = False
    for p in model.model.parameters():
        p.requires_grad = False
    for p in model.conaiki_gate.parameters():
        p.requires_grad = False
    for p in model.lm_head.parameters():
        p.requires_grad = False
    for p in model.conaiki_time.parameters():
        p.requires_grad = False