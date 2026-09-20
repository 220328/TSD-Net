from methods.SINet import SINet_ResNet50
from methods.HDPNet_model import Model

MODEL_REGISTRY = {
    'HDPNet': {
        'model_class': Model,
        'loss_fn': 'multi_bce',
        'default_args': {
            'img_size': 384
        }
    },
    'SINet': {
        'model_class': SINet_ResNet50,
        'loss_fn': 'multi_bce_2outputs',
        'default_args': {}
    }
}

def get_model(model_name, **kwargs):
    """
    获取模型实例
    Args:
        model_name: 模型名称
        **kwargs: 模型初始化参数
    Returns:
        model: 模型实例
        loss_fn_name: 对应的损失函数名称
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} not found in registry")
    
    model_config = MODEL_REGISTRY[model_name]
    model_class = model_config['model_class']
    loss_fn_name = model_config['loss_fn']
    
    # 合并默认参数和传入参数
    model_args = model_config['default_args'].copy()
    model_args.update(kwargs)
    
    return model_class(**model_args), loss_fn_name