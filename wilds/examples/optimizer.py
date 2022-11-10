from torch.optim import SGD, Adam
from transformers import AdamW
from pytorch_dnn_arsenal.optimizer import build_optimizer, OptimizerSetting

# def initialize_optimizer_kwargs(config):
#     if config.optimizer_kwargs == None:
#         if config.optim_momentum != None:
#             config.optimizer_kwargs['momentum'] = config.optim_momentum

#         # For Adaptive Optimizer
#         if config.optim_eps != None:
#             config.optimizer_kwargs['eps'] = config.optim_eps

#         # For RMSProp     
#         if config.optim_alpha != None:
#             config.optimizer_kwargs['alpha'] = config.optim_alpha

#         # For Adam     
#         if config.optim_beta_1 != None:
#             config.optimizer_kwargs['beta_1'] = config.optim_beta_1
#         if config.optim_beta_2 != None:
#             config.optimizer_kwargs['beta_2'] = config.optim_beta_2

def initialize_optimizer_with_arsenal(config, model):

    # initialize optimizers
    optimizer = build_optimizer(
            OptimizerSetting(
                name=config.optimizer_name,
                lr=config.lr,
                weight_decay=config.weight_decay,
                model=model,
                momentum=config.momentum, # sgd, momentum_sgd, sgd_nesterov, rmsprop
                eps=config.eps, # adam, rmsprop (term added to the denominator to improve numerical stability)
                alpha=config.alpha, # rmsprop(smoothing constant of 2nd moment)
                beta_1=config.beta_1, # adam (smoothing constant of 1st moment)
                beta_2=config.beta_2  # adam (smoothing constant of 2nd moment)
            )
        )

    return optimizer

def initialize_optimizer(config, model):

    # config = initialize_optimizer_kwargs(config)

    # initialize optimizers
    if config.optimizer=='SGD':
        params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = SGD(
            params,
            lr=config.lr,
            weight_decay=config.weight_decay,
            **config.optimizer_kwargs)
    elif config.optimizer=='AdamW':
        if 'bert' in config.model or 'gpt' in config.model:
            no_decay = ['bias', 'LayerNorm.weight']
        else:
            no_decay = []

        params = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': config.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        optimizer = AdamW(
            params,
            lr=config.lr,
            **config.optimizer_kwargs)
    elif config.optimizer == 'Adam':
        params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = Adam(
            params,
            lr=config.lr,
            weight_decay=config.weight_decay,
            **config.optimizer_kwargs)
    else:
        raise ValueError(f'Optimizer {config.optimizer} not recognized.')

    return optimizer

def initialize_optimizer_with_model_params(config, params):
    if config.optimizer=='SGD':
        optimizer = SGD(
            params,
            lr=config.lr,
            weight_decay=config.weight_decay,
            **config.optimizer_kwargs
        )
    elif config.optimizer=='AdamW':
        optimizer = AdamW(
            params,
            lr=config.lr,
            weight_decay=config.weight_decay,
            **config.optimizer_kwargs
        )
    elif config.optimizer == 'Adam':
        optimizer = Adam(
            params,
            lr=config.lr,
            weight_decay=config.weight_decay,
            **config.optimizer_kwargs
        )
    else:
        raise ValueError(f'Optimizer {config.optimizer} not supported.')

    return optimizer
