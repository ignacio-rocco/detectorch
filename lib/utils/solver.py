def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_step_index(cur_iter,lr_steps=[0, 240000, 320000],max_iter=360000):
    """Given an iteration, find which learning rate step we're at."""
    assert lr_steps[0] == 0, 'The first step should always start at 0.'
    steps = lr_steps + [max_iter]
    for ind, step in enumerate(steps):  # NoQA
        if cur_iter < step:
            break
    return ind - 1


def lr_func_steps_with_decay(cur_iter,base_lr=0.01,gamma=0.1):
    """For cfg.SOLVER.LR_POLICY = 'steps_with_decay'
    Change the learning rate specified iterations based on the formula
    lr = base_lr * gamma ** lr_step_count.
    Example:
    cfg.SOLVER.MAX_ITER: 90
    cfg.SOLVER.STEPS:    [0,    60,    80]
    cfg.SOLVER.BASE_LR:  0.02
    cfg.SOLVER.GAMMA:    0.1
    for cur_iter in [0, 59]   use 0.02 = 0.02 * 0.1 ** 0
                 in [60, 79]  use 0.002 = 0.02 * 0.1 ** 1
                 in [80, inf] use 0.0002 = 0.02 * 0.1 ** 2
    """
    ind = get_step_index(cur_iter)
    return base_lr * gamma ** ind

def get_lr_at_iter(it,warm_up_iters=500,warm_up_factor=0.3333333333333333,warm_up_method='linear'):
    """Get the learning rate at iteration it according to the cfg.SOLVER
    settings.
    """
    lr = lr_func_steps_with_decay(it)
    if it < warm_up_iters:
        if warm_up_method == 'linear':
            alpha = it / warm_up_iters
            warm_up_factor = warm_up_factor * (1 - alpha) + alpha
        elif warm_up_method != 'constant':
            raise KeyError('Unknown WARM_UP_METHOD: {}'.format(warm_up_method))
        lr *= warm_up_factor
    return lr