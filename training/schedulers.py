from torch.optim.lr_scheduler import StepLR, ExponentialLR, CosineAnnealingWarmRestarts, CosineAnnealingLR


class SchedulerFactory:
    def __init__(self, optimizer, scheduler_name, **kwargs):
        self.scheduler = None
        if scheduler_name == 'step_lr':
            self.scheduler = StepLR(optimizer, **kwargs)
        elif scheduler_name == 'exp_lr':
            self.scheduler = ExponentialLR(optimizer, )
        elif scheduler_name == 'cosine_wr':
            self.scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=0.00001)
        elif scheduler_name == 'cosine_lr':
            self.scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=0.00001)

    def step(self):
        if self.scheduler is not None:
            self.scheduler.step()
