from torch.optim.lr_scheduler import StepLR, ExponentialLR, CosineAnnealingWarmRestarts


class SchedulerFactory:
    @staticmethod
    def get_scheduler(optimizer, scheduler_name, **kwargs):
        if scheduler_name == 'step_lr':
            return StepLR(optimizer, **kwargs)
        elif scheduler_name == 'exp_lr':
            return ExponentialLR(optimizer, **kwargs)
        elif scheduler_name == 'cosine_wr':
            return CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        # Add more schedulers here
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")
