import torch


def create_checkpoint(epoch, model, optimizer_f, scheduler_f, loss, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer_f.optimizer.state_dict(),
        'scheduler_state_dict': scheduler_f.scheduler.state_dict(),
        'loss': loss,
        'learning_rate': optimizer_f.optimizer.param_groups[0]['lr']
    }
    torch.save(checkpoint, filename)


def load_checkpoint(filename, model, optimizer_f, scheduler_f, method='model', map_location='cpu'):
    if method == 'resume':
        checkpoint = torch.load(filename, map_location=map_location)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer_f.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler_f.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        learning_rate = checkpoint['learning_rate']
        return epoch, loss, learning_rate
    elif method == 'model':
        checkpoint = torch.load(filename, map_location=map_location)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise ValueError('method must be resume or model')

