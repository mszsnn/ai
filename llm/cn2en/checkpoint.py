# 模型的保存和恢复
# 100 次epoch比较慢
import torch


def save_checkpoint(model, optimizer, epoch, loss, path):
    # 保存断点参数
    check_point = {
        'epoch': epoch,
        "model_state_dict": model.state_dict(),
        'optimize_state_dict': optimizer.state_dict(),
        "loss": loss,
    }

    torch.save(check_point, path)

    print('模型保存成功', path)



def load_checkpoint(model, optimizer, path, device):

    check_point = torch.load(path, map_location=device)

    # 恢复模型参数
    model.load_state_dict(check_point['model_state_dict'])

    # 恢复optimizer参数
    if optimizer is not None:
        optimizer.load_state_dict(check_point['optimize_state_dict'])

    epoch = check_point['epoch']
    loss = check_point['loss']

    print('模型加载成功')

    print('epoch', epoch)

    print(f'loss{epoch}', loss)

    return epoch, loss