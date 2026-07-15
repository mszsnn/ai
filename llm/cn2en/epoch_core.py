
# 创建核心一次完整迭代， 计算平均 loss

def train_one_epoch(model, train_loader, optimizer, criterion, device):
    """
    :params model 模型
    :params train_loader 训练的批数据加载器
    :params optimizer 优化器， 选择哪些参数进行更新
    :params criterion 计算 loss
    :params device 设备
    """

    # 训练模式， dropout 等都生效
    model.train()

    total_loss = 0

    for src, decoder_input, labels in train_loader:
        src = src.to(device)
        decoder_input = decoder_input.to(device)
        labels = labels.to(device)

        # forward

        logits = model(
            src,
            decoder_input
        )

        # 计算损失
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1)
        )

        # 清空梯度， 否则梯度会累计
        optimizer.zero_grad()

        # 参数更新
        loss.backward()

        # 更新参数
        optimizer.step()

        total_loss = total_loss + loss.item()


    return total_loss / len(train_loader)






