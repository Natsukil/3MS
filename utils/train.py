import torch.nn

def evaluate_accuracy_gpu(net, data_iter, device=None):  #@save
    """使用GPU计算模型在数据集上的精度"""
    if isinstance(net, nn.Module):
        net.eval()  # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = d2l.Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # BERT微调所需的（之后将介绍）
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(d2l.accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

def train_ch6(net, train_iter, test_iter, num_epochs, lr, device, save_path=None, load_dir=None):
    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    model_path = ""
    if save_path:
        current_time = datetime.datetime.now().strftime("%m-%d-%H-%M-%S")
        directory = os.path.join(save_path, current_time)
        os.makedirs(directory, exist_ok=True)  # 创建目录
        model_path = os.path.join(directory, 'best.ckpt')

    print('trainin on', device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss_f = nn.CrossEntropyLoss()
    # animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
    #                         legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)

    best_test_acc = 0
    if load_dir:
        net.load_state_dict(torch.load(load_dir))
        print(f"load model from {load_dir}")
        best_test_acc = evaluate_accuracy_gpu(net, test_iter)
        print(f"pretrain test acc: {best_test_acc}")
    else:
        net.apply(init_weights)
        print("new model")

    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，样本数
        metric = d2l.Accumulator(3)
        net.train()
        with tqdm(total=len(train_iter.dataset), desc=f'Epoch {epoch + 1}/{num_epochs}', unit='img') as pbar:
            for i, (X, y) in enumerate(train_iter):
                batch_size = X.shape[0]
                timer.start()
                optimizer.zero_grad()
                X, y = X.to(device), y.to(device)
                y_hat = net(X)
                loss = loss_f(y_hat, y)
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    metric.add(loss * X.shape[0], d2l.accuracy(y_hat, y), X.shape[0])
                timer.stop()
                pbar.update(batch_size)
                train_l = metric[0] / metric[2]
                train_acc = metric[1] / metric[2]
                pbar.set_postfix(loss=f'{train_l:.4f}', train_acc=f'{train_acc:.4f}')
                # if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                #     animator.add(epoch + (i + 1) / num_batches,
                #                  (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        print(f'epoch {epoch}, loss {train_l:.3f}, train acc {train_acc:.3f}, test_acc {test_acc:.3f}')
        # animator.add(epoch + 1, (None, None, test_acc))
        if (epoch + 1) % 3 == 0 and test_acc > best_test_acc:
            best_test_acc = test_acc
            if save_path:
                torch.save(net.state_dict(), model_path)
                print(f"Saved best model checkpoint to {model_path} with test accuracy {test_acc:.3f}")

    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')