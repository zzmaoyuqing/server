import time

from torch.utils.tensorboard import SummaryWriter

import Data
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
from sklearn import metrics

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# 加载数据
dataset = Data.MyDataset()
train_data, val_data, test_data = Data.TensorDataset(dataset)

# data0, target0 = train_data[0]
# print('第一个sub_emb的shape:', data0.shape)   # torch.Size([2, 128])
# print('第一个sub_emb的target', target0)       # tensor([0.])

# def add_dim(tensor_dataset):
#     for idx, dataset in enumerate(tensor_dataset):
#         X, y = dataset
#         X = X.unsqueeze(0)
#         tensor_dataset.tensors[idx] = X
#         print('扩充维度后的shape:', X.shape)
#         print('扩充维度后的target：', y)
#     return tensor_dataset
# data0, target0 = train_data[0]
# print('第一个sub_emb的shape:', data0.shape)   # torch.Size([2, 128])
# print('第一个sub_emb的target', target0)       # tensor([0.])

# load数据
train_loader = DataLoader(train_data, batch_size=32)
test_loader = DataLoader(test_data, batch_size=32)

# # 在定义train_loader时，设置了batch_size=32，表示一次性从数据集中取出32个数据
# for data in train_loader:
#     emb_sub, targets = data
#     print(emb_sub.shape)                 # torch.Size([32, 2, 128])
#     print(targets)                       # tensor([[0.],[1.], [1.],...,[0,])

print("训练集的长度:{}".format(len(train_data)))
print("测试集的长度:{}".format(len(test_data)))


# 构建模型思路： 输入 -> 节点嵌入 -> 卷积 -> Max pooling -> 展平 -> sigmoid分类 -> 输出
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=128,
                      out_channels=1,
                      kernel_size=2),
        )
        self.fc = nn.Linear(in_features=127, out_features=2)


    def forward(self, input):
        input = input.to(torch.float32)
        input = input.permute(0, 2, 1)
        output = self.conv(input)
        output = self.fc(output)

        return output


def compute_metrics(all_trues, all_scores, threshold=0.5):
    all_preds = (all_scores >= threshold)

    acc = metrics.accuracy_score(all_trues, all_preds)
    pre = metrics.precision_score(all_trues, all_preds)
    rec = metrics.recall_score(all_trues, all_preds)
    f1 = metrics.f1_score(all_trues, all_preds)
    mcc = metrics.matthews_corrcoef(all_trues, all_preds)
    fpr, tpr, _ = metrics.roc_curve(all_trues, all_scores)
    AUC = metrics.auc(fpr, tpr)
    p, r, _ = metrics.precision_recall_curve(all_trues, all_scores)
    AUPR = metrics.auc(r, p)

    return acc, f1, pre, rec, mcc, AUC, AUPR




def train(model, epoch, train_loader):

    # 定义损失函数、优化器
    loss_func = torch.nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        loss_func = loss_func.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 开始训练
    # 设置训练的参数
    total_train_step = 0   # 记录训练的次数
    total_test_step = 0    # 记录测试的次数
    epochs = epoch         # 训练的轮数

    # 训练可视化
    writer = SummaryWriter("logs_train_process")
    # 训练模型
    start_time = time.time()
    for i in range(epochs):
        print("----------------------第{}轮训练------------------------".format(i+1))

        # 训练步骤
        for i, data in enumerate(train_loader):
            X, y = data
            y = y.long()
            print('X的维度shape：', X.shape)   # 最后一个iteration： torch.Size([16,2,28])
            # X1 = X.unsqueeze(1)
            # print('扩充后X的维度shape：', X1.shape)

            if torch.cuda.is_available():
                X = X.cuda()             # 即emb_sub的训练集
                y = y.cuda()             # 即emb_sub训练集对应的y标签

            y_pred = model(X)            # 预测值
            loss = loss_func(y_pred, y)  # 计算loss值，用预测的y标签和训练集的y标签

            # 优化器
            optimizer.zero_grad()  # 梯度清零
            loss.backward()        # 反向传播
            optimizer.step()

            total_train_step += 1
            if total_train_step % 100 == 0:
                end_time = time.time()
                print("运行时间：", end_time - start_time)
                print("第{}次训练的loss{} ".format(total_train_step, loss.item()))
                writer.add_scalar("CNN_train_loss", loss, total_train_step)

        sum_loss = 0  # 记录总体损失值

        # 每轮训练完成跑一下测试数据看看情况
        accurate = 0
        model.eval()  # 也可以不写，规范的话就写，用来表明是测试步骤
        with torch.no_grad():
            for data in test_loader:
                # 这里的每一次循环 都是一个minibatch  一次for循环里面有32个数据。
                X, y = data
                if torch.cuda.is_available():
                    X = X.cuda()
                    y = y.cuda()

                y_pred = model(X)
                loss_in = loss(y_pred, y)

                sum_loss += loss_in
                print('这里是output', y_pred)
                accurate += (y_pred.argmax(1) == y).sum()

        print('第{}轮测试集的正确率:{:.2f}%'.format(epoch + 1, accurate / len(test_data) * 100))

        writer.add_scalar('看一下测试集损失', sum_loss, i)
        writer.add_scalar('看一下当前测试集正确率', accurate / len(test_data) * 100, i)
        i += 1

        torch.save(model, 'model_pytorch/model_{}.pth'.format(epoch + 1))
        print("第{}轮模型训练数据已保存".format(epoch + 1))

model = CNN()
train(model, 5, train_loader)