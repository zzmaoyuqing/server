import numpy as np
import torch as t
import torch.utils.data as Data

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class MyDataset(Dataset):
    def __init__(self):
        # 1. import the dataset with trainset : valdset : testset = 0.6 : 0.2 : 0.2

        self.train_data = np.load('dataset/train_set/train_data.npy')

        self.train_target = np.load('dataset/train_set/train_target.npy')
        self.train_len = self.train_data.shape[0]

        self.val_data = np.load('dataset/val_set/val_data.npy')
        self.val_target = np.load('dataset/val_set/val_target.npy')
        self.val_len = self.val_data.shape[0]

        self.test_data = np.load('dataset/test_set/test_data.npy',)
        self.test_target = np.load('dataset/test_set/test_target.npy')
        self.test_len = self.test_data.shape[0]



    def __getitem__(self, idx):
        # 根据索引返回数据和对应的标签
        return self.train_data[idx], self.train_target[idx], self.val_data[idx], self.val_target[idx], self.test_data[idx], self.test_target[idx]

    def __len__(self):
        # 返回文件数据的数目
        return self.train_len,  self.val_len, self.test_len

def TensorDataset(dataset):
    train_x = dataset.train_data                            # ndarray(3056,2,128)
    train_x = t.tensor(train_x)                             # Tensor(3.56,2,128)
    train_y = dataset.train_target                          # ndarray(1,3056)
    train_y = t.tensor(train_y.reshape(len(train_x), -1))   # ndarray(1,3056)--->Tensor(1,3056)---->Tensor(3056,1)


    val_x = dataset.val_data
    val_x = t.tensor(val_x)
    val_y = dataset.val_target
    val_y = t.tensor(val_y.reshape(len(val_x), -1))


    test_x = dataset.test_data
    test_x = t.tensor(test_x)
    test_y = dataset.test_target
    test_y = t.tensor(test_y.reshape(len(test_x), -1))

    torch_train_dataset = Data.TensorDataset(train_x, train_y)
    torch_val_dataset = Data.TensorDataset(val_x, val_y)
    torch_test_dataset = Data.TensorDataset(test_x, test_y)

    return torch_train_dataset, torch_val_dataset, torch_test_dataset

if __name__ == '__main__':
    # 读入数据
    dataset = MyDataset()
    torch_train_dataset, torch_val_dataset, torch_test_dataset = TensorDataset(dataset)

    # train_loader = DataLoader(torch_train_dataset, batch_size=4)
    # for data in train_loader:
    #     emb, target = data
    #     print(emb.shape)
    #     print(target)
