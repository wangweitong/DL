import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch.nn as nn
from util.load_data_util import get_batch_loader
from sklearn import preprocessing

EPOCHS = 1000
BATCH_SIZE = 1000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


""" ************************************************************************************ """
"""                                      数据读取和转换                                    """
""" ************************************************************************************ """
def load_dataset():
    """
    FFM模型里Load数据的方式与FM模型中基本一致, 除了Field部分
    :return:
    """
    # Step1: 获取基本信息
    header = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
    df_user = pd.read_csv('../data/Movielens100K/u.user', sep='|', names=header)
    header = ['item_id', 'title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown', 'Action', 'Adventure',
              'Animation', 'Children', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
              'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    df_item = pd.read_csv('../data/Movielens100K/u.item', sep='|', names=header, encoding="ISO-8859-1")
    df_item = df_item.drop(columns=['title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown'])

    df_user['age'] = pd.cut(df_user['age'], [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
                            labels=['0-10', '10-20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90',
                                    '90-100'])
    df_user = pd.get_dummies(df_user, columns=['gender', 'occupation', 'age'])
    df_user = df_user.drop(columns=['zip_code'])

    user_features = df_user.columns.values.tolist()
    movie_features = df_item.columns.values.tolist()
    cols = user_features + movie_features
    cols.remove('user_id')
    cols.remove('item_id')

    # Step2: 把特征进行归类处理(分成4个field)
    # 这里, 如果我们把Field分成4类, Gender, Occupation, Age, Other
    field_index, feature2field = {}, {}
    other_idxs = []
    for idx, col in enumerate(cols):
        infos = col.split('_')
        if len(infos) == 2:
            field = infos[0]
            field_index[field] = field_index.get(field, len(field_index))
            feature2field[idx] = field_index[field]
        if len(infos) == 1:
            other_idxs.append(idx)
    for idx in other_idxs:
        feature2field[idx] = len(field_index)

    # Step3: 根据user_id, item_id进行Merge, 得到对应的数据
    header = ['user_id', 'item_id', 'rating', 'timestamp']
    df_train = pd.read_csv('../data/Movielens100K/ua.base', sep='\t', names=header)
    df_train = df_train.merge(df_user, on='user_id', how='left')
    df_train = df_train.merge(df_item, on='item_id', how='left')

    df_test = pd.read_csv('../data/Movielens100K/ua.test', sep='\t', names=header)
    df_test = df_test.merge(df_user, on='user_id', how='left')
    df_test = df_test.merge(df_item, on='item_id', how='left')

    # Step4: Label的变换, 以避免Cuda中报错
    # 需要对Label进行一定的转换, 因为原始的Label是[1, 2, 3, 4, 5]
    # 而 cuda中, 如果直接以这种Label的话, 会报错(Label 需要在[0, n_class - 1]范围
    # 因此, 需要转成[0, 1, 2, 3, 4]
    map_dict = dict()
    label_set = sorted(set(df_train['rating']) | set(df_test['rating']))
    for x in label_set:
        map_dict[x] = map_dict.get(x, len(map_dict))

    df_train['rating'] = df_train.rating.apply(lambda x: map_dict[x])
    df_test['rating'] = df_test.rating.apply(lambda x: map_dict[x])

    # # 如果想要使用"二分类"的话, 可以使用下面的方式来处理
    # df_train['rating'] = df_train.rating.apply(lambda x: 1 if int(x) == 1 else 0)
    # df_test['rating'] = df_test.rating.apply(lambda x: 1 if int(x) == 1 else 0)

    # Step5: 输出成np.array格式
    train_labels = np.array(df_train['rating'].astype(np.int32))
    test_labels = np.array(df_test['rating'].astype(np.int32))
    return df_train[cols].values, train_labels, df_test[cols].values, test_labels, feature2field


""" ************************************************************************************ """
"""                                          FFM层                                       """
""" ************************************************************************************ """
class FFM_layer(nn.Module):
    def __init__(self, field_map_dict, fea_num, reg_l1=0.01, reg_l2=0.01, class_num=1, latent_factor_dim=10):
        super(FFM_layer, self).__init__()
        self.reg_l1 = reg_l1
        self.reg_l2 = reg_l2
        self.fea_num = fea_num
        self.field_map_dict = field_map_dict  # 需要有个下标对应的字典

        field_num = len(field_map_dict)
        self.linear = nn.Linear(fea_num, class_num)   # 用于bias与线性部分
        self.v = nn.Parameter(torch.randn(fea_num, field_num, latent_factor_dim, class_num))

    def forward(self, x):
        # 先计算得到线性的那一部分
        linear_part = self.linear(x)

        # 计算交叉部分
        interaction_part = 0.0
        for i in range(self.fea_num):
            for j in range(i + 1, self.fea_num):
                v_ifj = self.v[i, self.field_map_dict[j], :, :]
                v_jfi = self.v[j, self.field_map_dict[i], :, :]

                xij = torch.unsqueeze(x[:, i] * x[:, j], dim=1)
                v_ijji = torch.unsqueeze(torch.sum(v_ifj * v_jfi, dim=0), dim=0)

                interaction_part += torch.mm(xij, v_ijji)

        output = linear_part + interaction_part
        output = torch.log_softmax(output, dim=1)
        return output


""" ************************************************************************************ """
"""                                     训练和测试FM模型                                   """
""" ************************************************************************************ """
def train_FFM_model_demo():

    # Step1: 导入数据
    x_train, y_train, x_test, y_test, feature2field = load_dataset()
    x_train = preprocessing.scale(x_train, with_mean=True, with_std=True)
    x_test = preprocessing.scale(x_test, with_mean=True, with_std=True)
    class_num = len(set([y for y in y_train] + [y for y in y_test]))

    # FFM模型
    ffm = FFM_layer(field_map_dict=feature2field, fea_num=x_train.shape[1], reg_l1=0.01, reg_l2=0.01,
                    class_num=class_num, latent_factor_dim=10).to(DEVICE)

    # 定义损失函数还有优化器
    optm = torch.optim.Adam(ffm.parameters())

    train_loader = get_batch_loader(x_train, y_train, BATCH_SIZE, shuffle=True)
    test_loader = get_batch_loader(x_test, y_test, BATCH_SIZE, shuffle=False)

    for epoch in range(1, EPOCHS + 1):
        train(ffm, DEVICE, train_loader, optm, epoch)
        test(ffm, DEVICE, test_loader)


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device, dtype=torch.float32), target.to(device).long()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)

        regularization_loss = 0
        for param in model.parameters():
            regularization_loss += model.reg_l1 * torch.sum(torch.abs(param))
            regularization_loss += model.reg_l2 * torch.sum(torch.pow(param, 2))
        loss += regularization_loss

        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % 500 == 0:
            print('Train Epoch: {} [{} / {} ({:.0f}%]\tLoss:{:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()
            ))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device, dtype=torch.float32), target.to(device).long()
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()

            regularization_loss = 0
            for param in model.parameters():
                regularization_loss += model.reg_l1 * torch.sum(torch.abs(param))
                regularization_loss += model.reg_l2 * torch.sum(torch.pow(param, 2))
            test_loss += regularization_loss

            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),  100. * correct / len(test_loader.dataset)
    ))


if __name__ == '__main__':
    train_FFM_model_demo()