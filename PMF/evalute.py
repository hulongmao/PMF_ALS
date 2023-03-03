'''
获得 Recall@M
'''
import scipy.io as sio
import math
import numpy as np

M = 300
S = sio.loadmat('model/pmf_only_dense.mat')
m_U = S["P"]
m_V = S["Q"]
m_num_users = m_U.shape[0]
m_num_items = m_V.shape[0]

train_users = []
with open('data/citeulike-a/cf-train-10-users.dat') as f:
    for line in f:
        items = [int(item) for item in line.strip().split()[1:]]
        train_users.append(items)

test_users = []
with open('data/citeulike-a/cf-test-10-users.dat') as f:
    for line in f:
        items = [int(item) for item in line.strip().split()[1:]]
        test_users.append(items)

class Evaluator:

    def __init__(self, m_num_users, m_num_items, m_U, m_V):
        self.m_num_users = m_num_users
        self.m_num_items = m_num_items
        self.m_U = m_U
        self.m_V = m_V

    def predict(self, train_users, test_users, M):
        batch_size = 100
        n = int(math.ceil(1.0*self.m_num_users/batch_size))
        num_hit = np.zeros(self.m_num_items)
        recall = np.zeros(self.m_num_users)
        inter_mask = np.zeros(self.m_num_users)  # 掩码，指示用户是否有交互
        for i in range(n):
            u_tmp = self.m_U[i*batch_size:min((i+1)*batch_size, self.m_num_users)]  #此批的 user embedding
            score = np.dot(u_tmp, self.m_V.T)
            # print(i, ' batch, max score:', score.max(), ' min score:', score.min())  # max:0.75  min:-0.35
            gap = score.max() - score.min()

            bs = min((i+1)*batch_size, self.m_num_users) - i*batch_size
            gt = np.zeros((bs, self.m_num_items))
            for j in range(bs):
                ind = i*batch_size + j
                gt[j,train_users[ind]] = 1
            score = score - gap * gt                     # 去掉训练集的影响，保证训练的item得分最低。gt*gap， 将gt从原始范围（0~1）缩放到（0~gap）
            ind_rec = np.argsort(score, axis=1)[:,::-1]  # 沿行的方向进行排序的元素索引, [:,::-1]实现倒序

            # construct ground truth
            bs = min((i+1)*batch_size, self.m_num_users) - i*batch_size
            gt = np.zeros((bs, self.m_num_items))
            for j in range(bs):
                ind = i*batch_size + j
                gt[j,test_users[ind]] = 1
            # sort gt according to ind_rec
            rows = np.array(range(bs))[:, np.newaxis]
            gt = gt[rows, ind_rec]                          # 按照score从高到低的索引，重排测试集的得分

            # 标记测试集交互个数为0的行（也就np.sum(gt, axis=1)为0的行）
            total_interact = np.sum(gt, axis=1)
            index_zero = np.where(total_interact == 0)
            total_interact[index_zero] = 100000    #将交互为0的设置为一个很大的数，如100000
            recall[i*batch_size:min((i+1)*batch_size, self.m_num_users)] = 1.0*np.sum(gt[:, :M], axis=1)/total_interact
            inter_mask[index_zero] = 1
            num_hit += np.sum(gt, axis=0)

        print('recall 5551:', recall[:20])
        recall = np.ma.masked_array(recall, inter_mask)
        recall = np.mean(recall)
        return recall

if __name__ == '__main__':
    evalutor = Evaluator(m_num_users, m_num_items, m_U, m_V)
    recall = evalutor.predict(train_users, test_users, M)
    print('recall:', recall)

    # # 以下为recall@M的理解
    # score = np.array([[1.0, 0.5, 0.4, 0.2, -0.1, 1.5, 1.1, 0.7],
    #                   [-0.2,0.6, 0.9, 0.1, 1.2, 0.8, 1.3, 0.6]])
    # ind = np.argsort(score,axis=1)
    # ind_v = ind[:,::-1]
    # print('sort index:\n', ind_v)
    # rows = np.array(range(2))[:, np.newaxis]
    # # print('row:', rows)
    # gt = np.array([[0,1,1,0,0,1,0,0], [1,0,1,0,0,1,0,0]])    # 实际交互情况
    # re = gt[rows, ind_v]
    # print('re:\n', re)
    # M = 4
    # recall = np.zeros(2)
    # recall[0:2] = 1.0*np.sum(re[:, :M], axis=1)/np.sum(re, axis=1)
    # print('recall:', recall)

    print('haha')



