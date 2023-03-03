'''
for citeulike-a
'''
#coding:utf-8
import numpy as np
import scipy.io as sio
import time

def rmse(r_pred, test_mat):
    y_pred = r_pred[test_mat>0]
    print('mae_rmse y_pred top 5:', y_pred[:5])
    y_true = test_mat[test_mat>0]
    print('mae_rmse y_true top 5:', y_true[:5])
    rmse = np.sqrt(np.mean(np.square(y_pred-y_true)))
    return rmse

def load_rating(path):
    arr = []
    for line in open(path):
        a = line.strip().split()
        if a[0]==0:
            l = []
        else:
            l = [int(x) for x in a[1:]]
        arr.append(l)
    return arr

def load_cite_data():
    data = {}
    data_dir = "data/citeulike-a/"
    data["train_users"] = load_rating(data_dir + "cf-train-10-users.dat")
    data["train_items"] = load_rating(data_dir + "cf-train-10-items.dat")
    data["test_users"] = load_rating(data_dir + "cf-test-10-users.dat")
    data["test_items"] = load_rating(data_dir + "cf-test-10-items.dat")

    return data

def sequence2mat(sequence, N, M):
    # input:
    # sequence: the list of rating information
    # N: row number, i.e. the number of users
    # M: column number, i.e. the number of items
    # output:
    # mat: user-item rating matrix
    records_array = np.array(sequence)
    mat = np.zeros([N,M])
    row = records_array[:,0].astype(int)
    col = records_array[:,1].astype(int)
    values = records_array[:,2].astype(np.float32)
    mat[row,col]=values

    return mat

def load_data(file_dir):
    # output:
    # N: the number of user;
    # M: the number of item
    # data: the list of rating information
    user_ids_dict, rated_item_ids_dict = {},{}
    N, M, u_idx, i_idx = 0,0,0,0
    data = []
    f = open(file_dir)
    for line in f.readlines():
        if '::' in line:
            u, i, r, _ = line.split('::')
        else:
            u, i, r, _ = line.split()

        if int(u) not in user_ids_dict:
            user_ids_dict[int(u)]=u_idx
            u_idx+=1
        if int(i) not in rated_item_ids_dict:
            rated_item_ids_dict[int(i)]=i_idx
            i_idx+=1
        data.append([user_ids_dict[int(u)],rated_item_ids_dict[int(i)],float(r)])

    f.close()
    N = u_idx
    M = i_idx

    return N, M, data, rated_item_ids_dict

class PMF_SGD():
    def __init__(self,trainingSet,testSet, N, M, K, lRate, regU, regI, maxEpoch):
        self.trainingSet = trainingSet
        self.testSet = testSet
        self.lRate = lRate
        self.regU = regU
        self.regI = regI
        self.maxEpoch = maxEpoch
        self.N = N
        self.M = M
        self.K = K
        self.batch_size = 100
        self.train_size = len(trainingSet)
        self.R, self.confidence_Mat = self.R_Confidence()
        self.P = np.random.normal(0, 0.1, (self.N, self.K))  # user latent vector
        self.Q = np.random.normal(0, 0.1, (self.M, self.K))  # item latent vector

    def R_Confidence(self):
        R = np.zeros((N,M))
        a = 1
        b = 0.01
        confidence_Matrix = np.zeros((N,M)) * b
        for entry in self.trainingSet:
            u, i, rating = entry
            R[u][i] = rating
            confidence_Matrix[u][i] = a
        return R, confidence_Matrix

    def trainModel(self):
        '''
        pure SGD, use one training sample at a time.
        i.e., update P[u] with one Q[i] instead of all Q, same as updating Q[i].
        '''
        last_rmse = 1000
        epoch = 0
        while epoch < self.maxEpoch:
            self.loss = 0
            for entry in self.trainingSet:
                u, i, rating = entry
                error = rating - self.P[u].dot(self.Q[i])
                self.loss += error**2
                p = self.P[u]
                q = self.Q[i]
                #update latent vectors
                self.P[u] += self.lRate*(error*q-self.regU*p)
                self.Q[i] += self.lRate*(error*p-self.regI*q)

            self.loss += self.regU*(self.P*self.P).sum() + self.regI*(self.Q*self.Q).sum()
            epoch += 1
            preds = self.prediction(self.P, self.Q)
            test_mat = sequence2mat(sequence = self.testSet, N = self.N, M = self.M)
            test_rmse = rmse(preds, test_mat)
            if test_rmse < last_rmse:
                print('epoch ', epoch, ' loss:', self.loss, ' test rmse:', test_rmse)
                last_rmse = test_rmse
            else:
                print('test rmse not reduce. done!')
                break

    def trainModel_ab(self):
        '''
        update P[u] with all Q, same as updating Q[i].
        In addtion, the confidence C is considered, Cij = a if Rij = 1 and Cij = b otherwise.
        '''
        last_rmse = 1000
        epoch = 0
        loss_old = 200000
        self.loss = 0
        converage = 1.0
        while epoch < self.maxEpoch and converage > 1e-4:
            loss_old = self.loss
            self.loss = 0
            # updata P
            for u in range(N):
                rating = self.R[u]
                c = self.confidence_Mat[u]                #(M,), j:1~M
                error = rating - self.P[u].dot(self.Q.T)  #(M,), j:1~M
                p = self.P[u]
                self.P[u] += self.lRate*((c * error).dot(self.Q) - self.regU*p)
                self.loss += np.sum(c * error**2)
            # update Q
            for i in range(M):
                rating = self.R[:,i]
                c = self.confidence_Mat[:,i]                #(N,), i:1~N
                error = rating - self.Q[i].dot(self.P.T)    #(N,), i:1~N
                q = self.Q[i]
                self.Q[i] += self.lRate*((c * error).dot(self.P) - self.regI*q)
            self.loss += self.regU*(self.P*self.P).sum() + self.regI*(self.Q*self.Q).sum()
            epoch += 1
            preds = self.prediction(self.P, self.Q)
            test_mat = sequence2mat(sequence = self.testSet, N = self.N, M = self.M)
            test_rmse = rmse(preds, test_mat)
            converage = abs((loss_old - self.loss) / (loss_old+0.01))
            if self.loss < loss_old:
                print('loss is descreasing.')
            print('epoch ', epoch, ' loss:', self.loss, ' converage:', converage, ' test rmse:', test_rmse)

    def prediction(self, P, Q):
        N,K = P.shape
        rating_list=[]
        for u in range(N):
            u_rating = np.sum(P[u,:]*Q, axis=1)
            rating_list.append(u_rating)
        r_pred = np.array(rating_list)
        return r_pred

    def save_model(self, pmf_path=None):
        if pmf_path is not None:
            sio.savemat(pmf_path,{"P": self.P, "Q": self.Q})

import math
import scipy
class PMF_ALS:
    def __init__(self, num_users, num_items, num_factors, lambda_u, lambda_v, max_iter, random_seed=0):
        self.m_num_users = num_users
        self.m_num_items = num_items
        self.m_num_factors = num_factors
        self.m_U = 0.1 * np.random.randn(self.m_num_users, self.m_num_factors)
        self.m_V = 0.1 * np.random.randn(self.m_num_items, self.m_num_factors)
        self.max_iter = max_iter
        self.lambda_u = lambda_u
        self.lambda_v = lambda_v

    def pmf_ab(self, users, items, testSet):
        '''
        算法：共轭梯度法求解协同过滤中的 ALS
             https://zhuanlan.zhihu.com/p/88139448
        实现：hulongma
        注： 实现中分别采用：（1）ALS公式推导的标准算法，每轮耗时大约6070s；
                          （2）考虑V.T * Ci * V中的Ci是对角阵， 对于V.T * Ci采用数组的点乘加快运算，每轮每轮耗时约87s
        '''
        a = 1
        b = 0.01
        loss = math.exp(20)
        converge = 1.0
        it = 0
        while (it < self.max_iter and converge > 1e-6):
            loss_old = loss
            loss = 0
            begin = time.time()

            # update U
            # Ui = (V.T * Ci * V + lambda_u * I)^-1 * V.T * Ci * Ri
            # Ci is a diagonal matrix with Cij, j = 1,..., M as its diagonal elements
            # Ri is a vector for user i
            for i in range(self.m_num_users):
                item_ids = users[i]
                ei = np.ones(self.m_num_items) * b
                ei[item_ids] = a
                Ri = np.zeros(self.m_num_items)
                Ri[item_ids] = 1

                # 原汁原味的ALS算法
                # Ci = np.diag(ei)
                # A = self.m_V.T.dot(Ci).dot(self.m_V)
                # A += self.lambda_u * np.eye(self.m_num_factors)
                # x = self.m_V.T.dot(Ci).dot(Ri[:,np.newaxis])

                # 使用简便算法, 目的是加快V.T * Ci * V的运行速度
                # 参考：Python numpy矩阵乘以一个对角矩阵 http://cn.voidcc.com/question/p-oemsnwop-tk.html
                Ci = ei
                AA = (Ci * self.m_V.T)  # m_V.dot(Ci) -> (ei * self.m_V.T)
                A = AA.dot(self.m_V)
                A += self.lambda_u * np.eye(self.m_num_factors)
                x = AA.dot(Ri[:, np.newaxis])
                self.m_U[i, :] = np.squeeze(scipy.linalg.solve(A, x))

                loss += np.sum(ei * (Ri - self.m_U[i].dot(self.m_V.T)) ** 2)
                loss += self.lambda_u * np.sum(self.m_U[i]*self.m_U[i])


            # update V
            # Vj = (U.T * Cj * U + lambda_v * I)^-1 * U.T * Cj * Rj
            # Cj is a diagonal matrix with Cij, i = 1,..., N as its diagonal elements
            # Rj is a vector for item j
            for j in range(self.m_num_items):
                user_ids = items[j]
                m = len(user_ids)
                if m > 0:
                    ej = np.ones(self.m_num_users) * b
                    ej[user_ids] = a
                    Rj = np.zeros(self.m_num_users)
                    Rj[user_ids] = 1

                    # 原汁原味的ALS算法
                    # Cj = np.diag(ej)
                    # A = self.m_U.T.dot(Cj).dot(self.m_U)
                    # A += self.lambda_v * np.eye(self.m_num_factors)
                    # x = self.m_U.T.dot(Cj).dot(Rj[:,np.newaxis])

                    # 使用简便算法
                    Cj = ej
                    AA = (Cj * self.m_U.T)
                    A = AA.dot(self.m_U)
                    A += self.lambda_v * np.eye(self.m_num_factors)
                    x = AA.dot(Rj[:, np.newaxis])
                    self.m_V[j, :] = np.squeeze(scipy.linalg.solve(A, x))

                loss += self.lambda_v * np.sum(self.m_V[j]*self.m_V[j])

            it += 1
            converge = abs(1.0*(loss - loss_old)/loss_old)


            if loss < loss_old:
                print("loss is decreasing!")

            preds = self.prediction(self.m_U, self.m_V)
            test_mat = sequence2mat(sequence = testSet, N =self.m_num_users, M = self.m_num_items)
            test_rmse = rmse(preds, test_mat)


            print("[iter=%04d], loss=%.5f, converge=%.10f, test_rmse=%.4f, time_consume=%4d" % (it, loss, converge,test_rmse, time.time()-begin))

    def pmf_estimate(self, users, items, testSet):
        """
        This method is based on the official implementation of CVAE, I can't guarantee its correctness.
        """
        a = 1
        b = 0.01
        min_iter = 1
        a_minus_b = a - b  #  confidense value, Cij = a if Rij = 1 and Cij = b otherwise
        converge = 1.0
        likelihood_old = 0.0
        likelihood = -math.exp(20)
        it = 0
        while (it < self.max_iter and converge > 1e-6):
            likelihood_old = likelihood
            likelihood = 0

            # update U
            # VV^T for v_j that has at least one user liked
            ids = np.array([len(x) for x in items]) > 0
            v = self.m_V[ids]
            VVT = np.dot(v.T, v)
            XX = VVT * b + np.eye(self.m_num_factors) * self.lambda_u
            for i in range(self.m_num_users):
                item_ids = users[i]
                n = len(item_ids)
                if n > 0:
                    A = np.copy(XX)
                    A += np.dot(self.m_V[item_ids, :].T, self.m_V[item_ids,:])*a_minus_b
                    x = a * np.sum(self.m_V[item_ids, :], axis=0)
                    self.m_U[i, :] = scipy.linalg.solve(A, x)

                    likelihood += -0.5 * self.lambda_u * np.sum(self.m_U[i]*self.m_U[i])

            # update V
            ids = np.array([len(x) for x in users]) > 0
            u = self.m_U[ids]
            XX = np.dot(u.T, u) * b
            for j in range(self.m_num_items):
                user_ids = items[j]
                m = len(user_ids)
                if m>0:
                    A = np.copy(XX)
                    A += np.dot(self.m_U[user_ids,:].T, self.m_U[user_ids,:])*a_minus_b
                    B = np.copy(A)
                    A += np.eye(self.m_num_factors) * self.lambda_v
                    x = a * np.sum(self.m_U[user_ids, :], axis=0)
                    self.m_V[j,:] = scipy.linalg.solve(A, x)

                    likelihood += -0.5 * self.lambda_v * np.sum(self.m_V[j]*self.m_V[j])
                    likelihood += a * np.sum(self.m_U[user_ids, :].dot(self.m_V[j,:][:, np.newaxis]),axis=0)
                    likelihood += -0.5 * self.m_V[j,:].dot(B).dot(self.m_V[j,:][:,np.newaxis])

            it += 1
            converge = abs(1.0*(likelihood - likelihood_old)/likelihood_old)

            if self.verbose:
                if likelihood < likelihood_old:
                    print("likelihood is decreasing!")

            preds = self.prediction(self.m_U, self.m_V)
            test_mat = sequence2mat(sequence = testSet, N =self.m_num_users, M = self.m_num_items)
            test_rmse = rmse(preds, test_mat)

            print("[iter=%04d], likelihood=%.5f, converge=%.10f, test_rmse=%.4f" % (it, likelihood, converge,test_rmse))

    def prediction(self, P, Q):
        N,K = P.shape
        M,K = Q.shape
        rating_list=[]
        for u in range(N):
            u_rating = np.sum(P[u,:]*Q, axis=1)
            rating_list.append(u_rating)
        r_pred = np.array(rating_list)
        return r_pred

    def save_model(self, pmf_path=None):
        if pmf_path is not None:
            sio.savemat(pmf_path,{"P": self.m_U, "Q": self.m_V})



if __name__ == '__main__':
    np.random.seed(0)
    data = load_cite_data()
    print('data train users', len(data['train_users']))
    print('data test users:', len(data['test_users']))
    # num_users=5551(0~5550), num_items=16980(0~16797)

    # 将data['train_users']和data['test_users']转换为 [[u, i, r],...]的格式
    train_list = []
    for u, items in enumerate(data['train_users']):
        for i in items:
            train_list.append([u, i, 1])
    print('train list:', train_list[:100])
    test_list = []
    for u, items in enumerate(data['test_users']):
        for i in items:
            test_list.append([u, i, 1])
    print('test list:', test_list[:100])
    print ('train length: %d \n test length: %d' %(len(train_list),len(test_list)))

    N = 5551   # num of users
    M = 16980  # num of items

    pmf = PMF_SGD(train_list, test_list, N, M, 50, 0.02, 0.1, 1, 500)
    pmf.trainModel()
    # 50, 0.03, 0.1, 1, 200, rmse:0.6338, recall:0.1616, 400: recall:0.1567
    pmf.trainModel_ab()
    # 50, 0.03, 0.1, 1, 100, 实际到了30轮就截止， recall:0.058
    # 50, 0.02, 0.1, 1, 100, 实际到了92轮就截止， recall:0.074
    # 50, 0.01, 0.1, 1, 800, 实际到了401轮就截止，recall:0.087
    # 50, 0.02, 0.1, 1, 500, 实际到了447轮就截止，rmse: 0.8164, recall:0.0919
    pmf.save_model(pmf_path="../model/pmf_only_dense.mat")

    pmf = PMF_ALS(N, M, 50, 0.1, 1, 200)
    pmf.pmf_estimate(data['train_users'], data["train_items"], test_list)
    #50, 0.1, 1, 100, likelihood一直上升，最终迭代到100轮， test_rmse=0.8934，racall: 0.1811
    #50, 0.1, 1, 200, likelihood一直上升，最终迭代到164轮， test_rmse=0.8933，racall: 0.1806
    pmf.save_model(pmf_path="../model/pmf_only_dense.mat")

    pmf = PMF_ALS(N, M, 50, 0.1, 1, 200)
    pmf.pmf_ab(data['train_users'], data["train_items"], test_list)
    # 50, 0.1, 1, 100, loss一直下降，最终迭代到100轮， test_rmse=0.8949， recall: 0.2219
    # 50, 0.1, 1, 200, loss一直下降，最终迭代到200轮， test_rmse=0.8946， recall: 0.2216
    pmf.save_model(pmf_path="../model/pmf_only_dense.mat")


