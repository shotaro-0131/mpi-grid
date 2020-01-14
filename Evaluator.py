import torch
from sklearn.metrics import r2_score
import Net
import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from sklearn.metrics import mean_squared_error
class Evaluator:

    def __init__(self, tr_x, tr_y):

        self.tr_id = tr_x.T[0]
        self.tr_x = tr_x.T[1:].T.reshape(len(tr_y), 155, 1, 14)
        self.tr_y = tr_y
        # self.te_x = te_x.reshape(840, 155, 1, 14)
        # self.te_y = te_y

        # print("init complited")

        # train_set = torch.utils.data.TensorDataset(torch.from_numpy(np.array(tr_x)).float(),
        #                                 torch.from_numpy(np.array(train_y).astype(np.float32)))
        # test_set = torch.utils.data.TensorDataset(torch.from_numpy(np.array(te_x)).float(),
        #                                 torch.from_numpy(test_y_.astype(np.float32)))
        # self.train_loader = DataLoader(train_set, batch_size=BATCHSIZE, shuffle=True, num_workers=2)
        # self.test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=True, num_workers=2)

    def setParams(self, params):
        self.num_layers = params[0]
        self.kernel_sizes = params[1:5]
        self.num_filters = params[5:9]
        self.poolings = params[9:12]
        self.pooling_size = params[12:15]
        self.mid_unit = params[15]
        # self.mid_unit2 = params[11]
        self.lr = params[16]

    def train(self, model, device, train_loader, optimizer):
        criterion = nn.MSELoss()
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device, dtype=torch.float32), target.to(device, dtype=torch.float32)
        #         data = data.transpose(0,1)[feature].transpose(0,1)
                optimizer.zero_grad()
                output = model(data).view(data.shape[0])
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
        return model

    def test(self, model, device, test_loader):
        model.eval()
        error = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device, dtype=torch.float32), target.to(device, dtype=torch.float32)
    #             data = data.transpose(0,1)[feature].transpose(0,1)
                pred = model(data).view(data.shape[0])
    #             print(data.shape,pred.shape,target.shape)
    #             pred = output.max(1, keepdim=True)[1]
                
            error = r2_score(y_true=target,y_pred=pred)
            # print(target.numpy().reshape(pred.shape[0]))
            # print(pred.numpy().reshape(pred.shape[0]))

            r = np.corrcoef(target.numpy().reshape(pred.shape[0]),pred.numpy().reshape(pred.shape[0]))[0][1]
            mse = mean_squared_error(y_true=target, y_pred=pred)

    #             error = np.corrcoef(pred,target)[0][1]
        return error, r, mse

    def get_optimizer(self, model):

        # weight_decay = 1e-2

        adam_lr = 0.1**(self.lr)
        optimizer = optim.Adam(model.parameters(), lr=adam_lr)

        return optimizer

    

    def run(self):
        EPOCH = 10
        BATCHSIZE = 100
        # device = "cuda" 
        device = "cuda" if torch.cuda.is_available() else "cpu"
        average = []
        ave =[]
        mses = []

        for train_index, test_index in KFold(n_splits=3, shuffle=True, random_state=2525).split(list(range(840))):
            tr_id = [i for i, k in enumerate(self.tr_id) if k in train_index]
            tr_x = self.tr_x[tr_id]
            tr_y = self.tr_y[tr_id]
            t_num =np.array([len([j for j,x in enumerate(self.tr_id) if k == x]) for i, k in enumerate(test_index)])
            f_t_id =np.array([i for i, k in enumerate(self.tr_id) if k in test_index])

            te_x = self.tr_x[f_t_id]
            te_y = self.tr_y[f_t_id]
            train_set = torch.utils.data.TensorDataset(torch.from_numpy(tr_x).float(),
                                    torch.from_numpy(tr_y.astype(np.float32)))
            # test_set = torch.utils.data.TensorDataset(torch.from_numpy(te_x).float(),
            #                                 torch.from_numpy(te_y.astype(np.float32)))
            train_loader = DataLoader(train_set, batch_size=BATCHSIZE, shuffle=True, num_workers=2)
            # test_loader = DataLoader(test_set, batch_size=len(test_set), shuffle=True, num_workers=2)

        

            model = Net.Net(self.num_layers, self.kernel_sizes, self.num_filters, self.poolings, self.pooling_size, self.mid_unit).to(device)
            optimizer = self.get_optimizer(model)
            
            for step in range(EPOCH):
                # print('*',end="")
                model = self.train(model, device, train_loader, optimizer)
            model.eval()
            pred = model(torch.from_numpy(np.array(te_x)).reshape(len(te_x),155,1,14).float())
            pred = np.array([pred.cpu()[i].item() for i in range(len(pred))])
            ps = []
            ta=[]
            v1 = 0
            for v in t_num:
                
#                 print(v)
                ps.append(sum(pred[v1:v+v1])/v)
                ta.append(te_y[v1])
                v1 += v
#             print(ta,va_y)
            ps = np.array(ps)
            ta = np.array(ta)
            error_rate = r2_score(y_true=ta,y_pred=ps)
            r = np.corrcoef(ta.reshape(ps.shape[0]),ps.reshape(ps.shape[0]))[0][1]
            mse = mean_squared_error(y_true=ta, y_pred=ps)
            # error_rate, r , mse= self.test(model, device, test_loader)
            mses.append(mse)
            average.append(error_rate)
            ave.append(r)
            # print(error_rate,r,end=" ")
        return sum(average)/3, sum(ave)/3, sum(mses)/3



