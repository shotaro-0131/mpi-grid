import torch
from sklearn.metrics import r2_score
from Net import Net
import numpy as np
from sklearn.model_selection import KFold, train_test_split
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

    def test(self,model, device, test_loader):
        model.eval()
        error = 0
        with torch.no_grad():
            y_pred = []
            y_true = []
            for data, target in test_loader:
                data, target = data.to(device, dtype=torch.float32), target.to(device, dtype=torch.float32)
                pred = model(data).view(data.shape[0])
                
                y_pred.append(np.mean(pred.cpu().numpy()))
                y_true.append(target[0].cpu().numpy())

            error = r2_score(y_true=y_true,y_pred=y_pred)

            r = np.corrcoef(y_true,y_pred)[0][1]

            mse = mean_squared_error(y_true=y_true,y_pred=y_pred)

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
            va_id, te_id, _, _ = train_test_split(test_index, test_index,random_state=2525, test_size=0.5)
            f_v_id =np.array([i for i, k in enumerate(self.tr_id) if k in va_id])
            v_num =np.array([len([j for j,x in enumerate(self.tr_id) if k == x]) for i, k in enumerate(va_id)])
            t_num =np.array([len([j for j,x in enumerate(self.tr_id) if k == x]) for i, k in enumerate(te_id)])
            f_t_id =np.array([i for i, k in enumerate(self.tr_id) if k in te_id])
            va_x = self.tr_x[f_v_id]
            va_y = self.tr_y[f_v_id]
            te_x = self.tr_x[f_t_id]
            te_y = self.tr_y[f_t_id]
            train_set = torch.utils.data.TensorDataset(torch.from_numpy(np.array(tr_x).reshape(len(tr_x),155,1,14)).float(),
                                    torch.from_numpy(np.array(tr_y).astype(np.float32)))
            test_set = torch.utils.data.TensorDataset(torch.from_numpy(np.array(te_x).reshape(len(te_x),155,1,14)).float(),
                                            torch.from_numpy(te_y.astype(np.float32)))
            validation_set = torch.utils.data.TensorDataset(torch.from_numpy(np.array(va_x).reshape(len(va_x),155,1,14)).float(),
                                            torch.from_numpy(va_y.astype(np.float32)))
            train_loader = DataLoader(train_set, batch_size=BATCHSIZE, shuffle=True, num_workers=2)
            test_loader = DataLoader(test_set, batch_size=14, shuffle=False, num_workers=2)
            validation_loader = DataLoader(validation_set, batch_size=14, shuffle=False, num_workers=2)
        

            model = Net(self.num_layers, self.kernel_sizes, self.num_filters, self.poolings, self.pooling_size, self.mid_unit).to(device)
            optimizer = self.get_optimizer(model)

            best = 0
            best_model = None
            for step in range(EPOCH):
                # print('*',end="")
                model = self.train(model, device, train_loader, optimizer)

                error_rate, r, mse = self.test(model, device, validation_loader)
                if 1 - error_rate < 1 - best:
                    best_model = model
                    best = error_rate
            error_rate, r, mse = self.test(best_model, device, test_loader)
            
            
                
            mses.append(mse)
            average.append(error_rate)
            ave.append(r)

        return sum(average)/3, sum(ave)/3, sum(mses)/3



