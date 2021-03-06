from mpi4py import MPI
import numpy as np
import pandas as pd
import time
import sys
import itertools
import Evaluator

def params():
    num_layer = [1,2,3,4]
    kernel_sizes = [3,5]
    num_filters = [32, 64, 128]
    pooling = [0, 1]
    mid_unit = [100, 200, 400]
    # mid_unit2 = [100, 200, 400]
    learning_rate = [1,2,3]

    params = []
    for n in num_layer:

        for pool in pooling:

            for m in mid_unit:

                # for m2 in mid_unit2:

                    for lr in learning_rate:


                        k_sizes = [kernel_sizes for i in range(n)]
                        while len(k_sizes) < 4:
                            k_sizes.append([0])
                        n_filters = [num_filters for i in range(n)]
                        while len(n_filters) < 4:
                            n_filters.append([0])




                        for k1 in k_sizes[0]:
                            for k2 in k_sizes[1]:
                                for k3 in k_sizes[2]:
                                    for k4 in k_sizes[3]:
                                        # for k5 in k_sizes[4]:
        #                                     for k6 in k_sizes[5]:

                                                    for f1 in n_filters[0]:
                                                        for f2 in n_filters[1]:
                                                            for f3 in n_filters[2]:
                                                                for f4 in n_filters[3]:
        #                                                             for f5 in n_filters[4]:

                                                                            p = [n]

                                                                            p.append(k1)
                                                                            p.append(k2)
                                                                            p.append(k3)
                                                                            p.append(k4)
                                                                            # p.append(k5)
        #                                                                     p.append(k6)
                                                                            p.append(f1)
                                                                            p.append(f2)
                                                                            p.append(f3)
                                                                            p.append(f4)
        #                                                                     p.append(f5)
                                                                            p.append(pool)
                                                                            p.append(m)
                                                                            # p.append(m2)
                                                                            p.append(lr)
                                                                            params.append(p)
    return params


def main():

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    print(rank)
    BATCH_SIZE = 2

    if rank == 0:


        # Model params


        #set up sendbuf and recvbuf
        l = params()
        # l = pd.read_csv("params.csv").drop(["ID"],axis="columns").values.tolist()
        train_x = pd.read_csv("mpitest/train_x.csv").values.astype(np.float32).reshape(1,-1)
        test_x = pd.read_csv("mpitest/test_x.csv").drop(["ID"],axis=1).values.astype(np.float32).reshape(1,-1)
        train_y = pd.read_csv("mpitest/train_y.csv").drop(["ID"],axis=1).values.astype(np.float32)
        test_y = pd.read_csv("mpitest/test_y.csv").drop(["ID"],axis=1).values.astype(np.float32)
        send_list = np.array(l, dtype=np.int32)
        print(len(l))

        recv_list = np.zeros(int(len(l)/BATCH_SIZE+1)*BATCH_SIZE*2, dtype=np.float32).reshape(int(len(l)/BATCH_SIZE+1)*BATCH_SIZE,2)

        # Initialize
        req = []

        for i in range(size-1):
            comm.Send(train_x,dest=i+1, tag=50)
            comm.Send(train_y,dest=i+1, tag=60)
            comm.Send(test_x,dest=i+1, tag=70)
            comm.Send(test_y,dest=i+1, tag=80)
            comm.Send(send_list[i*BATCH_SIZE:(i+1)*BATCH_SIZE], dest=i+1, tag=10)

        for i in range(size-1):
            req.append(comm.Irecv(recv_list[i*BATCH_SIZE:(i+1)*BATCH_SIZE], source=i+1, tag=20))

        stock = np.zeros(size-1).tolist()
        i = size

        while i < len(l):
            if sum(stock)!=0:
                target = stock.index(1)

                comm.Send(send_list[i:i+BATCH_SIZE], dest=target+1, tag=10)

                req[target] = comm.Irecv(recv_list[i:i+BATCH_SIZE], source=target+1, tag=20)

                i += BATCH_SIZE

            for j, k  in enumerate(req):
                if k.Get_status() == True :
                    stock[j] = 1
                    print(recv_list)
                    # pd.DataFrame(data=[recv_list
                else:
                    stock[j] = 0

        while sum(stock) != size-1:
            for j, k  in enumerate(req):

                stock[j] = 1 if k.Get_status() == True  else 0

        for i in range(size-1):

            comm.Isend([0, MPI.INT], dest=i+1, tag=30)
            comm.Isend([0, MPI.INT], dest=i+1, tag=20)

        # print(np.array(send_list).shape, np.array(recv_list).reshape(BATCH_SIZE,2)[:len(send_list)].shape)
        pd.DataFrame(data=np.concatenate([np.array(send_list), np.array(recv_list).reshape(int(len(l)/BATCH_SIZE+1)*BATCH_SIZE,2)[:len(l)]],axis=1)).to_csv("output.csv")

    # worker
    else:
        train_x = np.empty(9128*(14*155+1), dtype=np.float32)
        test_x = np.empty(840*14*155, dtype=np.float32)
        train_y = np.empty(9128, dtype=np.float32)
        test_y = np.empty(840, dtype=np.float32)
        comm.Recv(train_x,source=0, tag=50)
        comm.Recv(train_y, source=0, tag=60)
        comm.Recv(test_x, source=0, tag=70)
        comm.Recv(test_y, source=0, tag=80)
        finish = comm.irecv(source=0, tag=30)

        evaluator = Evaluator.Evaluator(train_x.reshape(9128, 14*155+1), train_y, test_x, test_y)
        print("%d is setup" %rank )

        while  finish.Get_status() == False:

            rdata = np.zeros(12*BATCH_SIZE, dtype=np.int32)
            req = comm.Irecv(rdata, source=0, tag=10)
            while True:
                if finish.Get_status() == True:
                    sys.exit(0)
                if req.Get_status() == True:
                    break


            rdata = rdata.reshape(BATCH_SIZE, 12)

            # TODO

            data = np.zeros(BATCH_SIZE*2, dtype=np.float32).reshape(BATCH_SIZE,2)

            for i in range(rdata.shape[0]):
                if rdata[i][0] == 0:
                    break
                evaluator.setParams(rdata[i])
                data[i][0],data[i][1] = evaluator.run()


            print(data)

            comm.Send(data,dest=0,tag=20)

        print("finish")


if __name__ == '__main__':
    print("begin")
    main()
