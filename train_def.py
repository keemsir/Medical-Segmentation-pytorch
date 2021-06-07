import numpy as np
import os
import logging
import time
from tqdm import tqdm

import torch
import torch.nn as nn

import loss_function
import network



class Train(object):
    def __init__(self, args, train_loader, val_loader):

        def LossSelection(self, loss_name):
            if loss_name == 'BCEWithLogitsLoss':
                criterion = loss_function.BCEWithLogitsLoss()
            elif loss_name == 'DiceLoss':
                criterion = loss_function.DiceLoss()
            elif loss_name == 'DiceBCELoss':
                criterion = loss_function.DiceBCELoss()
            elif loss_name == 'IoULoss':
                criterion = loss_function.IoULoss()
            elif loss_name == 'TverskyLoss':
                criterion = loss_function.TverskyLoss()
            elif loss_name == 'FocalLoss':
                criterion = loss_function.FocalLoss()
            elif loss_name == 'FocalTverskyLoss':
                criterion = loss_function.FocalTverskyLoss()
            return criterion

        # Data
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Hyper Parameters
        self.lr = args.learning_rate
        self.num_epoch = args.num_epoch
        self.batch_size = args.batch_size

        # Models
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = network.UNet().to(self.device)
        self.net = nn.DataParallel(self.net, device_ids=[0, 1])
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.loss = args.loss_function
        # 2021.06.01 remove
        # loss_fun = loss_function.(args.loss_function)
        # self.criterion = loss_function.BCEWithLogitsLoss()
        self.criterion = LossSelection(self, self.loss)

        # Path
        self.cur_dir = os.getcwd()
        self.pt_dir = os.path.join(self.cur_dir, 'pt/')  # dataset
        self.ckpt_h = os.path.join(self.cur_dir, 'checkpoint/')
        self.ckpt_dir = os.path.join(self.ckpt_h, '{}_{}/'.format(args.task, args.loss_function))
        self.log_dir = os.path.join(self.cur_dir, 'log/')  # tensor log

        # Logging
        self.logname = 'Log_' + args.task + '_' + args.network_model + '_' + args.loss_function + '.txt'
        self.filenameLog = os.path.join('./Logs', self.logname)

        logging.basicConfig(filename=self.filenameLog,
                            level=logging.INFO,
                            format='%(asctime)s,%(message)s',
                            datefmt='%Y-%m-%d %I:%M:%S')

        # Number
        self.num_train = len(self.train_loader)
        self.num_val = len(self.val_loader)
        # self.num_train_for_epoch = np.ceil(num_train / self.batch_size)  # np.ceil : 소수점 반올림
        # self.num_val_for_epoch = np.ceil(num_val / self.batch_size)


    # Network Save & Load
    def save(self, ckpt_dir, net, optim, epoch):
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir)

        torch.save({'net': net.state_dict(), 'optim': optim.state_dict()}, '%s/model_epoch%d.pth' % (ckpt_dir, epoch))


    def load(self, ckpt_dir, net, optim):
        if not os.path.exists(ckpt_dir):  # 저장된 네트워크가 없다면 인풋을 그대로 반환
            epoch = 0
            return net, optim, epoch

        ckpt_lst = os.listdir(ckpt_dir)  # ckpt_dir 아래 있는 모든 파일 리스트를 받아온다
        ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

        dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_lst[-1]))

        net.load_state_dict(dict_model['net'])
        optim.load_state_dict(dict_model['optim'])
        epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

        return net, optim, epoch




    def train(self):

        print('Starting the training !')

        # Epoch

        self.start_epoch = 0
        net, optim, self.start_epoch = Train.load(self, ckpt_dir=self.ckpt_dir, net=self.net, optim=self.optimizer)

        for epoch in range(self.start_epoch + 1, self.num_epoch + 1):

            net.train()

            print('\nLoss Function : {}({}), epoch {}/{}'.format(self.loss, self.criterion, epoch, self.num_epoch))

            loss_arr = []
            start_time = time.time()
            stream = tqdm(self.train_loader)

            for batch, (data, target) in enumerate(stream, start=1):
                # forward
                inputs = data.to(self.device)
                label = target.to(self.device)
                output = net(inputs)

                # backward
                optim.zero_grad()
                loss = self.criterion(output, label)  # Between output, label loss calculate
                loss.backward()  # gradient backpropagation
                optim.step()  # backpropa 된 gradient를 이용해서 각 layer의 parameters update

                # save loss
                loss_arr += [loss.item()]
                # count += 1

                # Display
                if batch % 20 == 0:
                    print('Train : Batch %04d \ %04d (%02d%%) | Loss %08f' % (
                        batch + 1, self.num_train, ((batch + 1) / self.num_train) * 100.,
                        loss.item()
                    ))

                    # np.mean(loss_arr)

                # tensorbord에 결과값들 저정하기
                # label = fn_tonumpy(label)
                # inputs = fn_tonumpy(fn_denorm(inputs,0.5,0.5))
                # output = fn_tonumpy(fn_classifier(output))

                # writer_train.add_image('label', label, num_train_for_epoch * (epoch - 1) + batch, dataformats='NHWC')
                # writer_train.add_image('input', inputs, num_train_for_epoch * (epoch - 1) + batch, dataformats='NHWC')
                # writer_train.add_image('output', output, num_train_for_epoch * (epoch - 1) + batch, dataformats='NHWC')

            # writer_train.add_scalar('loss', np.mean(loss_arr), epoch)

            # validation
            with torch.no_grad():
                net.eval()  # only evaluation
                correct = 0
                tensor_size = 0
                test_loss = []
                acc = 0
                tp, fp, fn, tn, preci, rec, f1_score, preci_f1, rec_f1, TPR, FPR = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                TP_es, FP_es, FN_es, TN_es = 0, 0, 0, 0
                count = 0
                # stream = tqdm(self.val_loader)

                for batch, (data, target) in enumerate(self.val_loader):
                    # forward
                    inputs = data.to(self.device)
                    label = target.to(self.device)
                    output = net(inputs)

                    # loss
                    loss = self.criterion(output, label)
                    # print(loss.item()) # loss information monitoring
                    test_loss += [loss.item()]

                    # Confusion Matrix
                    THR = torch.min(output) + (torch.max(output) - torch.min(output)) / 2
                    count += 1

                    b_flat = label.view(4, -1)
                    pred_flat = output.view(4, -1)

                    pred = torch.zeros_like(pred_flat)
                    thr = THR
                    pred[pred_flat >= thr] = 1
                    pred[pred_flat < thr] = 0

                    tp = torch.sum(b_flat * pred, dim=1)
                    fp = torch.sum(pred, dim=1) - tp
                    fn = torch.sum(b_flat, dim=1) - tp
                    tn = torch.numel(label) - tp

                    TP = torch.mean(tp)
                    FP = torch.mean(fp)
                    FN = torch.mean(fn)
                    TN = torch.mean(tn)

                    acc += (TP + TN) / (TP + FP + FN + TN)
                    preci_f1 = TP / (TP + FP + self.lr)
                    rec_f1 = TP / (TP + FN + self.lr)
                    TPR += TP / (TP + FN + self.lr)  # True Positive Rate
                    FPR += FP / (FP + TN + self.lr)  # False Positive Rate

                    preci += preci_f1
                    rec += rec_f1
                    f1_score += (2 * (preci_f1 * rec_f1)) / (preci_f1 + rec_f1)

                    TP_es += TP  # Epoch Sum
                    FP_es += FP
                    FN_es += FN
                    TN_es += TN

                    # Tensorboard 저장하기
                    # label = fn_tonumpy(label)
                    # inputs = fn_tonumpy(fn_denorm(inputs, mean=0.5, std=0.5))
                    # output = fn_tonumpy(fn_classifier(output))

                    # writer_val.add_image('label', label, num_val_for_epoch * (epoch - 1) + batch, dataformats='NHWC')
                    # writer_val.add_image('input', inputs, num_val_for_epoch * (epoch - 1) + batch, dataformats='NHWC')
                    # writer_val.add_image('output', output, num_val_for_epoch * (epoch - 1) + batch, dataformats='NHWC')

                # Log
                logging.info(
                    'Epoch : ,{:3d}, Train Loss : ,{:0.8f}, Validation Loss : ,{:0.8f}, Accuracy : ,{:0.8f}, Precision : ,{:0.8f}, Recall : ,{:0.8f}, F1_score : ,{:0.8f}, TP : ,{:0.8f}, FP : ,{:0.8f}, FN : ,{:0.8f}, TN : ,{:0.8f}'
                        .format(epoch, np.mean(loss_arr), np.mean(test_loss), acc / count, preci / count, rec / count,
                                (2 * (preci / count) * (rec / count) / (preci / count + rec / count)), TP_es / count,
                                FP_es / count,
                                FN_es / count, TN_es / count))
                print(count)

            # writer_val.add_scalar('loss', np.mean(loss_arr), epoch)
            # writer_acc.add_scalar('acc', epoch)

            # epoch이 끝날때 마다 네트워크 저장
            Train.save(self, ckpt_dir=self.ckpt_dir, net=self.net, optim=self.optimizer, epoch=epoch)

            # Time
            end_time = time.time() - start_time
            print('Training Complete in {:.0f}m {:.0f}s'.format(end_time // 60, end_time % 60))

            print('\nTrain set - Average Loss : %04f, Accuracy(incompletion..) : (%02d%%)\n' % (
                np.mean(loss_arr),
                np.mean(loss_arr) * 100.))

        # writer_train.close()
        # writer_val.close()
        print('{} Training Complete !! Epoch : {}'.format(self.criterion, self.start_epoch+1))
        print("Saved file name : '{}'".format(self.logname))




##
if __name__ == '__main__':
    pass