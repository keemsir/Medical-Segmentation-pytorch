import os
import argparse
import torch

import dicom_data_loader
from train_def import Train


##



def main():

    # # #   CUDA set   # # #
    torch.backends.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # # # #   PATH   # # # #
    cur_dir = os.getcwd()
    pt_dir = os.path.join(cur_dir, 'pt/')  # dataset
    ckpt_h = os.path.join(cur_dir, 'checkpoint/')
    ckpt_dir = os.path.join(ckpt_h, '{}_{}/'.format(args.task, args.loss_function))
    log_dir = os.path.join(cur_dir, 'log/')  # tensor log


    # Dataload
    train_loader, val_loader = dicom_data_loader.dataset_load(args, pt_dir)
    train = Train(args, train_loader, val_loader)

    train.train()





if __name__ == '__main__':

    # parser
    parser = argparse.ArgumentParser()


    # Basic
    parser.add_argument('--network_model', type=str, default='U_Net')
    parser.add_argument('--task', type=str, help='Can be task name or task uid', default='data_Oncologists_1')

    loss_list = ['BCEWithLogitsLoss', 'DiceLoss', 'DiceBCELoss', 'IoULoss', 'TverskyLoss', 'FocalLoss', 'FocalTverskyLoss'] # train_def.py Line 17
    loss_order = list(enumerate(loss_list))
    loss_index = int(input('{}\nInput loss function : '.format(loss_order)))
    loss_name = loss_list[loss_index]
    parser.add_argument('--loss_function', '-loss', type=str, default=loss_list[loss_index])

    # Parameter
    parser.add_argument('--batch_size', '-batch', type=int, default=4)
    parser.add_argument('--num_epoch', '-epoch', type=int, default=50)
    parser.add_argument('--learning_rate', '-lr', type=float, default=0.0001)

    args = parser.parse_args()

    print('Args :', args)

    main()



'''

    # fn_loss = nn.BCEWithLogitsLoss().to(device)  # nn.BCEWithLogitsLoss
    # fn_loss = FocalLoss().to(device)  # DiceLoss, DiceBCELoss, IoULoss, TverskyLoss, FocalLoss, ComboLoss
    # fn_loss = nn.MSELoss().to(device) # nn.BCEWithLogitsLoss, MSELoss

'''

##
# loss_order = ['BCE', 'Dice', 'DiceBCE', 'MSE', 'IoU', 'Tversky', 'Focal', 'Combo']

# for index, value in enumerate(loss_order):
#     print(index, '.', value, sep='')
#
# loss_order = list(enumerate(loss_order))
#
# print(loss_order)

