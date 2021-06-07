import numpy as np
import os
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import Dataset

import matplotlib.pyplot as plt

from tqdm import tqdm
import albumentations
import albumentations.pytorch
from albumentations.pytorch import ToTensorV2
from scipy.io import loadmat

from sklearn.model_selection import KFold




def dataset_load(args, pt_dir):

    train_dir = os.path.join(pt_dir, '{}.pt'.format(args.task))
    val_dir = os.path.join(pt_dir, '{}.pt'.format(args.task))

    train_dataset = torch.load(train_dir)
    val_dataset = torch.load(val_dir)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False)


    return train_loader, val_loader



class AlbumentationsDataset(Dataset):
    """TensorDataset with support of transforms."""
    #def __init__(self, tensors, batchSize=4, shuffle=False, transform=None):
    def __init__(self, tensors, transform=None):
        #assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform


    def __len__(self):
        return len(self.tensors[0])


    def __getitem__(self, index):
        x = self.tensors[0][index]
        y = self.tensors[1][index]

        #start_t = time.time()
        #print('transform start !')
        sample = {"image": x, "mask": y}

        if self.transform:
            augmented = self.transform(**sample)
            x = augmented['image']
            y = augmented['mask']

        #total_time = (time.time()-start_t)
        #print('Calculation time: {}'.format(total_time))
        return x, y


##

class AlbumentationsDataset_m(Dataset):
    """TensorDataset with support of transforms."""
    #def __init__(self, tensors, batchSize=4, shuffle=False, transform=None):
    def __init__(self, tensors, transform=None):
        #assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform


    def __len__(self):
        return len(self.tensors[0])


    def __getitem__(self, index):
        x = self.tensors[0][index]
        y = self.tensors[1][index]

        #start_t = time.time()
        #print('transform start !')
        sample = {"image": x, "mask": y}

        if self.transform:
            augmented = self.transform(**sample)
            x = augmented['image']
            y = augmented['mask']

        #total_time = (time.time()-start_t)
        #print('Calculation time: {}'.format(total_time))
        return x, y

##

path_mat = os.path.join(os.getcwd(), 'DB_NSCLC/')
batch_size = 4

def get_mat(path_mat=path_mat, batch_size=batch_size,
            oncologists_NUM=1, KF_NUM=5, CVNUM=1, num_workers=2): # num_workers = processor number

    # # path_mat = Make it a folder containing oncologists
    # # oncologists_NUM = folder name example = 'Oncologists_1' <- '1'

    oncologists = 'Oncologists_{}'.format(oncologists_NUM)

    oncol_1_ct = os.path.join(path_mat, '{}/ct/'.format(oncologists))
    oncol_1_mask = os.path.join(path_mat, '{}/mask/'.format(oncologists))
    # NSCLC_list = os.listdir(path_mat)
    # file_list = [os.path.join(path_mat, f) for f in os.listdir(path_mat) if f[:11] == 'Oncologists']
    file_list_ct = [os.path.join(oncol_1_ct, f) for f in os.listdir(oncol_1_ct) if
                    os.path.isfile(os.path.join(oncol_1_ct, f))]
    file_list_mask = [os.path.join(oncol_1_mask, f) for f in os.listdir(oncol_1_mask) if
                    os.path.isfile(os.path.join(oncol_1_mask, f))]
    file_list_ct.sort()
    file_list_mask.sort()


    kfold = KFold(n_splits=KF_NUM, shuffle=False)
    train_indexes = {}
    test_indexes = {}
    # rn = range(df.__len__())  # df = dataframe(Data)
    counter = 1

    for train_index, test_index in kfold.split(range(len(file_list_ct))):
        train_indexes['trainIndex_CV{0}'.format(counter)] = train_index
        test_indexes['testIndex_CV{0}'.format(counter)] = test_index
        counter += 1
        print(train_index, test_index)

    trainDickey_list = list(train_indexes.keys())
    trainIndex_CV_F = train_indexes.get(trainDickey_list[CVNUM - 1])

    testDickey_list = list(test_indexes.keys())
    testIndex_CV_F = test_indexes.get(testDickey_list[CVNUM - 1])

# # # # # # # # # # Train Data Set # # # # # # # # # # # # # # # # # # # #

    images = np.zeros((0, 512, 512))
    masks = np.zeros((0, 512, 512))


    for iter1 in tqdm(trainIndex_CV_F):
        filename_ct_temp = file_list_ct[iter1]
        filename_mask_temp = file_list_mask[iter1]

        trainCTtemp = loadmat(filename_ct_temp)

        try:
            trainCTtemp = trainCTtemp['ctImagesData_rot_zeroNorm_HalfNHalf']
            trainCTtemp = np.transpose(trainCTtemp, (2, 0, 1))
            images = np.concatenate((images, trainCTtemp), axis=0)
        except:
            print("trainCTtemp['ctImagesData_rot_zeroNorm_HalfNHalf'] error")

        trainMASKtemp = loadmat(filename_mask_temp)
        trainMASKtemp = trainMASKtemp['tempOncologistMaskF']
        trainMASKtemp = np.transpose(trainMASKtemp, (2, 0, 1))
        masks = np.concatenate((masks, trainMASKtemp), axis=0)

    images = images.astype('float32')
    images = torch.from_numpy(images)
    images = images.view(-1, 1, 512, 512)
    masks = masks.astype('float32')  # umm...
    masks = torch.from_numpy(masks)
    masks = masks.view(-1, 1, 512, 512)

    train_dataset = TensorDataset(images, masks)


# # # # # # # # # # Validation Data Set # # # # # # # # # # # # # # # # # # # #


    images_t = np.zeros((0, 512, 512))
    masks_t = np.zeros((0, 512, 512))

    for iter1 in tqdm(testIndex_CV_F):
        filename_ct_temp = file_list_ct[iter1]
        filename_mask_temp = file_list_mask[iter1]

        testCTtemp = loadmat(filename_ct_temp)

        try:
            testCTtemp = testCTtemp['ctImagesData_rot_zeroNorm_HalfNHalf']
            testCTtemp = np.transpose(testCTtemp, (2, 0, 1))
            images_t = np.concatenate((images_t, testCTtemp), axis=0)
        except:
            print("testCTtemp['ctImagesData_rot_zeroNorm_HalfNHalf'] error")

        testMASKtemp = loadmat(filename_mask_temp)

        testMASKtemp = testMASKtemp['tempOncologistMaskF']
        testMASKtemp = np.transpose(testMASKtemp, (2, 0, 1))
        masks_t = np.concatenate((masks_t, testMASKtemp), axis=0)

    images_t = images_t.astype('float32')
    images_t = torch.from_numpy(images_t)
    images_t = images_t.view(-1, 1, 512, 512)
    masks_t = masks_t.astype('float32')  # umm...
    masks_t = torch.from_numpy(masks_t)
    masks_t = masks_t.view(-1, 1, 512, 512)

    test_dataset = TensorDataset(images_t, masks_t)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

    print('Data load Completed !!')
    print("{}'s data".format(oncologists))
    print('Train image:{}, Train mask:{}'.format(images.shape, masks.shape))
    print('Test image:{}, Test mask:{}'.format(images_t.shape, masks_t.shape))



    albumentations_transform = albumentations.Compose([
        albumentations.HorizontalFlip(),  # Same with transforms.RandomHorizontalFlip()
        albumentations.Rotate(),
        # ToTensorV2()
    ])

    albumentations_transform_testSet = albumentations.Compose([
    ])

    train_dataset_transform = AlbumentationsDataset(tensors=(images, masks), transform=albumentations_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset_transform, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dataset_transform = AlbumentationsDataset(tensors=(images_t, masks_t), transform=albumentations_transform_testSet)
    test_loader = torch.utils.data.DataLoader(test_dataset_transform, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # return train_loader, test_loader
    return train_dataset, test_dataset
    # return images, masks


'''
    # images_total = np.zeros((0, 512, 512))
    # masks_total = np.zeros((0, 512, 512))

    for i in file_list: # Oncologists list
        images = np.zeros((0, 512, 512))
        masks = np.zeros((0, 512, 512))

        ct_dir = os.path.join(i, 'ct')
        ct_list = os.listdir(ct_dir)
        ct_list.sort()

        mask_dir = os.path.join(i, 'mask')
        mask_list = os.listdir(mask_dir)
        mask_list.sort()

        if len(ct_list) == len(mask_list):
            for j in range(len(ct_list)): # Per person
                ct_list_tmp = os.path.join(ct_dir, ct_list[j])
                mask_list_tmp = os.path.join(mask_dir, mask_list[j])

                images_tmp = loadmat(ct_list_tmp)
                images_tmp = images_tmp['ctImagesData_rot_zeroNorm_HalfNHalf']
                images_tmp = np.transpose(images_tmp, (2, 0, 1))
                images = np.concatenate((images, images_tmp), axis=0)

                masks_tmp = loadmat(mask_list_tmp)
                masks_tmp = masks_tmp['tempOncologistMaskF']
                masks_tmp = np.transpose(masks_tmp, (2, 0, 1))
                masks = np.concatenate((masks, masks_tmp), axis=0)

        # # # Total concatenate # # #
        # images_total = np.concatenate((images_total, images), axis=0)
        # masks_total = np.concatenate((masks_total, masks), axis=0)

        images = images.astype('float32')
        images = torch.from_numpy(images)
        images = images.view(-1, 1, 512, 512)

        masks = masks.astype('float32')
        masks = torch.from_numpy(masks)
        masks = masks.view(-1, 1, 512, 512)

        train_dataset = TensorDataset(images, masks)

        data_pt_name = 'data_{}.pt'.format(i[-13:])
        data_pt = os.path.join(pt_dir, data_pt_name)

        torch.save(train_dataset, data_pt)
        print('Saved as {} Completed'.format(data_pt_name))

'''

##

train, mask = get_mat()


##
'''

##
NUM = 7

plt.subplot(2,3,1)
plt.imshow(images[NUM,:,:])
plt.subplot(2,3,2)
plt.imshow(images[NUM+1,:,:])
plt.subplot(2,3,3)
plt.imshow(images[NUM+2,:,:])

plt.subplot(2,3,4)
plt.imshow(masks[NUM,:,:])
plt.subplot(2,3,5)
plt.imshow(masks[NUM+1,:,:])
plt.subplot(2,3,6)
plt.imshow(masks[NUM+2,:,:])
'''

##


albumentations_transform = albumentations.Compose([
    albumentations.HorizontalFlip(),  # Same with transforms.RandomHorizontalFlip()
    albumentations.Rotate(),
])


albumentations_transform_testSet = albumentations.Compose([ToTensorV2])

train_dataset_transform = AlbumentationsDataset(tensors=(images, masks), transform=albumentations_transform)
train_loader = torch.utils.data.DataLoader(train_dataset_transform, batch_size=4, shuffle=True,
                                           num_workers=2)

'''
test_dataset_transform = AlbumentationsDataset(tensors=(images_t, masks_t), transform=albumentations_transform_testSet)
test_loader = torch.utils.data.DataLoader(test_dataset_transform, batch_size=batch_size, shuffle=False,
                                          num_workers=num_workers)
'''

train_loader = torch.utils.data.DataLoader(a, batch_size=4, shuffle=True)
a = next(iter(train_loader))

plt.subplot(1,2,1)
plt.imshow(a[0,0,0,:,:])
plt.subplot(1,2,2)
plt.imshow(a[0,1,0,:,:])

##

a = AlbumentationsDataset_m(train)





