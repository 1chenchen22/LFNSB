import glob
import logging
import os
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms, datasets
from PIL import Image
from torch.optim.lr_scheduler import CosineAnnealingLR
import itertools
from losses import AffinityLoss
from networks.MFNSB2 import MFN
import torch.nn.functional as F
from torch.utils.data import Sampler, Dataset
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from sam import SAM
eps = sys.float_info.epsilon
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from itertools import product as itertools_product


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j] * 100, fmt) + '%',
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Actual', fontsize=18)
    plt.xlabel('Predicted', fontsize=18)
    plt.tight_layout()


def load_pretrained(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['iter']
    return model, optimizer, epoch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--aff_path', type=str, default='',
                        help='AffectNet dataset path.')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate for adam.')
    parser.add_argument('--workers', default=8, type=int, help='Number of data loading workers.')
    parser.add_argument('--epochs', type=int, default=40, help='Total training epochs.')
    parser.add_argument('--num_class', type=int, default=8, help='Number of class.')
    parser.add_argument('--exp_name', type=str, default='huatu2', help='Name of experiment.')
    parser.add_argument('--checkpoint_path', type=str, default='',help='Name of experiment.'
                         )

    return parser.parse_args()


class AffectNet(data.Dataset):
    def __init__(self, aff_path, phase, use_cache=True, transform=None):
        self.phase = phase
        self.transform = transform
        self.aff_path = aff_path

        if use_cache:
            cache_path = os.path.join(aff_path, 'affectnet.csv')
            if os.path.exists(cache_path):
                df = pd.read_csv(cache_path)
            else:
                df = self.get_df()
                df.to_csv(cache_path)
        else:
            df = self.get_df()

        self.data = df[df['phase'] == phase]

        self.file_paths = self.data.loc[:, 'img_path'].values
        self.label = self.data.loc[:, 'label'].values

        _, self.sample_counts = np.unique(self.label, return_counts=True)
        # print(f' distribution of {phase} samples: {self.sample_counts}')

    def get_df(self):
        train_path = os.path.join(self.aff_path, 'train_set/')
        val_path = os.path.join(self.aff_path, 'val_set/')
        data = []

        for anno in glob.glob(train_path + 'annotations/*_exp.npy'):
            idx = os.path.basename(anno).split('_')[0]
            img_path = os.path.join(train_path, f'images/{idx}.jpg')
            label = int(np.load(anno))
            data.append(['train', img_path, label])

        for anno in glob.glob(val_path + 'annotations/*_exp.npy'):
            idx = os.path.basename(anno).split('_')[0]
            img_path = os.path.join(val_path, f'images/{idx}.jpg')
            label = int(np.load(anno))
            data.append(['val', img_path, label])

        return pd.DataFrame(data=data, columns=['phase', 'img_path', 'label'])

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = Image.open(path).convert('RGB')
        label = self.label[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


# class ImbalancedDatasetSampler(Sampler):
#     def __init__(self, dataset):
#         self.dataset = dataset
#
#         labels = [label for _, label in dataset]
#         labels, counts = np.unique(labels, return_counts=True)
#         label_count_dict = dict(zip(labels, counts))
#
#         weights = [1.0 / label_count_dict[label] for _, label in dataset]
#         self.weights = torch.DoubleTensor(weights)
#
#     def __iter__(self):
#         return iter(torch.multinomial(self.weights, len(self.dataset), replacement=True).tolist())
#
#     def __len__(self):
#         return len(self.dataset)



class_names = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Angry']


def run_training():
    args = parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log_name = args.exp_name
    if not os.path.isdir('./logs'):
        os.mkdir('./logs')
    if not os.path.isdir('./checkpoints'):
        os.mkdir('./checkpoints')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',

                        datefmt='%Y-%m-%d %H:%M',
                        handlers=[logging.FileHandler('./logs/' + log_name + '.log', 'w', 'utf-8')])
    logging.getLogger('numexpr').setLevel(logging.WARNING)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True
    model = MFNSB(num_class=7, num_head=args.num_head)

    model.to(device)

    data_transforms = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.RandomAffine(20, scale=(0.8, 1), translate=(0.2, 0.2)),
        ], p=0.7),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(),
    ])

    # train_dataset = datasets.ImageFolder(f'{args.aff_path}/train_set', transform = data_transforms)
    train_dataset = AffectNet(args.aff_path, phase='train', transform=data_transforms)

    if args.num_class == 7:
        idx = [i for i, (_, label) in enumerate(train_dataset) if label != 7]
        train_dataset = data.Subset(train_dataset, idx)

    print('Whole train set size:', train_dataset.__len__())
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               sampler=ImbalancedDatasetSampler(train_dataset),
                                               shuffle=False,
                                               pin_memory=True)

    data_transforms_val = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    # val_dataset = datasets.ImageFolder(f'{args.aff_path}/val_set', transform = data_transforms_val)
    val_dataset = AffectNet(args.aff_path, phase='val', transform=data_transforms_val)  # loading dynamically
    #
    # if args.num_class == 7:
    #     idx = [i for i, (_, label) in enumerate(val_dataset) if label != 7]
    #     val_dataset = data.Subset(val_dataset, idx)

    print('Validation set size:', val_dataset.__len__())

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             num_workers=args.workers,
                                             shuffle=False,
                                             pin_memory=True)

    criterion_af = Cosine-Harmonyloss()
    criterion_cls = torch.nn.CrossEntropyLoss().to(device)

    params = list(model.parameters())
    # optimizer = torch.optim.Adam(params,args.lr,weight_decay = 0)
    base_optimizer = torch.optim.SGD

    optimizer = SAM(model.parameters(), torch.optim.Adam, lr=args.lr, rho=0.05, adaptive=False, )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.98)
    # optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=1e-4, momentum=0.9)
    # 定义余弦退火调度器
    #scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=4e-4)
    # if os.path.exists(args.checkpoint_path):
    #     model, optimizer, start_epoch = load_pretrained(model, optimizer, args.checkpoint_path)
    #     tqdm.write("Pretrained model loaded. Starting from epoch %d" % start_epoch)
    # else:
    #     tqdm.write("No pretrained model found. Starting training from scratch.")
    #     start_epoch = 1

    best_acc = 0
    for epoch in tqdm(range(1, args.epochs + 1)):
        running_loss = 0.0
        correct_sum = 0
        iter_cnt = 0
        model.train()

        for (imgs, targets) in train_loader:
            iter_cnt += 1
            optimizer.zero_grad()
            imgs = imgs.to(device)
            targets = targets.to(device)

            out, feat, heads = model(imgs)

            loss = criterion_cls(out, targets) +criterion_af(feat, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.first_step(zero_grad=True)

            imgs = imgs.to(device)
            targets = targets.to(device)

            out, feat, heads = model(imgs)
            loss = criterion_cls(out, targets) +criterion_af(feat, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.second_step(zero_grad=True)
            # loss.backward()
            # optimizer.step()
            # 计算锐度
            sharpness = loss.item()
            running_loss += loss
            _, predicts = torch.max(out, 1)
            correct_num = torch.eq(predicts, targets).sum()
            correct_sum += correct_num

        acc = correct_sum.float() / float(train_dataset.__len__())
        running_loss = running_loss / iter_cnt
        # 记录学习率变化
        current_lr = optimizer.param_groups[0]['lr']
        tqdm.write('[Epoch %d] Training accuracy: %.4f. Loss: %.3f. LR %.6f' % (
        epoch, acc, running_loss, optimizer.param_groups[0]['lr']))

        logging.info("[Epoch %d]  Training accuracy:  %.4f. Loss: %.3f. Learning rate: %.6f" % (
            epoch, acc, running_loss, current_lr))
        with torch.no_grad():
            running_loss = 0.0
            iter_cnt = 0
            bingo_cnt = 0
            sample_cnt = 0
            model.eval()
            for imgs, targets in val_loader:
                imgs = imgs.to(device)
                targets = targets.to(device)
                out, feat, heads = model(imgs)
                loss = criterion_cls(out, targets) + 0.1 * criterion_at(heads)
                running_loss += loss
                iter_cnt += 1
                _, predicts = torch.max(out, 1)
                correct_num = torch.eq(predicts, targets)
                bingo_cnt += correct_num.sum().cpu()
                sample_cnt += out.size(0)

            running_loss = running_loss / iter_cnt
            scheduler.step()

            acc = bingo_cnt.float() / float(sample_cnt)
            acc = np.around(acc.numpy(), 4)

            # y_true = np.concatenate(y_true)
            # y_pred = np.concatenate(y_pred)
            # balanced_acc = np.around(balanced_accuracy_score(y_true, y_pred), 4)

            best_acc = max(acc, best_acc)
            tqdm.write("[Epoch %d] Validation accuracy:%.4f. Loss:%.3f" % (epoch, acc, running_loss))
            tqdm.write("best_acc:" + str(best_acc))
            logging.info("[Epoch %d]  Validation accuracy:  %.4f. Loss: %.3f" % (
                epoch, acc, running_loss))
            logging.info("Best_acc:" + str(best_acc))

        if args.num_class == 7 and acc == best_acc:
            torch.save({'iter': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(), },
                       os.path.join('checkpoints', "affecnet7_epoch" + str(epoch) + "_acc" + str(acc) + ".pth"))
            tqdm.write('Model saved.')
            # Compute confusion matrix
            matrix = confusion_matrix(targets.data.cpu().numpy(), predicts.cpu().numpy())
            np.set_printoptions(precision=2)

            # Plot and save confusion matrix
            plt.figure(figsize=(10, 8))
            plot_confusion_matrix(matrix, classes=class_names, normalize=True,
                                  title='aff-DB Confusion Matrix (acc: %0.2f%%)' % (acc * 100))
            # Ensure the save directory exists
            save_dir = 'checkpoints'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.savefig(os.path.join('checkpoints', "aff_epoch" + str(epoch) + "_acc" + str(acc) + "_bacc" + ".png"))
            plt.close()


        elif args.num_class == 8 and acc == best_acc:
            torch.save({'iter': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(), },
                       os.path.join('checkpoints', "affecnet8_epoch" + str(epoch) + "_acc" + str(acc) + ".pth"))
            tqdm.write('Model saved.')
            # Compute confusion matrix
            matrix = confusion_matrix(targets.data.cpu().numpy(), predicts.cpu().numpy())
            np.set_printoptions(precision=2)

            # Plot and save confusion matrix
            plt.figure(figsize=(10, 8))
            plot_confusion_matrix(matrix, classes=class_names, normalize=True,
                                  title='affnet-8 Confusion Matrix (acc: %0.2f%%)' % (acc * 100))
            # Ensure the save directory exists
            save_dir = 'checkpoints'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.savefig(os.path.join('checkpoints', "aff_epoch" + str(epoch) + "_acc" + str(acc) + "_bacc" + ".png"))
            plt.close()


if __name__ == "__main__":
    run_training()