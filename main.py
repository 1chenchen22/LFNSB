import logging
import cv2
import os
import sys
from tqdm import tqdm
import argparse
import pandas as pd
import numpy as np
from PIL import Image
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
import torch.nn as nn
import torch.utils.data as data
from torchvision import transforms, datasets
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt
import itertools
import torch.nn.functional as F

from networks.MFNSB2 import MFNSB

from sklearn.metrics import confusion_matrix
from losses import AffinityLoss
eps = sys.float_info.epsilon
#方法4和5
def parse_args():
    parser = argparse.ArgumentParser()
   # parser.add_argument('--raf_path', type=str, default='/data/rafdb/', help='Raf-DB dataset path.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate for sgd.')
    parser.add_argument('--workers', default=7, type=int, help='Number of data loading workers.')
    parser.add_argument('--epochs', type=int, default=40, help='Total training epochs.')
    parser.add_argument('--num_head', type=int, default=2, help='Number of attention head.')
    parser.add_argument('--exp_name', type=str, default='loss6', help='Name of experiment.')
    parser.add_argument('--raf_path', type=str, default='',
                        help='Raf-DB dataset path.')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Name of experiment.')

    return parser.parse_args()
def load_pretrained(model, optimizer, checkpoint_path):
    try:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['iter']
        return model, optimizer, epoch, True
    except (EOFError, RuntimeError, KeyError) as e:
        print(f"Error loading checkpoint: {e}")
        return model, optimizer, 0, False



class RafDataSet(data.Dataset):
    def __init__(self, raf_path, phase, transform=None):
        self.phase = phase  # 数据集的阶段，训练或测试
        self.transform = transform  # 数据集的变换操作
        self.raf_path = raf_path  # 数据集的路径

        # 读取包含图像名称和标签的文件
        df = pd.read_csv(os.path.join(self.raf_path, 'EmoLabel/list_patition_label.txt'),
                         sep=' ', header=None, names=['name', 'label'])

        if phase == 'train':
            self.data = df[df['name'].str.startswith('train')]  # 如果是训练阶段，选择以'train'开头的数据
        else:
            self.data = df[df['name'].str.startswith('test')]  # 如果是测试阶段，选择以'test'开头的数据

        file_names = self.data.loc[:, 'name'].values  # 获取图像文件名
        # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral
        self.label = self.data.loc[:, 'label'].values  # 获取标签值

        _, self.sample_counts = np.unique(self.label, return_counts=True)  # 统计不同标签的样本数量

        self.file_paths = []  # 存储图像文件的路径
        for f in file_names:
            f = f.split(".")[0]
            f = f + "_aligned.jpg"
            path = os.path.join(self.raf_path, 'Image/aligned', f)
            self.file_paths.append(path)

    def __len__(self):
        return len(self.file_paths)  # 返回数据集的长度

    def __getitem__(self, idx):
        path = self.file_paths[idx]  # 获取指定索引的图像文件路径
        image = cv2.imread(path)  # 读取图像
        label = self.label[idx]  # 获取图像对应的标签
        image = Image.fromarray(image[:, :, ::-1])  # 转换图像的通道顺序
        if self.transform is not None:
            image = self.transform(image)  # 对图像应用变换操作
        return image, label  # 返回处理后的图像和标签
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
                        handlers=[logging.FileHandler('./logs/' + log_name + '.log', 'a', 'utf-8')])
    logging.getLogger('numexpr').setLevel(logging.WARNING)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = True

    model=MFNSB(num_class=7, num_head=args.num_head)
    model.to(device)

    data_transforms = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.RandomRotation(5),
            transforms.RandomCrop(112, padding=8)
        ], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(scale=(0.02, 0.25)),
    ])
    # expected_classes = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Angry']
    #expected_classes=['1neutral', '2happy', '3sad', '4suprise', '5fear', '6disgust', '7angry']
    expected_classes=['Surprise', 'Fear', 'Disgust','Happy', 'Sad', 'Angy', 'Neutral']
    train_dataset = datasets.ImageFolder(f'{args.raf_path}/train', transform=data_transforms)
    # 检查数据集类顺序
    # print("类顺序:", train_dataset.classes)
    print('Whole train set size:', train_dataset.__len__())

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               shuffle=True,
                                               pin_memory=True)

    data_transforms_val = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])

    val_dataset = datasets.ImageFolder(f'{args.raf_path}/test', transform=data_transforms_val)
    #val_dataset = RafDataSet(args.raf_path, phase='test', transform=data_transforms)
    print('Validation set size:', val_dataset.__len__())
    # 检查数据集类顺序
    print("类顺序:", val_dataset.classes)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             num_workers=args.workers,
                                             shuffle=False,
                                             pin_memory=True)

    criterion_cls = torch.nn.CrossEntropyLoss()

    criterion_af = Cosine-Harmonyloss()
    params = list(model.parameters())
    optimizer = torch.optim.SGD(params, lr=args.lr, weight_decay=1e-4, momentum=0.9)
   # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # 定义余弦退火调度器
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=4e-3)
    best_acc = 0
    # if os.path.exists(args.checkpoint_path):
    #     model, optimizer, start_epoch, is_loaded = load_pretrained(model, optimizer, args.checkpoint_path)
    #     if is_loaded:
    #         tqdm.write(f"Pretrained model loaded. Starting from epoch {start_epoch}")
    #     else:
    #         tqdm.write("Checkpoint file is corrupted. Starting training from scratch.")
    #         start_epoch = 1
    # else:
    #     tqdm.write("No pretrained model found. Starting training from scratch.")
    #     start_epoch = 1
    # flood 操作
    def flood(loss, b):
        return (loss - b).abs() + b

    # 定义阈值
    loss_threshold = 0.52
    flood_level = 0.5
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

            loss = criterion_cls(out, targets) + criterion_af(feat,targets)
            # 进行 flooding 操作
            if loss.item() < loss_threshold:
                loss = flood(loss, flood_level)
            loss.backward()
            optimizer.step()
            running_loss += loss
            _, predicts = torch.max(out, 1)
            correct_num = torch.eq(predicts, targets).sum()
            correct_sum += correct_num

        acc = correct_sum.float() / float(train_dataset.__len__())
        running_loss = running_loss / iter_cnt
        logging.info("[Epoch %d] Validation accuracy: %.4f. Loss: %.3f" % (
            epoch, acc, running_loss))
        logging.info("Best_acc:" + str(best_acc))

        tqdm.write('[Epoch %d] Training accuracy: %.4f. Loss: %.3f. LR %.6f' % (
        epoch, acc, running_loss, optimizer.param_groups[0]['lr']))

        with torch.no_grad():
            running_loss = 0.0
            iter_cnt = 0
            bingo_cnt = 0
            sample_cnt = 0
            ## for calculating balanced accuracy
            y_true = []
            y_pred = []

            model.eval()
            for (imgs, targets) in val_loader:
                imgs = imgs.to(device)
                targets = targets.to(device)

                out, feat, heads = model(imgs)
                loss = criterion_cls(out, targets) + criterion_af(feat, targets)
                running_loss += loss

                _, predicts = torch.max(out, 1)
                correct_num = torch.eq(predicts, targets)
                bingo_cnt += correct_num.sum().cpu()
                sample_cnt += out.size(0)

                y_true.append(targets.cpu().numpy())
                y_pred.append(predicts.cpu().numpy())

                if iter_cnt == 0:
                    all_predicted = predicts
                    all_targets = targets
                else:
                    all_predicted = torch.cat((all_predicted, predicts), 0)
                    all_targets = torch.cat((all_targets, targets), 0)
                iter_cnt += 1
            running_loss = running_loss / iter_cnt
            scheduler.step()

            acc = bingo_cnt.float() / float(sample_cnt)
            acc = np.around(acc.numpy(), 4)
            best_acc = max(acc, best_acc)

            y_true = np.concatenate(y_true)
            y_pred = np.concatenate(y_pred)
            balanced_acc = np.around(balanced_accuracy_score(y_true, y_pred), 4)

            tqdm.write(
                "[Epoch %d] Validation accuracy:%.4f. bacc:%.4f. Loss:%.3f" % (epoch, acc, balanced_acc, running_loss))
            tqdm.write("best_acc:" + str(best_acc))
            torch.cuda.empty_cache()
            if acc > 0.91 and acc == best_acc:
                torch.save({'iter': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(), },
                           os.path.join('checkpoints', "rafdb_epoch" + str(epoch) + "_acc" + str(acc) + "_bacc" + str(
                               balanced_acc) + ".pth"))
                tqdm.write('Model saved.')

                # Compute confusion matrix
                matrix = confusion_matrix(all_targets.data.cpu().numpy(), all_predicted.cpu().numpy())
                np.set_printoptions(precision=2)
                plt.figure(figsize=(10, 8))
                # 计算并打印每个类别的准确率
                #class_accuracies = compute_class_accuracies(matrix)
                # Plot normalized confusion matrix
                plot_confusion_matrix(matrix, classes=expected_classes, normalize=True,
                                      title='RAF-DB Confusion Matrix (acc: %0.2f%%)' % (acc * 100))

                plt.savefig(os.path.join('checkpoints', "rafdb_epoch" + str(epoch) + "_acc" + str(acc) + "_bacc" + str(
                    balanced_acc) + ".png"))
                plt.close()


if __name__ == "__main__":
    run_training()
