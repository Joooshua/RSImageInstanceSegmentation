import time
import os
import sync_transforms
from dataloader import train_dataset, val_dataset, train_loader, val_loader
import torch.nn as nn
import torch
from models.hrnet_v2.HRNet import HRNetV2
from libs import average_meter, metric, lr_scheduler
from libs.nn.modules import focal_loss
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm
from prettytable import PrettyTable
import torchvision
from torchvision import transforms
from PIL import Image
from collections import OrderedDict
from tensorboardX import SummaryWriter

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"  # specify which GPU(s) to be used

directory = "./%s/%s/" % ('hrnet', 'v2_1006')
if not os.path.exists(directory):
    os.makedirs(directory)


class_names = ['water',
            'transportation',
            'architecture',
            'cultivated',
            'grassland',
            'forest',
            'soil',
            'others'
            ]
num_classes = len(class_names)

class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class Trainer(object):
    def __init__(self):
        resize_scale_range = [float(scale) for scale in '0.5, 2.0'.split(',')]
        sync_transform = sync_transforms.Compose([
            sync_transforms.RandomScale(256, 256, resize_scale_range),
            sync_transforms.RandomFlip(0.5)
        ])
        self.resore_transform = transforms.Compose([
            DeNormalize([.485, .456, .406], [.229, .224, .225]),
            transforms.ToPILImage()
        ])
        self.visualize = transforms.Compose([transforms.ToTensor()])

        print('class names {}.'.format(class_names))
        print('Number samples {}.'.format(len(train_dataset)))

        print('category number', num_classes)
        self.class_loss_weight = torch.Tensor([0.4821961288394846,
                                               0.8343937983086624,
                                               0.2794605498254758,
                                               0.3085449327549615,
                                               0.6014994708103181,
                                               0.3310404847345453,
                                               1.0,
                                               0.27785329391819286])
        #self.criterion = focal_loss.FocalLoss(class_num=8, alpha=self.class_loss_weight, gamma=2, size_average=True).cuda()

        #self.criterion = nn.CrossEntropyLoss(weight=self.class_loss_weight, reduction='mean', ignore_index=-1).cuda()
        self.criterion = focal_loss.OhemCrossEntropy(weight=self.class_loss_weight,ignore_label=-1, thres=0.7,min_kept=100000).cuda()
        model = HRNetV2(n_class=num_classes)

        #weight_path = './hrnet/v2_0928/epoch_212_acc_0.87825_kappa_0.85707.pth'
        #state_dict = torch.load(weight_path)
        #new_state_dict = OrderedDict()
        #for k, v in state_dict.items():
        #    name = k[7:]
        #    new_state_dict[name] = v
        #model.load_state_dict(new_state_dict)

        print(model)

        model = model.cuda()
        self.model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))

        self.optimizer = torch.optim.Adadelta(model.parameters(),
                                                  lr=0.001,
                                                  weight_decay=0.00005)#base = 0.1 decay = 1e-4

        self.scheduler = lr_scheduler.CosineAnnealingWarmUpRestarts(self.optimizer, T_0=250, T_mult=2, eta_max=0.1, T_up=50)
        self.max_iter = 220 * len(train_loader)
        #self.mixup_transform = sync_transforms.Mixup()


    def training(self, epoch):
        self.model.train()

        train_loss = average_meter.AverageMeter()

        curr_iter = epoch * len(train_loader)
        #lr = 0.1 * (1 - float(curr_iter) / self.max_iter) ** 0.9
        lr = self.scheduler.get_lr()
        conf_mat = np.zeros((num_classes, num_classes)).astype(np.int64)
        tbar = tqdm(train_loader)

        for imgs, target in tbar:
            # assert data[0].size()[2:] == data[1].size()[1:]
            # data = self.mixup_transform(data, epoch)
            imgs = Variable(imgs)
            target = Variable(target)
            imgs = imgs.cuda()
            target = target.cuda()

            self.optimizer.zero_grad()
            outputs = self.model(imgs)
            # torch.max(tensor, dim)
            _, preds = torch.max(outputs, 1)
            preds = preds.data.cpu().numpy().squeeze().astype(np.uint8)

            loss = self.criterion(outputs, target)

            train_loss.update(loss, 64)
            writer.add_scalar('train_loss', train_loss.avg, curr_iter)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            tbar.set_description('epoch {}, training loss {}, with learning rate {}.'.format(epoch, train_loss.avg, lr))
            target = target.data.cpu().numpy().squeeze().astype(np.uint8)
            conf_mat += metric.confusion_matrix(pred=preds.flatten(),
                                                label=target.flatten(),
                                                num_classes=num_classes)


        train_acc, train_acc_per_class, train_acc_cls, train_IoU, train_FWIoU, train_kappa = metric.evaluate(conf_mat)
        writer.add_scalar(tag='train_loss_per_epoch', scalar_value=train_loss.avg, global_step=epoch, walltime=None)
        writer.add_scalar(tag='train_acc', scalar_value=train_acc, global_step=epoch, walltime=None)
        writer.add_scalar(tag='train_kappa', scalar_value=train_kappa, global_step=epoch, walltime=None)
        table = PrettyTable(["index", "class name", "acc", "IoU"])
        for i in range(num_classes):
            table.add_row([i, class_names[i], train_acc_per_class[i], train_IoU[i]])
        print(table)
        print("train_acc:", train_acc)
        print("train_FWIoU:", train_FWIoU)
        print("kappa:", train_kappa)

    def validating(self, epoch):
        self.model.eval()
        conf_mat = np.zeros((num_classes, num_classes)).astype(np.int64)
        tbar = tqdm(val_loader)
        with torch.no_grad():
            for imgs, target in tbar:
                # assert data[0].size()[2:] == data[1].size()[1:]
                imgs = Variable(imgs)
                target = Variable(target)
                imgs = imgs.cuda()
                target = target.cuda()

                self.optimizer.zero_grad()
                outputs = self.model(imgs)
                _, preds = torch.max(outputs, 1)
                preds = preds.data.cpu().numpy().squeeze().astype(np.uint8)
                target = target.data.cpu().numpy().squeeze().astype(np.uint8)
                score = _.data.cpu().numpy()
                conf_mat += metric.confusion_matrix(pred=preds.flatten(),
                                                label=target.flatten(),
                                                num_classes=num_classes)
        print(conf_mat)
        val_acc, val_acc_per_class, val_acc_cls, val_IoU, val_FWIoU, val_kappa = metric.evaluate(conf_mat)
        writer.add_scalars(main_tag='val_single_acc',
                           tag_scalar_dict={class_names[i]: val_acc_per_class[i] for i in range(len(class_names))},
                           global_step=epoch, walltime=None)
        writer.add_scalars(main_tag='val_single_iou',
                           tag_scalar_dict={class_names[i]: val_IoU[i] for i in range(len(class_names))},
                           global_step=epoch, walltime=None)
        writer.add_scalar('val_acc', val_acc, epoch)
        writer.add_scalar('val_acc_cls', val_acc_cls, epoch)
        writer.add_scalar('val_FWIoU', val_FWIoU, epoch)
        writer.add_scalar('val_kappa', val_kappa, epoch)
        model_name = 'epoch_%d_acc_%.5f_kappa_%.5f' % (epoch, val_acc, val_kappa)
        if val_kappa > 0:
            torch.save(self.model.state_dict(), os.path.join(directory, model_name+'.pth'))
            best_kappa = val_kappa
        table = PrettyTable(["index", "class name", "acc", "IoU"])
        for i in range(num_classes):
            table.add_row([i, class_names[i], val_acc_per_class[i], val_IoU[i]])
        print(table)
        print("val_acc:", val_acc)
        print("val_FWIoU:", val_FWIoU)
        print("kappa:", val_kappa)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    if not os.path.exists(directory + '/log'):
        os.makedirs(directory + '/log')
    writer = SummaryWriter(directory + '/log')
    trainer = Trainer()

    for epoch in range(0, 220):
        trainer.training(epoch)
        trainer.validating(epoch)
