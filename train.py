import os
import torch.utils.data
from torch.nn import DataParallel
from datetime import datetime
from config import BATCH_SIZE, resume, save_dir, LR, WD, T_max, ETA_min, SAVE_FREQ
from core import resnet

from dataset import CUB


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
start_epoch = 1
save_dir = os.path.join(save_dir, datetime.now().strftime('%Y%m%d_%H%M%S'))
if os.path.exists(save_dir):
    raise NameError('model dir exists!')
os.makedirs(save_dir)

# read dataset
trainset = CUB(root='./CUB_200_2011', is_train=True, data_len=None)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=0, drop_last=False)
testset = CUB(root='./CUB_200_2011', is_train=False, data_len=None)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=0, drop_last=False)
# define model
net = resnet.ResNet50(pretrained=False, num_classes=200)
if resume:
    ckpt = torch.load(resume)
    net.load_state_dict(ckpt['net_state_dict'])
    start_epoch = ckpt['epoch'] + 1
creterion = torch.nn.CrossEntropyLoss()

# define optimizers
optimizer = torch.optim.Adam(net.parameters(), LR, weight_decay=WD)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, ETA_min)
net = net.cuda()
net = DataParallel(net)

for epoch in range(start_epoch, T_max):
    scheduler.step()
    # begin training
    print('--' * 50)
    net.train()
    for i, data in enumerate(trainloader):
        img, label = data[0].cuda(), data[1].cuda()
        batch_size = img.size(0)
        optimizer.zero_grad()
        concat_logits = net(img)
        loss = creterion(concat_logits, label)
        print("epoch{} total loss:{}".format(epoch,loss.item()))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    if epoch % SAVE_FREQ == 0:
        train_loss = 0
        train_correct = 0
        total = 0
        net.eval()
        for i, data in enumerate(trainloader):
            with torch.no_grad():
                img, label = data[0].cuda(), data[1].cuda()
                batch_size = img.size(0)
                concat_logits= net(img)
                concat_loss = creterion(concat_logits, label)
                # calculate accuracy
                _, concat_predict = torch.max(concat_logits, 1)
                total += batch_size
                train_correct += torch.sum(concat_predict.data == label.data)
                train_loss += concat_loss.item() * batch_size

        train_acc = float(train_correct) / total
        train_loss = train_loss / total

        print(
            'epoch:{} - train loss: {:.3f} and train acc: {:.3f} total sample: {}'.format(
                epoch,
                train_loss,
                train_acc,
                total))

	# evaluate on test set
        test_loss = 0
        test_correct = 0
        total = 0
        for i, data in enumerate(testloader):
            with torch.no_grad():
                img, label = data[0].cuda(), data[1].cuda()
                batch_size = img.size(0)
                concat_logits = net(img)
                concat_loss = creterion(concat_logits, label)
                # calculate accuracy
                _, concat_predict = torch.max(concat_logits, 1)
                total += batch_size
                test_correct += torch.sum(concat_predict.data == label.data)
                test_loss += concat_loss.item() * batch_size

        test_acc = float(test_correct) / total
        test_loss = test_loss / total
        print(
            'epoch:{} - test loss: {:.3f} and test acc: {:.3f} total sample: {}'.format(
                epoch,
                test_loss,
                test_acc,
                total))

	# save model
        net_state_dict = net.module.state_dict()
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        torch.save({'epoch': epoch,
                    'test_acc': test_acc,
                    'net_state_dict': net_state_dict},
                   os.path.join(save_dir, 'epoch{}testacc{:.3f}.ckpt'.format(epoch, test_acc)))

print('finishing training')
