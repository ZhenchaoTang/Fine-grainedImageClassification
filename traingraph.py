import os
import torch.utils.data
from datetime import datetime
from torch.optim.lr_scheduler import MultiStepLR
from config import BATCH_SIZE, PROPOSAL_NUM, LR, WD, save_dir
from layer import model, dataset

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
start_epoch = 1
save_dir = os.path.join(save_dir, datetime.now().strftime('%Y%m%d_%H%M%S'))
if os.path.exists(save_dir):
    raise NameError('model dir exists!')
os.makedirs(save_dir)

trainset = dataset.CUB(root='./CUB_200_2011', is_train=True, data_len=None)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=0, drop_last=False)
testset = dataset.CUB(root='./CUB_200_2011', is_train=False, data_len=None)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=0, drop_last=False)

net = model.CNNtoGraph(pretrain=False, freeze=False)

raw_parameters = list(net.net.pretrained_model.parameters())
part_parameters = list(net.net.proposal_net.parameters())
concat_parameters = list(net.net.concat_net.parameters())
partcls_parameters = list(net.net.partcls_net.parameters())

graph_fc_parameters = list(net.fc.parameters())
graph_classify_parameters = list(net.classify.parameters())

raw_optimizer = torch.optim.SGD(raw_parameters, lr=LR, momentum=0.9, weight_decay=WD)
concat_optimizer = torch.optim.SGD(concat_parameters, lr=LR, momentum=0.9, weight_decay=WD)
part_optimizer = torch.optim.SGD(part_parameters, lr=LR, momentum=0.9, weight_decay=WD)
partcls_optimizer = torch.optim.SGD(partcls_parameters, lr=LR, momentum=0.9, weight_decay=WD)
fc_optimizer = torch.optim.SGD(graph_fc_parameters, lr=LR, momentum=0.9, weight_decay=WD)
classify_optimizer = torch.optim.SGD(graph_classify_parameters, lr=LR, momentum=0.9, weight_decay=WD)

schedulers = [MultiStepLR(raw_optimizer, milestones=[60, 100], gamma=0.1),
              MultiStepLR(concat_optimizer, milestones=[60, 100], gamma=0.1),
              MultiStepLR(part_optimizer, milestones=[60, 100], gamma=0.1),
              MultiStepLR(partcls_optimizer, milestones=[60, 100], gamma=0.1),
              MultiStepLR(fc_optimizer, milestones=[60, 100], gamma=0.1),
              MultiStepLR(classify_optimizer, milestones=[60, 100], gamma=0.1)]

creterion = torch.nn.CrossEntropyLoss()

from layer import ReverseCrossEntropy
rce=ReverseCrossEntropy.RCE(num_cls=200,ratio=1)
net = net.cuda()

for epoch in range(start_epoch, 600):
    for scheduler in schedulers:
        scheduler.step()
    print('--' * 50)
    net.train()
    for i, data in enumerate(trainloader):
        img, label = data[0].cuda(), data[1].cuda()
        batch_size=img.size(0)
        raw_optimizer.zero_grad()
        part_optimizer.zero_grad()
        concat_optimizer.zero_grad()
        partcls_optimizer.zero_grad()
        fc_optimizer.zero_grad()
        classify_optimizer.zero_grad()
        gnn_logits, raw_logits, concat_logits, part_logits, top_n_index, top_n_prob, top_n_cdds, part_feats = net(img)
        part_loss = model.list_loss(part_logits.view(batch_size * PROPOSAL_NUM, -1),
                                    label.unsqueeze(1).repeat(1, PROPOSAL_NUM).view(-1)).view(batch_size, PROPOSAL_NUM)
        raw_loss = creterion(raw_logits, label)
        concat_loss = creterion(concat_logits, label)
        rank_loss = model.ranking_loss(top_n_prob, part_loss)
        partcls_loss = creterion(part_logits.view(batch_size * PROPOSAL_NUM, -1),
                                 label.unsqueeze(1).repeat(1, PROPOSAL_NUM).view(-1))
        gnn_loss = creterion(gnn_logits, label)
        total_loss = raw_loss + rank_loss + concat_loss + partcls_loss + gnn_loss
        print("epoch {} and loss:{}".format(epoch, total_loss.item()))
        total_loss.backward()
        raw_optimizer.step()
        part_optimizer.step()
        concat_optimizer.step()
        partcls_optimizer.step()
        fc_optimizer.step()
        classify_optimizer.step()

    if epoch % 1 == 0:
        net.eval()
        test_correct = 0
        test_correct_with_nts = 0
        total = 0
        for i, data in enumerate(testloader):
            with torch.no_grad():
                img, label = data[0].cuda(), data[1].cuda()
                batch_size = img.size(0)
                gnn_logits, raw_logits, concat_logits, part_logits, top_n_index, top_n_prob, top_n_cdds, part_feats = net(img)
                _, concat_predict = torch.max(gnn_logits, 1)
                total += batch_size
                test_correct += torch.sum(concat_predict.data == label.data)
                _, concat_predict_with_nts = torch.max(gnn_logits+concat_logits, 1)
                test_correct_with_nts += torch.sum(concat_predict_with_nts.data == label.data)
        test_acc = float(test_correct) / total
        test_acc_with_nts = float(test_correct_with_nts) / total
        print(
            'epoch:{} - test acc: {:.3f} - test acc with navigator out:{:.3f} - total sample: {}'.format(
                epoch,
                test_acc,
                test_acc_with_nts,
                total))
        net_state_dict = net.state_dict()
        torch.save({'epoch': epoch,
                    'test_acc': test_acc,
                    'net_state_dict': net_state_dict},
                   os.path.join(save_dir, 'epoch{}testacc{:.3f}testacc_navigator{:.3f}.ckpt'.format(epoch, test_acc,
                                                                                              test_acc_with_nts)))
print('finishing training')