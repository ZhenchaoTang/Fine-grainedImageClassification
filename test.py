import os
import torch.utils.data
from config import BATCH_SIZE
from layer import model, dataset
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
testset = dataset.CUB(root=r"./CUB_200_2011", is_train=False, data_len=None)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=0, drop_last=False)

net = model.CNNtoGraph()
test_model="./checkpoints/model.ckpt"
ckpt = torch.load(test_model)
net.load_state_dict(ckpt['net_state_dict'])
net = net.cuda()

net.eval()
test_correct = 0
total = 0

for i, data in enumerate(testloader):
    with torch.no_grad():
        img, label = data[0].cuda(), data[1].cuda()
        batch_size = img.size(0)
        gnn_logits, raw_logits, concat_logits, part_logits, top_n_index, top_n_prob, top_n_cdds, part_feats = net(img)
        _, concat_predict = torch.max(gnn_logits+concat_logits, 1)
        total += batch_size
        test_correct += torch.sum(concat_predict.data == label.data)
test_acc = float(test_correct) / total
print('test set acc: {:.3f} total sample: {}'.format(test_acc, total))
print('finishing testing')