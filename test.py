import os
import torch.utils.data
from torch.nn import DataParallel
from config import BATCH_SIZE,test_model
from core import resnet, resnet_posture, vit, vit_posture

from dataset import CUB

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# read dataset
testset = CUB(root='./CUB_200_2011', is_train=False, data_len=None)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                         shuffle=False, num_workers=0, drop_last=False)
# define model
net = resnet_posture.PMRC(pretrained=False, num_classes=200)
# net = resnet.ResNet50(pretrained=False, num_classes=200)
ckpt = torch.load(test_model)
net.load_state_dict(ckpt)
net = net.cuda()
net = DataParallel(net)
creterion = torch.nn.CrossEntropyLoss()

net.eval()
# evaluate on test set
test_loss = 0
test_correct = 0
total = 0
for i, data in enumerate(testloader):
    with torch.no_grad():
        img, label = data[0].cuda(), data[1].cuda()
        batch_size = img.size(0)
        concat_logits = net(img)
        # calculate loss
        concat_loss = creterion(concat_logits, label)
        # calculate accuracy
        _, concat_predict = torch.max(concat_logits, 1)
        total += batch_size
        test_correct += torch.sum(concat_predict.data == label.data)
        test_loss += concat_loss.item() * batch_size

test_acc = float(test_correct) / total
test_loss = test_loss / total
print('test set loss: {:.3f} and test set acc: {:.3f} total sample: {}'.format(test_loss, test_acc, total))

print('finishing testing')
