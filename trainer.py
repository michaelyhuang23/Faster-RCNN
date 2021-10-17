import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.datasets.coco import CocoDetection
from torchvision import transforms
from torch.utils.data import DataLoader
import utils
from engine import train_one_epoch, evaluate
from torch.utils.tensorboard import SummaryWriter

model = fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=True)

class DetectDataset(CocoDetection):
    def __init__(self, root, annFile) -> None:
        super().__init__(root, annFile, transforms.ToTensor())
        self.novel_cls = [1, 17, 16, 21, 18, 19, 20, 2, 9, 6, 3, 4, 7, 44, 62, 67, 64, 63, 72]
        self.base_cls = [i for i in range(1,91) if i not in self.novel_cls]
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        if target is None or len(target)==0:
            return None, None
        nTarget = {'boxes':[], 'labels':[], 'image_id':target[0]['image_id']}
        for t in target:
            box = t['bbox'].copy()
            box[2] += box[0]
            box[3] += box[1]
            if box[2]-box[0] < 1 or box[3]-box[1] < 1:
                continue
            cate = t['category_id']
            if cate in self.novel_cls:
                continue
            nTarget['boxes'].append(box)
            nTarget['labels'].append(cate)
        nTarget = {key:torch.tensor(val) for key, val in nTarget.items()}
        return img, nTarget
    def __len__(self):
        return super().__len__()

valset = DetectDataset('val2017', 'annotations/instances_val2017.json')
trainset = DetectDataset('train2017', 'annotations/instances_train2017.json')

val = DataLoader(valset,batch_size = 1, shuffle=True, collate_fn=utils.collate_fn)
train = DataLoader(trainset,batch_size = 1, shuffle=True, collate_fn=utils.collate_fn)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'running on {device}')
device_id = torch.cuda.current_device()
print(f'using gpu {torch.cuda.get_device_name(device_id)}')
model.to(device)
#model.load_state_dict(torch.load('fasterrcnn_val1.weights'))
writer = SummaryWriter()
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params,lr=0.0001)
#optimizer = torch.optim.SGD(params, lr=0.0002, momentum=0.9, weight_decay=0.0001)
model.load_state_dict(torch.load('fasterrcnn_train3.weights'))
num_epochs = 10
for epoch in range(num_epochs):
    train_one_epoch(writer, model, optimizer, train, device, epoch, print_freq=10000)
    evaluate(writer, model, val, device, print_freq=1000)
    torch.save(model.state_dict(), f'fasterrcnn_train4.weights')

