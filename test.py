import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.datasets.coco import CocoDetection
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.models.detection.ssdlite import _prediction_block
import utils
from engine import train_one_epoch, evaluate
from torch.utils.tensorboard import SummaryWriter
import pickle

model = fasterrcnn_resnet50_fpn(pretrained=False, pretrained_backbone=True)

class DetectDataset(CocoDetection):
    def __init__(self, root, annFile, indices_file=None) -> None:
        super().__init__(root, annFile, transforms.ToTensor())
        self.novel_cls = [1, 17, 16, 21, 18, 19, 20, 2, 9, 6, 3, 4, 7, 44, 62, 67, 64, 63, 72]
        self.base_cls = [i for i in range(1,91) if i not in self.novel_cls]
        if indices_file is None:
            self.valid_index = list(range(super().__len__()))
            new_valid_index = []
            for i in range(super().__len__()):
                img, anno = self[i]
                if img != None and anno != None:
                    new_valid_index.append(i)
            self.valid_index = new_valid_index
            with open('valid_indices_coco', 'wb') as f:
                pickle.dump(self.valid_index, f)
        else:
            with open(indices_file,'rb') as f:
                self.valid_index = pickle.load(f)

    def __getitem__(self, index):
        index = self.valid_index[index]
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
        if len(nTarget['labels']) == 0:
            return None, None
        nTarget = {key:torch.tensor(val) for key, val in nTarget.items()}
        return img, nTarget
    def __len__(self):
        return len(self.valid_index)

valset = DetectDataset('val2017', 'annotations/instances_val2017.json')
trainset = DetectDataset('train2017', 'annotations/instances_train2017.json')

val = DataLoader(valset,batch_size = 16, shuffle=True, collate_fn=utils.collate_fn)
train = DataLoader(trainset,batch_size = 16, shuffle=True, collate_fn=utils.collate_fn)

img, anno = next(iter(val))

print(len(img))
print(len(anno))