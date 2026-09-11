import os
from PIL import Image
import json
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import Dataset


def build_transform(img_size=224):
    return transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
    ])


class COCODataset(Dataset):

    def __len__(self) -> int:
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]
        img_file = f'COCO_train2014_{int(ann["image_id"]):012d}.jpg'
        image_path = os.path.join(self.vis_root, img_file)
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        caption = ann["caption"]
        return {
            "image": image,
            "text_input": caption,
            "image_id": self.img_ids[ann["image_id"]],
        }

    def __init__(self, data_root):
        ann_path = os.path.join(data_root, 'annotations/captions_train2014.json')
        self.vis_root=os.path.join(data_root, 'train2014')
        self.annotation = []
        self.annotation.extend(json.load(open(ann_path, "r"))['annotations'])
        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann["image_id"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1
        self.transform = build_transform()


class COCOKarpathyDataset(Dataset):
    """One sample per caption over the Karpathy split of COCO.

    `train` alone is 82,783 images; `train`+`restval` is the 113,287-image split
    that the captioning literature calls "Karpathy train".
    """

    def __init__(self, karpathy_json, image_dirs, splits=("train", "restval"),
                 max_samples=None, img_size=224):
        self.image_dirs = dict(image_dirs)
        splits = set(splits)

        with open(karpathy_json, "r") as f:
            data = json.load(f)["images"]

        self.annotation = []
        self.img_ids = {}
        for img in data:
            if img["split"] not in splits:
                continue
            folder = img["filepath"]
            if folder not in self.image_dirs:
                raise KeyError(f"no directory configured for COCO folder {folder!r}")
            path = os.path.join(self.image_dirs[folder], img["filename"])
            cocoid = img["cocoid"]
            if cocoid not in self.img_ids:
                self.img_ids[cocoid] = len(self.img_ids)
            for sent in img["sentences"]:
                self.annotation.append((path, sent["raw"].strip(), cocoid))

        if max_samples is not None:
            self.annotation = self.annotation[:max_samples]

        self.transform = build_transform(img_size)
        print(f"[COCOKarpathyDataset] splits={sorted(splits)} "
              f"images={len(self.img_ids)} captions={len(self.annotation)}")

    def __len__(self) -> int:
        return len(self.annotation)

    def __getitem__(self, index):
        path, caption, cocoid = self.annotation[index]
        image = Image.open(path).convert("RGB")
        return {
            "image": self.transform(image),
            "text_input": caption,
            "image_id": self.img_ids[cocoid],
        }
