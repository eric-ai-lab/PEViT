import os
import torch
import torch.utils.data
from PIL import Image
from torchvision import transforms


class Voc2007Classification(torch.utils.data.Dataset):
    def __init__(self, data_root, image_set="train", transform=None):
        """
        Pascal voc2007 training/validation data: http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
        test data: http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
        """
        self.data_root = self._update_path(data_root, image_set)
        self.transform = transform
        self.labels = self._read_annotation(image_set)
        self.images = list(self.labels.keys())

    @staticmethod
    def _update_path(data_root, image_set):
        if image_set == "train" or image_set == "val":
            data_root += "train/VOCdevkit/VOC2007"
        elif image_set == "test":
            data_root += "test/VOCdevkit 2/VOC2007"
        else:
            raise Exception("Incorrect image set!")
        return data_root

    def __getitem__(self, index):
        img_path = os.path.join(self.data_root, 'JPEGImages/' + self.images[index] + '.jpg')
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
        label = self.labels[self.images[index]]
        label = torch.LongTensor(label)
        return image, label

    def __len__(self):
        return len(self.images)

    def _read_annotation(self, image_set="train"):
        """
        Annotation interpolation, refer to:
        http://host.robots.ox.ac.uk/pascal/VOC/voc2007/htmldoc/voc.html#SECTION00093000000000000000
        """
        object_categories = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
                             'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                             'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
        annotation_folder = os.path.join(self.data_root, "ImageSets/Main/")
        files = [file_name for file_name in os.listdir(annotation_folder) if file_name.endswith("_" + image_set + ".txt")]
        labels_all = dict()
        for file_name in files:
            label_str = file_name.split("_")[0]
            label_int = object_categories.index(label_str)
            with open(annotation_folder + "/" + file_name, "r") as fread:
                for line in fread.readlines():
                    index = line[:6]
                    if index not in labels_all.keys():
                        labels_all[index] = [0] * len(object_categories)
                    flag = 1
                    if line[7:9] and int(line[7:9]) != 1:
                        flag = -1
                    if flag == 1:
                        labels_all[index][label_int] = 1
        return labels_all

