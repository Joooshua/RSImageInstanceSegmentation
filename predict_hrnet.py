import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from models.hrnet_v2.HRNet import HRNetV2
from torch.autograd import Variable
import torch
import os
from PIL import Image
import cv2
from collections import OrderedDict
import torch.nn as nn
from tqdm import tqdm

from torchvision import transforms

#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="1, 2"  # specify which GPU(s) to be used

input_path = './test/image_B/'
output_path = './test/results_B3/'
weight_path = './hrnet/v2_1015/epoch_19_acc_0.94614_kappa_0.93693.pth' #80.655 after SWA 20 epoch
matches = [100, 200, 300, 400, 500, 600, 700, 800]

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class TestDataset(Dataset):
    def __init__(self, input_path, transform):
        self.filename = os.listdir(input_path)
        self.transform = transform

    def __getitem__(self, index):
        image_path = os.path.join(input_path, self.filename[index])
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        img_index = os.path.splitext(self.filename[index])[0]
        return self.transform(img), img_index

    def __len__(self):
        return len(self.filename)



def predict():

    test_loader = DataLoader(
        TestDataset(input_path, transform),
        batch_size=32, shuffle=False, num_workers=4, pin_memory=True
    )

    model = HRNetV2(n_class=8)

    state_dict = torch.load(weight_path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model = model.cuda()
    model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))

    model.eval()
    with torch.no_grad():
        for image, img_index in tqdm(test_loader):   # type(img_index): list
            image = image.cuda()
            #outputs = model(image)
            predict_1 = model(image)

            predict_2 = model(torch.flip(image, [-1]))
            predict_2 = torch.flip(predict_2, [-1])

            predict_3 = model(torch.flip(image, [-2]))
            predict_3 = torch.flip(predict_3, [-2])

            predict_4 = model(torch.flip(image, [-1, -2]))
            predict_4 = torch.flip(predict_4, [-1, -2])

            outputs = (predict_1 + predict_2 + predict_3 + predict_4)/4
            outputs = outputs.data.cpu().numpy() # outputs(batch_size, 8,256,256)

            for idx, output in enumerate(outputs):
                output = output.argmax(axis=0)  # (256,256)
                save_img = np.zeros((256, 256), dtype=np.uint16)
                for i in range(256):
                    for j in range(256):
                        save_img[i][j] = matches[int(output[i][j])]
                cv2.imwrite(os.path.join(output_path, img_index[idx] + '.png'), save_img)

if __name__ == '__main__':

    predict()
