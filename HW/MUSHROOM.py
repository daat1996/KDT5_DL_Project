import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from rembg import remove



class MUSHROOMModel(nn.Module):

    # 모델 구성요소
    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        self.layer1 = nn.Linear(in_dim, hidden_dim[0])
        # 은닉충을 자유롭게 설정가능하도록
        self.layer_List = nn.ModuleList([])
        if len(hidden_dim) > 1:
            for a in range(len(hidden_dim) - 1):
                self.layer_h = nn.Linear(hidden_dim[a], hidden_dim[a + 1])
                self.layer_List.append(self.layer_h)
        self.layer2 = nn.Linear(hidden_dim[-1], out_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.relu(self.layer1(x))
        for layer in self.layer_List:
            y = self.relu(layer(y))
        y = self.layer2(y)
        return y


def classify_img(model_name='new_mushroom_model.pth', file_name='test4.jpg'):
    model = torch.load(model_name)
    model.eval()

    m_list = ['edible', 'inedible', 'poisonous']
    img = cv2.imread(file_name, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (150, 150))

    plt.imshow(img)

    feature_img = img.reshape(-1)
    norm_img = feature_img / 255.

    input_img = torch.FloatTensor(norm_img)

    print(m_list[model(input_img).argmax()])


model_name = input('사용할 모델을 입력하세요(끝까지): ')
file_name = input('이미지 이름을 입력하세요(끝까지): ')
classify_img(model_name, file_name)