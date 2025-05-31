import torch
from model import *
from NusceneData import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TransfuserNet(device, pred_len=8 )
#checkpoint = torch.load("./model_ckpt/models_2022/transfuser/model_seed3_37.pth")
checkpoint = torch.load("./log/model_29.pth")
pretrained_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
model.load_state_dict(pretrained_dict,strict=False)


dataset = NuScenesData(root_dir="./data/train/", max_len=1100,transform_img=None)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)
train_size = 1000
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])





