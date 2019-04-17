#%%
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as CM

from mcnn_model import MCNN
from my_dataloader import CrowdDataset


def cal_mae(img_root,gt_dmap_root,model_param_path):
    '''
    Calculate the MAE of the test data.
    img_root: the root of test image data.
    gt_dmap_root: the root of test ground truth density-map data.
    model_param_path: the path of specific mcnn parameters.
    '''
    device=torch.device("cuda")
    mcnn=MCNN().to(device)
    mcnn.load_state_dict(torch.load(model_param_path))
    dataset=CrowdDataset(img_root,gt_dmap_root,4)
    dataloader=torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False)
    mcnn.eval()
    mae=0
    for i,(img,gt_dmap) in enumerate(dataloader):
        img=img.to(device)
        gt_dmap=gt_dmap.to(device)
        # forward propagation
        et_dmap=mcnn(img)
        mae+=abs(et_dmap.data.sum()-gt_dmap.data.sum()).item()
        del img,gt_dmap,et_dmap

    print("model_param_path:"+model_param_path+" mae:"+str(mae/len(dataloader)))

def estimate_density_map(img_root,gt_dmap_root,model_param_path,index):
    '''
    Show one estimated density-map.
    img_root: the root of test image data.
    gt_dmap_root: the root of test ground truth density-map data.
    model_param_path: the path of specific mcnn parameters.
    index: the order of the test image in test dataset.
    '''
    device=torch.device("cuda")
    mcnn=MCNN().to(device)
    mcnn.load_state_dict(torch.load(model_param_path))
    dataset=CrowdDataset(img_root,gt_dmap_root,4)
    dataloader=torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False)
    mcnn.eval()
    for i,(img,gt_dmap) in enumerate(dataloader):
        if i==index:
            img=img.to(device)
            gt_dmap=gt_dmap.to(device)
            # forward propagation
            et_dmap=mcnn(img).detach()
            et_dmap=et_dmap.squeeze(0).squeeze(0).cpu().numpy()
            print(et_dmap.shape)
            plt.imshow(et_dmap,cmap=CM.jet)
            break


if __name__=="__main__":
    torch.backends.cudnn.enabled=False
    img_root='D:\\workspaceMaZhenwei\\MCNN-pytorch\\data\\Shanghai_part_A\\test_data\\images'
    gt_dmap_root='D:\\workspaceMaZhenwei\\MCNN-pytorch\\data\\Shanghai_part_A\\test_data\\ground_truth'
    model_param_path='D:\\workspaceMaZhenwei\\MCNN-pytorch\\checkpoints\\epoch_63.param'
    # cal_mae(img_root,gt_dmap_root,model_param_path)
    # estimate_density_map(img_root,gt_dmap_root,model_param_path,3) 