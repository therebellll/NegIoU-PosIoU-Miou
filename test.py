from src.ssd_model import SSD300
from src.res50_backbone import resnet50
import torch
import transform
from my_dataset import NightDataSet
import os
import train_utils.train_eval_utils as utils
from train_utils.coco_utils import get_coco_api_from_dataset


def create_model(num_classes=21, device=torch.device('cpu')):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    pre_train_path = "./src/resnet50.pth"
    backbone = resnet50()
    model = SSD300(backbone=backbone, num_classes=num_classes, pretrain_path=pre_train_path)

    # load train weights
    train_weights = "./save_weights/model.pth"
    train_weights_dict = torch.load(train_weights, map_location=device)['model']

    model.load_state_dict(train_weights_dict, strict=False)
    model.to(device)
    # # https://ngc.nvidia.com/catalog/models -> search ssd -> download FP32
    # pre_ssd_path = "./src/nvidia_ssdpyt_fp32.pt"
    # pre_model_dict = torch.load(pre_ssd_path, map_location=device)
    # pre_weights_dict = pre_model_dict["model"]
    #
    # del_conf_loc_dict = {}
    # for k, v in pre_weights_dict.items():
    #     split_key = k.split(".")
    #     if "conf" in split_key:
    #         continue
    #     del_conf_loc_dict.update({k: v})
    #
    # missing_keys, unexpected_keys = model.load_state_dict(del_conf_loc_dict, strict=False)
    # if len(missing_keys) != 0 or len(unexpected_keys) != 0:
    #     print("missing_keys: ", missing_keys)
    #     print("unexpected_keys: ", unexpected_keys)

    return model


def main(parser_data):
    device = torch.device(parser_data.device if torch.cuda.is_available() else "cpu")
    print(device)

    if not os.path.exists("save_weights"):
        os.mkdir("save_weights")

    data_transform = {
        "test": transform.Compose([transform.Resize(),
                                   transform.ToTensor(),
                                   transform.Normalization()])
    }

    night_root = parser_data.data_path
    test_dataset = NightDataSet(night_root, data_transform['test'], train_set='test.txt')
    test_data_loader = torch.utils.data.DataLoader(test_dataset,
                                                   batch_size=4,
                                                   shuffle=False,
                                                   num_workers=0,
                                                   collate_fn=utils.collate_fn)

    model = create_model(num_classes=3, device=device)
    print(model)
    model.to(device)

    # define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=5,
                                                   gamma=0.5)

    # 如果指定了上次训练保存的权重文件地址，则接着上次结果接着训练
    if parser_data.resume != "":
        checkpoint = torch.load(parser_data.resume)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        parser_data.start_epoch = checkpoint['epoch'] + 1
        print("the training process from epoch{}...".format(parser_data.start_epoch))

    test_val_map = []

    val_data = None

    for epoch in range(parser_data.start_epoch, parser_data.epochs):
        utils.evaluate(model=model, data_loader=test_data_loader,
                       device=device, data_set=val_data, mAP_list=test_val_map)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)

    # 训练设备类型
    parser.add_argument('--device', default='cuda:1', help='device')
    # 训练数据集的根目录
    parser.add_argument('--data-path', default='./', help='dataset')
    # 文件保存地址
    parser.add_argument('--output-dir', default='./save_weights', help='path where to save')
    # 若需要接着上次训练，则指定上次训练保存权重文件地址
    parser.add_argument('--resume', default='', type=str, help='resume from checkpoint')
    # 指定接着从哪个epoch数开始训练
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    # 训练的总epoch数
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')

    args = parser.parse_args()
    print(args)

    # 检查保存权重文件夹是否存在，不存在则创建
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args)
