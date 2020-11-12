from src.res50_backbone import resnet50
from torch import nn, Tensor
import torch
from torch.jit.annotations import Optional, List, Dict, Tuple, Module
from src.utils import dboxes300_coco, Encoder, PostProcess,calc_iou_tensor,calc_c_tensor,calc_iou_tensor_diag
import torchvision.transforms.functional as func
import numpy as np
import cv2
import torch.nn.functional as F


class SSD640(nn.Module):
    def __init__(self, backbone=None, num_classes=21, pretrain_path=None):
        super(SSD640, self).__init__()
        if backbone is None:
            raise Exception("backbone is None")
        if not hasattr(backbone, "out_channels"):
            raise Exception("the backbone not has attribute: out_channel")

        self.feature_extractor = backbone
        if pretrain_path is not None:
            self.feature_extractor.load_state_dict(torch.load(pretrain_path))

        self.num_classes = num_classes
        # self._build_additional_features([2048, 1024, 512, 256])

        self.num_defaults = [1, 2]
        location_extractors = []
        confidence_extractors = []

        for nd, oc in zip(self.num_defaults, self.feature_extractor.out_channels):
            # nd is number_default_boxes, oc is output_channel
            location_extractors.append(nn.Conv2d(oc, nd * 4, kernel_size=3, padding=1))
            confidence_extractors.append(nn.Conv2d(oc, nd * self.num_classes, kernel_size=3, padding=1))

        self.loc = nn.ModuleList(location_extractors)
        self.conf = nn.ModuleList(confidence_extractors)
        self._init_weights()

        self.default_box = dboxes300_coco()
        self.compute_loss = Loss(self.default_box)
        self.encoder = Encoder(self.default_box)
        self.postprocess = PostProcess(self.default_box)

    def _build_additional_features(self, input_size):
        """
        为backbone(resnet50)添加额外的一系列卷积层，得到相应的一系列特征提取器
        :param input_size:
        :return:
        """
        additional_blocks = []
        # input_size = [2048, 1024, 512, 256] for resnet50
        middle_channels = [512, 256, 128, 64]
        for i, (input_ch, output_ch, middle_ch) in enumerate(zip(input_size[:-1], input_size[1:], middle_channels)):
            layer = nn.Sequential(
                nn.Conv2d(input_ch, middle_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(middle_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(middle_ch, output_ch, kernel_size=3, padding=1, stride=2, bias=False),
                nn.BatchNorm2d(output_ch),
                nn.ReLU(inplace=True),
            )
            additional_blocks.append(layer)
        self.additional_blocks = nn.ModuleList(additional_blocks)

    def _init_weights(self):
        layers = [*self.loc, *self.conf]
        for layer in layers:
            for param in layer.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)

    # Shape the classifier to the view of bboxes
    def bbox_view(self, features, loc_extractor, conf_extractor):
        locs = []
        confs = []
        for f, l, c in zip(features, loc_extractor, conf_extractor):
            # [batch, n*4, feat_size, feat_size] -> [batch, 4, -1]
            locs.append(l(f).view(f.size(0), 4, -1))
            # [batch, n*classes, feat_size, feat_size] -> [batch, classes, -1]
            confs.append(c(f).view(f.size(0), self.num_classes, -1))

        locs, confs = torch.cat(locs, 2).contiguous(), torch.cat(confs, 2).contiguous()
        return locs, confs

    def forward(self, image, targets=None):
        # mean = [0.03973, 0.04146, 0.04213]
        # std = [0.00436, 0.00387, 0.00282]
        #
        # image_for_show = []
        # for data in image:
        #     dtype = data.dtype
        #     mean = torch.as_tensor(mean, dtype=dtype, device=data.device)
        #     std = torch.as_tensor(std, dtype=dtype, device=data.device)
        #     if mean.ndim == 1:
        #         mean = mean[:, None, None]
        #     if std.ndim == 1:
        #         std = std[:, None, None]
        #     data.mul_(std).add_(mean)
        #
        #     im = np.asarray(func.to_pil_image(data.cpu()))
        #     im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        #     image_for_show.append(im)
        #     # cv2.imshow("origin_img", im)
        #     # cv2.waitKey(0)

        # Feature Map 38x38x1024, 19x19x512, 10x10x512, 5x5x256, 3x3x256, 1x1x256
        detection_features = torch.jit.annotate(List[Tensor], [])  # [x]
        x = image
        return_layers = {'layer2': '0', 'layer3': '1'}
        for name, layer in self.feature_extractor.named_children():
            x = layer(x)
            if name in return_layers:
                detection_features.append(x)
                del return_layers[name]
            if not return_layers:
                break
        # for layer in self.additional_blocks:
        #     x = layer(x)
        #     detection_features.append(x)

        # Feature Map
        locs, confs = self.bbox_view(detection_features, self.loc, self.conf)

        # For SSD 640x360, shall return nbatch x 10800 x {nlabels, nlocs} results

        if self.training:
            if targets is None:
                raise ValueError("In training mode, targets should be passed")
            # bboxes_out (Tensor 8732 x 4), labels_out (Tensor 8732)
            bboxes_out = targets['boxes']
            bboxes_out = bboxes_out.transpose(1, 2).contiguous()
            # print(bboxes_out.is_contiguous())
            labels_out = targets['labels']
            # print(labels_out.is_contiguous())

            # for idx, ori_img in enumerate(image_for_show):
            #     labels_for_show = torch.nonzero(labels_out[idx] > 0).squeeze(1)
            #     # 这里的boxes_for_show是(x,y,w,h)形式
            #     boxes_for_show = self.default_box.dboxes_ltrb[labels_for_show]
            #     for per_box in boxes_for_show:
            #         pt1 = (per_box[0] * ori_img.shape[1], per_box[1] * ori_img.shape[0])
            #         pt2 = (per_box[2] * ori_img.shape[1], per_box[3] * ori_img.shape[0])
            #         print(pt2[0] - pt1[0], pt2[1] - pt1[1])
            #         cv2.rectangle(ori_img, pt1, pt2, (0, 0, 255), 1)
            #     cv2.imshow("origin_img_with_boxes", ori_img)
            #     cv2.waitKey(0)

            # ploc, plabel, gloc, glabel
            loss = self.compute_loss(locs, confs, bboxes_out, labels_out)
            return {"total_losses": loss}

        # 将预测回归参数叠加到default box上得到最终预测box，并执行非极大值抑制虑除重叠框
        # results = self.encoder.decode_batch(locs, confs)
        results = self.postprocess(locs, confs)
        return results


class Loss(nn.Module):
    """
        Implements the loss as the sum of the followings:
        1. Confidence Loss: All labels, with hard negative mining
        2. Localization Loss: Only on positive labels
        Suppose input dboxes has the shape 8732x4
    """

    def __init__(self, dboxes):
        super(Loss, self).__init__()
        self.scale_xy = 1.0 / dboxes.scale_xy
        self.scale_wh = 1.0 / dboxes.scale_wh

        self.location_loss = nn.SmoothL1Loss(reduction='none')
        # self.location_loss = nn.SmoothL1Loss(reduce=False)
        self.dboxes = nn.Parameter(dboxes(order="xywh").transpose(0, 1).unsqueeze(dim=0),
                                   requires_grad=False)

        # Two factor are from following links
        # http://jany.st/post/2017-11-05-single-shot-detector-ssd-from-scratch-in-tensorflow.html
        self.confidence_loss = nn.CrossEntropyLoss(reduction='none')
        # self.confidence_loss = nn.CrossEntropyLoss(reduce=False)
        self.alfa = 1
        self.gamma = 2

    def f_Iou(self, Ious, isPositive):
        # fIou =  torch.pow((1 - Ious), self.gamma)
        if isPositive:
            fIou = self.alfa * torch.pow( Ious , self.gamma)
        else:
            fIou = self.alfa * torch.pow((1 - Ious), self.gamma)
        return fIou

    def cross_entropy_Iou(self, input, target, Ious, isPositive):

        tmp = F.log_softmax(input, 1)
        fIou = self.f_Iou(Ious, isPositive)
        fIou = fIou.cuda(device = 'cuda:0')
        tmp1 = tmp * fIou
        return F.nll_loss(tmp1, target, weight = None, size_average=None, ignore_index = -100, reduce = False, reduction = 'mean')

    def _location_vec(self, loc):
        # type: (Tensor)
        """
        Generate Location Vectors
        计算ground truth相对anchors的回归参数
        :param loc:
        :return:
        """
        gxy = self.scale_xy * (loc[:, :2, :] - self.dboxes[:, :2, :]) / self.dboxes[:, 2:, :]
        gwh = self.scale_wh * (loc[:, 2:, :] / self.dboxes[:, 2:, :]).log()

        # gxy = 1 * (loc[:, :2, :] - self.dboxes[:, :2, :]) / self.dboxes[:, 2:, :]
        # gwh = 1 * (loc[:, 2:, :] / self.dboxes[:, 2:, :]).log()
        return torch.cat((gxy, gwh), dim=1).contiguous()

    def _location_vec_inverse(self, ploc):
        # type: (Tensor)
        """
        Generate Location Vectors
        计算ground truth相对anchors的回归参数
        :param loc:
        :return:
        """
        cxy = ploc[:, :2, :] * self.dboxes[:, 2:, :] / self.scale_xy + self.dboxes[:, :2, :]
        cwh = torch.exp(ploc[:, 2:, :] / self.scale_wh) * self.dboxes[:, 2:, :]

        # cxy = ploc[:, :2, :] * self.dboxes[:, 2:, :]  + self.dboxes[:, :2, :]
        # cwh = torch.exp(ploc[:, 2:, :] ) * self.dboxes[:, 2:, :]
        return torch.cat((cxy, cwh), dim=1).contiguous()

    def _xywh2ltrb(self, boxes):
        # For IoU calculation
        # ltrb is left top coordinate and right bottom coordinate
        # 将(x, y, w, h)转换成(xmin, ymin, xmax, ymax)，方便后续计算IoU(匹配正负样本时)
        boxes_ltrb = boxes.clone()
        boxes_ltrb[:, 0, :] = boxes[:, 0, :] - 0.5 * boxes[:, 2, :]
        boxes_ltrb[:, 1, :] = boxes[:, 1, :] - 0.5 * boxes[:, 3, :]
        boxes_ltrb[:, 2, :] = boxes[:, 0, :] + 0.5 * boxes[:, 2, :]
        boxes_ltrb[:, 3, :] = boxes[:, 1, :] + 0.5 * boxes[:, 3, :]
        return boxes_ltrb

    def forward(self, ploc, plabel, gloc, glabel):
        # type: (Tensor, Tensor, Tensor, Tensor)
        """
            ploc, plabel: Nx4x8732, Nxlabel_numx8732
                predicted location and labels

            gloc, glabel: Nx4x8732, Nx8732
                ground truth location and labels
        """

        dbox_ious_max=torch.zeros(8,5440).cuda('cuda:0')
        dbox_ious_New = torch.zeros(8, 5440).cuda('cuda:0')
        c_ofmaxIOU=torch.zeros(8,5440).cuda('cuda:0')
        # 获取正样本的mask  Tensor: [N, 8732]
        mask = glabel > 0
        # mask1 = torch.nonzero(glabel)

        ploc_cxcy = self._location_vec_inverse(ploc)
        ploc_cxcy_ltwb = self._xywh2ltrb(ploc_cxcy)
        gloc_ltrb = self._xywh2ltrb(gloc)
        # gloc_ltrb_tmp = gloc_ltrb.permute(0,2,1)
        # gboxes_ltrb = gloc_ltrb_tmp[mask]

        for kk in range(8):
            maskTmp = glabel[kk, :] > 0
            gboxes_ltrb = gloc_ltrb[kk, :, maskTmp]
            gboxes_ltrb = gboxes_ltrb.transpose(0, 1)
            pboxes_oneImg = ploc_cxcy_ltwb[kk, :, :]
            pboxes_oneImg = pboxes_oneImg.transpose(0, 1)

            ious_pbox_gt = calc_iou_tensor(gboxes_ltrb, pboxes_oneImg)  # [nboxes, 8732]
            best_truth_ious, best_truth_idx = ious_pbox_gt.max(dim=0)  # 寻找每个default box匹配到的最大IoU bboxes_in
            # best_dbox_ious, best_dbox_idx = ious_pbox_gt.max(dim=1)
            # matches = gboxes_ltrb[best_truth_idx]
            # c_pbox_gt = calc_c_tensor(matches, pboxes_oneImg)

            # c_ofmaxIOU[kk,:]=c_pbox_gt
            dbox_ious_max[kk, :] = best_truth_ious


            # modification for iou loss
            tmpxx = gloc_ltrb[kk,:,:]
            ious_pbox_gt_New = calc_iou_tensor_diag(tmpxx.permute(1,0), pboxes_oneImg)  # [nboxes, 8732]
            # diag_ious_pbox_gt = ious_pbox_gt_New.diagonal()
            # pos_ious_pbox = diag_ious_pbox_gt[maskTmp]
            dbox_ious_New[kk, :] = ious_pbox_gt_New

        # iou_loss = 1 - dbox_ious_max
        iou_loss = 1 - dbox_ious_New
        # iou_loss = torch.sqrt(iou_loss)
        # 计算一个batch中的每张图片的正样本个数 Tensor: [N]
        pos_num = mask.sum(dim=1)

        # 计算gt的location回归参数 Tensor: [N, 4, 8732]
        vec_gd = self._location_vec(gloc)

        pboxes=self._location_vec_inverse(ploc)

        # add loss ratio
        with torch.no_grad():
            tmp = 2 * self.dboxes[:, 2:, :] / (pboxes[:, 2:, :] + gloc[:, 2:, :])/self.scale_xy
        # vec_gd1 = vec_gd * 1
        # ploc1 = ploc * 1

        vec_gd[:, :2, :] = vec_gd[:, :2, :] * tmp
        ploc[:, :2, :] = ploc[:, :2, :] * tmp

        # if tmp.max() > 2:
        #     print('haha')
        # sum on four coordinates, and mask
        # 计算定位损失(只有正样本)
        loc_loss = 2* (iou_loss+self.location_loss(ploc[:, :2, :], vec_gd[:, :2, :]).sum(dim=1))  # Tensor: [N, 8732]
        loc_loss = (mask.float() * loc_loss).sum(dim=1)  # Tenosr: [N]

        # loc_loss1 = self.location_loss(ploc1, vec_gd1).sum(dim=1)  # Tensor: [N, 8732]
        # loc_loss1 = (mask.float() * loc_loss1).sum(dim=1)  # Tenosr: [N]

        # hard negative mining Tenosr: [N, 8732]
        # con1 = self.confidence_loss(plabel, glabel)
        con = self.cross_entropy_Iou(plabel, glabel, dbox_ious_New.unsqueeze(dim=1), isPositive=True)

        # positive mask will never selected
        # 获取负样本
        # con_neg1 = con.clone()
        con_neg = self.cross_entropy_Iou(plabel, glabel, dbox_ious_max.unsqueeze(dim=1), isPositive=False)
        con_neg[mask] = torch.tensor(0.0)
        # 按照confidence_loss降序排列 con_idx(Tensor: [N, 8732])
        _, con_idx = con_neg.sort(dim=1, descending=True)
        _, con_rank = con_idx.sort(dim=1)  # 这个步骤比较巧妙

        # number of negative three times positive
        # 用于损失计算的负样本数是正样本的3倍（在原论文Hard negative mining部分），
        # 但不能超过总样本数8732
        neg_num = torch.clamp(3 * pos_num, max=mask.size(1)).unsqueeze(-1)
        neg_mask = con_rank < neg_num  # Tensor [N, 8732]

        # confidence最终loss使用选取的正样本loss+选取的负样本loss
        con_loss = (con * mask.float() + con_neg * neg_mask.float()).sum(dim=1)  # Tensor [N]

        # avoid no object detected
        # 避免出现图像中没有GTBOX的情况
        total_loss = loc_loss + con_loss
        num_mask = (pos_num > 0).float()  # 统计一个batch中的每张图像中是否存在GTBOX
        pos_num = pos_num.float().clamp(min=1e-6)  # 防止出现分母为零的情况
        ret = (total_loss * num_mask / pos_num).mean(dim=0)  # 只计算存在GTBOX的图像损失
        return ret
