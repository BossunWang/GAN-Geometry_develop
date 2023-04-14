import torch
import numpy as np
import matplotlib.pylab as plt
import lpips
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from PIL import Image
import argparse
import os
import sys
import cv2

sys.path.append("../../Face_Attribute")

from resnet_from_EAC import EAC_FULL_Model
from core import get_full_hessian, hessian_compute_enc, save_imgrid, show_imgrid, plot_spectra
from core.GAN_utils import DCGAN_wrapper, loadDCGAN, BigGAN_wrapper, loadBigGAN, upconvGAN
from core.hessian_axis_visualize import vis_eigen_action, vis_eigen_explore, vis_distance_curve
from utils.utils import getHead, getModel, KernelPCAHead
from model.attribute_head import KernelPCAHead


class ModelWrapperV4_eval_kpca_hm(torch.nn.Module):
    def __init__(self, model, age_head, gender_head, masked_head, emotion_head, emotion_kpca_head):
        super(ModelWrapperV4_eval_kpca_hm, self).__init__()
        self.model = model
        self.age_head = age_head
        self.gender_head = gender_head
        self.masked_head = masked_head
        self.emotion_head = emotion_head
        self.emotion_kpca_head = emotion_kpca_head

    def forward(self, image):
        output = self.model(image)
        body, feature = output

        age_logit = self.age_head(feature)
        gender_logit = self.gender_head(feature)
        masked_logit = self.masked_head(feature)
        emotion_output = self.emotion_head(output)
        emotion_logit, hm = emotion_output
        hm_size = hm.size(0)
        emotion_kpca = self.emotion_kpca_head(hm.view(hm_size, -1))

        return age_logit, gender_logit, masked_logit, emotion_logit, emotion_kpca


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ImDist = lpips.LPIPS(net="squeeze", ).to(device)

    args.device = device
    feat_dim = args.feat_dim
    age_num_classes = args.age_class
    gender_num_classes = args.gender_class
    masked_num_classes = args.masked_class
    emotion_num_classes = args.emotion_class
    emotion_value_num_classes = args.emotion_value_class

    age_head, gender_head, masked_head, emotion_head \
        = getHead(args, feat_dim, age_num_classes, gender_num_classes, masked_num_classes, emotion_num_classes)
    backbone = getModel(args, feat_dim, age_num_classes, gender_num_classes, masked_num_classes, emotion_num_classes)

    save_folder = "../../Face_Attribute/utils/Analysis_v9/Happy_resample"
    K_fit_rows = np.load(os.path.join(save_folder, "K_fit_rows.npy"))
    K_fit_all = np.load(os.path.join(save_folder, "K_fit_all.npy"))
    X_fit = np.load(os.path.join(save_folder, "X_fit.npy"))
    scaled_alphas = np.load(os.path.join(save_folder, "scaled_alphas.npy"))

    KPCA = KernelPCAHead(X_fit.shape[0], X_fit.shape[1], scaled_alphas.shape[1]
                         , K_fit_rows, K_fit_all, scaled_alphas, X_fit).to("cpu")

    wrapped_model = ModelWrapperV4_eval_kpca_hm(backbone, age_head, gender_head, masked_head, emotion_head,
                                                KPCA).to(args.device)

    checkpoint = torch.load(args.pretrain_model, map_location=args.device)
    wrapped_model.load_state_dict(checkpoint)
    wrapped_model.eval()

    for p in wrapped_model.parameters():
        p.requires_grad = False

    for p in ImDist.parameters():
        p.requires_grad = False

    img_size = 112
    preprocess = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=args.mean, std=args.std),
    ])

    input_image = Image.open(args.data_path).convert('RGB')
    input_image_LR = input_image.resize((img_size // 4, img_size // 4))

    input_tensor = preprocess(input_image)
    input_tensor_LR = preprocess(input_image_LR)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
    input_batch_LR = input_tensor_LR.unsqueeze(0)  # create a mini-batch as expected by the model

    input_batch = input_batch.to(device).requires_grad_(True)
    input_batch_LR = input_batch_LR.to(device).requires_grad_(True)

    feature, _ = wrapped_model.model(input_batch)
    feature_LR, _ = wrapped_model.model(input_batch_LR)

    feature = feature.detach().clone().requires_grad_(False)
    feature_LR = feature_LR.detach().clone().requires_grad_(False)

    perturb_vec = (feature - feature_LR).requires_grad_(True)

    fc_weights = wrapped_model.age_head.weight.data
    step = 5
    age_group_list = []
    for g in range(0, len(fc_weights) - 1, step):
        ag = torch.sum(fc_weights[g: g + step], dim=0, keepdim=True) / step
        age_group_list.append(ag.reshape(-1))

    fc_weights = torch.stack(age_group_list)
    fc_weights = fc_weights.view(1, age_num_classes // step, 2048, 1, 1)
    fc_weights = Variable(fc_weights, requires_grad=False)

    # attention
    feat_perturb = (feature_LR + perturb_vec).unsqueeze(1)  # N * 1 * C * H * W
    hm_perturb = feat_perturb * fc_weights
    hm_perturb = hm_perturb.sum(2).sum(1, keepdim=True)  # N * self.num_labels * H * W
    hm_perturb = hm_perturb - torch.min(hm_perturb)
    hm_perturb_img = hm_perturb / torch.max(hm_perturb)

    feat = feature_LR.unsqueeze(1)  # N * 1 * C * H * W
    hm = feat * fc_weights
    hm = hm.sum(2).sum(1, keepdim=True)  # N * 1 * H * W
    hm = hm - torch.min(hm)
    hm_img = hm / torch.max(hm)

    hm_perturb_show_img = hm_perturb_img.reshape(4, 4, 1).detach().cpu().numpy()
    hm_perturb_show_img = cv2.resize(hm_perturb_show_img, (img_size, img_size))
    hm_show_img = hm_img.reshape(4, 4, 1).detach().cpu().numpy()
    hm_show_img = cv2.resize(hm_show_img, (img_size, img_size))

    cv2.imshow("org img", np.array(input_image)[:, :, ::-1].copy())
    cv2.imshow("hm", hm_perturb_show_img)
    cv2.imshow("LR img", np.array(input_image_LR.resize((img_size, img_size)))[:, :, ::-1].copy())
    cv2.imshow("hm LR", hm_show_img)
    cv2.waitKey(0)

    preprocess = lambda img: F.interpolate(img, (256, 256), mode='bilinear', align_corners=True)
    d_sim = ImDist(preprocess(hm_img), preprocess(hm_perturb_img))
    print("d_sim:", d_sim)
    gradient = torch.autograd.grad(d_sim, perturb_vec, create_graph=True, retain_graph=True)[0]
    print("gradient:", gradient.shape)

    # Compute the Hessian, and it's eigen-decomposition at a randomly sampled vector.
    # To see results faster, we compute top eigen pairs using Lanzcos iteration.
    # Uncomment lines to explore more random samples. (ImageNet 1000 class)
    eva_FI, evc_FI, H_FI = hessian_compute_enc(perturb_vec, gradient, hessian_method="BackwardIter", cutoff=20)

    print("eva_FI:", eva_FI.shape)
    print("evc_FI:", evc_FI.shape)
    print("H_FI:", H_FI.shape)

    # Spectrum of Hessian
    # Let's see how fast these eigenvalues decay along the spectrum.
    plt.figure("Top Hessian Spectrum age head")
    plt.plot(eva_FI[::-1])
    plt.title("Top Hessian Spectrum")
    plt.ylabel("Eigenvalue")
    plt.xlabel("Rank")
    plt.savefig("Top_Hessian_Spectrum_age_head")
    plt.cla()

    plot_spectra([eva_FI], savename="age_head_spectrum_all", titstr="Age Head Top Spectrum", save=True)
    plt.cla()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='')
    parser.add_argument("--margin", type=float, default=-0.05)
    parser.add_argument("--model_name", type=str, default='ResNet50')
    parser.add_argument("--age_head_name", type=str, default='',
                        help=" The folder to save models.")
    parser.add_argument("--gender_head_name", type=str, default='',
                        help=" The folder to save models.")
    parser.add_argument("--masked_head_name", type=str, default='')
    parser.add_argument("--emotion_head_name", type=str, default='')
    parser.add_argument("--emotion_value_head_name", type=str, default='')
    parser.add_argument('--age_class', type=int, default=101,
                        help='The training epochs.')
    parser.add_argument('--gender_class', type=int, default=2,
                        help='The training epochs.')
    parser.add_argument('--masked_class', type=int, default=3)
    parser.add_argument('--emotion_class', type=int, default=7)
    parser.add_argument('--emotion_value_class', type=int, default=5)
    parser.add_argument("--feat_dim", type=int, default=512)
    parser.add_argument("--att_size", type=int, default=7)
    parser.add_argument('--pretrain_model', type=str, default="",
                        help='The path of pretrained model')
    parser.add_argument('--mean', type=float, nargs='+')
    parser.add_argument('--std', type=float, nargs='+')
    parser.add_argument('--return_body', action='store_true', default=False)
    args = parser.parse_args()
    main(args)