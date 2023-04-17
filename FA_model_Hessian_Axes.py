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

sys.path.append("../Face_Attribute")

from resnet_from_EAC import EAC_FULL_Model
from core import get_full_hessian, hessian_compute_enc, save_imgrid, show_imgrid, plot_spectra
from core.GAN_utils import DCGAN_wrapper, loadDCGAN, BigGAN_wrapper, loadBigGAN, upconvGAN
from core.hessian_axis_visualize import vis_eigen_action, vis_eigen_explore, vis_distance_curve
from utils.utils import getHead, getModel


class ModelWrapperV4_eval_hm(torch.nn.Module):
    def __init__(self, model, age_head, gender_head, masked_head, emotion_head, emotion_value_head):
        super(ModelWrapperV4_eval_hm, self).__init__()
        self.model = model
        self.age_head = age_head
        self.gender_head = gender_head
        self.masked_head = masked_head
        self.emotion_head = emotion_head
        self.emotion_value_head = emotion_value_head

    def forward(self, image):
        output = self.model(image)

        if type(output) is tuple:
            body, feature = output
        else:
            feature = output

        age_logit = self.age_head(feature)
        gender_logit = self.gender_head(feature)
        masked_logit = self.masked_head(feature)
        emotion_output = self.emotion_head(output)
        emotion_logit, hm = emotion_output
        hm_size = hm.size(0)
        emotion_value_logit = self.emotion_value_head(hm.view(hm_size, -1))

        return age_logit, gender_logit, masked_logit, emotion_logit, emotion_value_logit, hm


def test_img(img_path, wrapped_model, preprocess, img_size, age_num_classes, ImDist, device, args):
    input_image = Image.open(img_path).convert('RGB')
    input_image_LR = input_image.resize((img_size // 8, img_size // 8))

    input_tensor = preprocess(input_image)
    input_tensor_LR = preprocess(input_image_LR)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
    input_batch_LR = input_tensor_LR.unsqueeze(0)  # create a mini-batch as expected by the model

    input_batch = input_batch.to(device).requires_grad_(True)
    input_batch_LR = input_batch_LR.to(device).requires_grad_(True)

    feature, feature_emb = wrapped_model.model(input_batch)
    feature_LR, feature_LR_emb = wrapped_model.model(input_batch_LR)

    age_logit = wrapped_model.age_head(feature_emb)
    age_lr_logit = wrapped_model.age_head(feature_LR_emb)

    print("age pred:", torch.argmax(age_logit, dim=-1).item())
    print("age LR pred:", torch.argmax(age_lr_logit, dim=-1).item())

    feature_emb = feature_emb.detach().clone().requires_grad_(False)
    feature_LR_emb = feature_LR_emb.detach().clone().requires_grad_(False)

    perturb_vec = (feature_emb - feature_LR_emb).requires_grad_(True)

    fc_weights = wrapped_model.age_head.weight.data

    # attention
    feat_perturb = (feature_emb + perturb_vec).unsqueeze(1)  # N * 1 * C * H * W
    hm_perturb = feat_perturb * fc_weights
    hm_perturb = hm_perturb.sum(1)
    feat = feature_emb.unsqueeze(1)  # N * 1 * C * H * W
    hm = feat * fc_weights
    hm = hm.sum(1)

    print(hm_perturb.shape)
    print(hm.shape)

    cosine = F.linear(F.normalize(hm_perturb), F.normalize(hm))
    print("cosine:", cosine)

    cv2.imshow("org img", np.array(input_image)[:, :, ::-1].copy())
    cv2.imshow("LR img", np.array(input_image_LR.resize((img_size, img_size)))[:, :, ::-1].copy())

    # calculate gradient
    gradient = torch.autograd.grad(cosine, perturb_vec, create_graph=True, retain_graph=True)[0]
    print("gradient:", gradient.shape)

    # Compute the Hessian, and it's eigen-decomposition at a randomly sampled vector.
    # To see results faster, we compute top eigen pairs using Lanzcos iteration.
    # Uncomment lines to explore more random samples. (ImageNet 1000 class)
    eva_FI, evc_FI, H_FI = hessian_compute_enc(perturb_vec, gradient, hessian_method="BackwardIter", cutoff=20)

    print("eva_FI:", eva_FI)
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
    #
    # test top eigen vector
    top_eigen_vec = torch.from_numpy(evc_FI[:, -1].reshape(1, 2048)).to(device)
    top_eigen_val = eva_FI[-1]
    print("top_eigen_val:", top_eigen_val)
    # print("top_eigen_vec:", top_eigen_vec)
    # feature_eig = feature_LR - top_eigen_vec * top_eigen_val
    feature_eigen_emb = feature_LR_emb + top_eigen_vec
    # feature_eig = feature_LR - F.relu(top_eigen_vec) * 10
    age_eigen_logit = wrapped_model.age_head(feature_eigen_emb)
    print("age eigen pred:", torch.argmax(age_eigen_logit, dim=-1).item())

    # feat_eig = feature_eig.unsqueeze(1)  # N * 1 * C * H * W
    # hm_eig = feat_eig * fc_weights
    # hm_eig = hm_eig.sum(2).sum(1, keepdim=True)  # N * 1 * H * W
    # hm_eig = hm_eig - torch.min(hm_eig)
    # hm_eig_img = hm_eig / torch.max(hm_eig)
    #
    # hm_eig_show_img = hm_eig_img.reshape(4, 4, 1).detach().cpu().numpy()
    # hm_eig_show_img = cv2.resize(hm_eig_show_img, (img_size, img_size))
    # cv2.imshow("hm eig", hm_eig_show_img)
    cv2.waitKey(0)


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

    age_head, gender_head, masked_head, emotion_head, emotion_value_head \
        = getHead(args, feat_dim, age_num_classes, gender_num_classes, masked_num_classes
                  , emotion_num_classes, emotion_value_num_classes)
    backbone = getModel(args, feat_dim, age_num_classes, gender_num_classes, masked_num_classes, emotion_num_classes)

    wrapped_model = ModelWrapperV4_eval_hm(backbone, age_head, gender_head, masked_head
                                           , emotion_head, emotion_value_head).to(args.device)

    checkpoint = torch.load(args.pretrain_model, map_location=args.device)
    wrapped_model.load_state_dict(checkpoint['model'])
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

    for root, dirs, files in os.walk(args.data_path):
        for name in files:
            img_path = os.path.join(root, name)
            test_img(img_path, wrapped_model, preprocess, img_size, age_num_classes, ImDist, device, args)


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