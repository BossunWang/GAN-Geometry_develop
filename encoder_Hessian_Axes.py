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

from resnet_from_EAC import EAC_FULL_Model
from core import get_full_hessian, hessian_compute_enc, save_imgrid, show_imgrid, plot_spectra
from core.GAN_utils import DCGAN_wrapper, loadDCGAN, BigGAN_wrapper, loadBigGAN, upconvGAN
from core.hessian_axis_visualize import vis_eigen_action, vis_eigen_explore, vis_distance_curve


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ImDist = lpips.LPIPS(net="squeeze", ).to(device)
    model = EAC_FULL_Model(args).to(device)

    for p in model.parameters():
        p.requires_grad = False

    for p in ImDist.parameters():
        p.requires_grad = False

    img_size = 112
    preprocess = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image_path = "/media/glory/46845c74-37f7-48d7-8b72-e63c83fa4f68/face_dataset/wiki_retinafacecrop/00/37500_1944-01-23_2010.jpg"
    input_image = Image.open(image_path)
    input_image_LR = input_image.resize((img_size // 4, img_size // 4))

    input_tensor = preprocess(input_image)
    input_tensor_LR = preprocess(input_image_LR)
    input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
    input_batch_LR = input_tensor_LR.unsqueeze(0)  # create a mini-batch as expected by the model

    input_batch = input_batch.to(device).requires_grad_(True)
    input_batch_LR = input_batch_LR.to(device).requires_grad_(True)

    feature = model(input_batch)
    feature_LR = model(input_batch_LR)

    feature = feature.view(feature.size(0), -1).detach().clone().requires_grad_(False)
    feature_LR = feature_LR.view(feature_LR.size(0), -1).detach().clone().requires_grad_(False)

    perturb_vec = (feature_LR - feature).requires_grad_(True)

    # for p in model.fc.parameters():
    #     p.requires_grad = True

    fc_weights = model.fc.weight.data
    fc_weights = fc_weights.view(1, 7, 2048, 1, 1)
    fc_weights = Variable(fc_weights, requires_grad=False)

    # attention
    feat_perturb = (feature + perturb_vec).unsqueeze(1)  # N * 1 * C * H * W
    hm_perturb = feat_perturb * fc_weights
    hm_perturb = hm_perturb.sum(2).sum(1, keepdim=True)  # N * self.num_labels * H * W

    feat = feature.unsqueeze(1)  # N * 1 * C * H * W
    hm = feat * fc_weights
    hm = hm.sum(2).sum(1, keepdim=True)  # N * 1 * H * W

    preprocess = lambda img: F.interpolate(img, (256, 256), mode='bilinear', align_corners=True)
    d_sim = ImDist(preprocess(hm), preprocess(hm_perturb))
    print("d_sim:", d_sim)
    gradient = torch.autograd.grad(d_sim, perturb_vec, create_graph=True, retain_graph=True)[0]
    print("gradient:", gradient.shape)

    # Compute the Hessian and it's eigen-decomposition at a randomly sampled vector.
    # To see results faster, we compute top eigen pairs using Lanzcos iteration.
    # Uncomment lines to explore more random samples. (ImageNet 1000 class)
    eva_FI, evc_FI, H_FI = hessian_compute_enc(perturb_vec, gradient, hessian_method="BackwardIter", cutoff=20)

    # # Transforms encoded in the eigen directions
    # # Next, visualize the transformations that these eigen directions encodes.
    # # We shall see the top eigen vectors
    # refvect = feat.cpu().numpy()
    # mtg, codes_all, = vis_eigen_explore(refvect, evc_FI, eva_FI, G, eiglist=[1, 2, 4, 8, 16], transpose=False,
    #                                     maxdist=0.5, scaling=None, rown=7, sphere=False,
    #                                     save=False, namestr="demo")
    # distmat, ticks, fig = vis_distance_curve(refvect, evc_FI, eva_FI, G, ImDist, eiglist=[1, 2, 4, 8, 16],
    #                                          maxdist=0.5, rown=5, sphere=False, distrown=15, namestr="demo")
    #
    # # Spectrum of Hessian
    # # Let's see how fast these eigenvalues decay along the spectrum.
    # plt.plot(eva_FI[::-1])
    # plt.title("Top hessian Spectrum")
    # plt.ylabel("Eigenvalue")
    # plt.xlabel("Rank")
    # plt.show()
    #
    # plot_spectra([eva_FI], titstr="BigGAN Top Spectrum", save=False)
    # plt.show()
    # plt.cla()
    #
    # eva_BP, evc_BP, H_BP = hessian_compute(G, feat, ImDist, hessian_method="BP")  # ~ 91 sec
    # # Un comment following lines to try other methods and see the speed
    # # eva_BI_h, evc_BI_h, H_BI_h = hessian_compute(G, feat, ImDist, hessian_method="BackwardIter", cutoff=None)
    # # eva_FI_h, evc_FI_h, H_FI_h = hessian_compute(G, feat, ImDist, hessian_method="ForwardIter", cutoff=None, EPS=1E-4) # ~ 88 sec
    #
    # plot_spectra([eva_BP], titstr="BigGAN full", save=False)
    # plt.show()
    # plt.cla()
    # cc = np.corrcoef(H_BP.flatten(), H_FI.flatten())[0, 1]
    # print("Correlation similarity of the Full Hessian matrix and the rank 20 approximation: %.4f" % cc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--pretrain_model', type=str, default='../../Face_Attribute/pretrained_weights/resnet50_ft_weight.pkl',
                        help='pretrained_backbone_path')
    parser.add_argument('--att_size', type=int, default=4)
    args = parser.parse_args()
    main(args)