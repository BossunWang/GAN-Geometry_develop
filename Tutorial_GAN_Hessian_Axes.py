import torch
import numpy as np
import matplotlib.pylab as plt
import lpips

from core import get_full_hessian, hessian_compute, save_imgrid, show_imgrid, plot_spectra
from core.GAN_utils import DCGAN_wrapper, loadDCGAN, BigGAN_wrapper, loadBigGAN, upconvGAN
from core.hessian_axis_visualize import vis_eigen_action, vis_eigen_explore, vis_distance_curve


def main():
    ImDist = lpips.LPIPS(net="squeeze", )
    BGAN = loadBigGAN()  # Default to be "biggan-deep-256"
    BGAN.cuda().eval()
    BGAN.requires_grad_(False)
    G = BigGAN_wrapper(BGAN)

    # Compute the Hessian and it's eigen-decomposition at a randomly sampled vector.
    # To see results faster, we compute top eigen pairs using Lanzcos iteration.
    # Uncomment lines to explore more random samples. (ImageNet 1000 class)
    feat = G.sample_vector(device="cuda", class_id=145)  # class King Penguin
    print("feat:", feat.size())
    # feat = G.sample_vector(device="cuda", class_id=17) # class Jay
    # feat = G.sample_vector(device="cuda") # sample a random class
    eva_FI, evc_FI, H_FI = hessian_compute(G, feat, ImDist, hessian_method="BackwardIter", cutoff=20)
    # eva_FI, evc_FI, H_FI = hessian_compute(G, feat, ImDist, hessian_method="ForwardIter", cutoff=20)

    # Transforms encoded in the eigen directions
    # Next, visualize the transformations that these eigen directions encodes.
    # We shall see the top eigen vectors
    refvect = feat.cpu().numpy()
    mtg, codes_all, = vis_eigen_explore(refvect, evc_FI, eva_FI, G, eiglist=[1, 2, 4, 8, 16], transpose=False,
                                        maxdist=0.5, scaling=None, rown=7, sphere=False,
                                        save=False, namestr="demo")
    distmat, ticks, fig = vis_distance_curve(refvect, evc_FI, eva_FI, G, ImDist, eiglist=[1, 2, 4, 8, 16],
                                             maxdist=0.5, rown=5, sphere=False, distrown=15, namestr="demo")

    # Spectrum of Hessian
    # Let's see how fast these eigenvalues decay along the spectrum.
    plt.plot(eva_FI[::-1])
    plt.title("Top hessian Spectrum")
    plt.ylabel("Eigenvalue")
    plt.xlabel("Rank")
    plt.show()

    plot_spectra([eva_FI], titstr="BigGAN Top Spectrum", save=False)
    plt.show()
    plt.cla()

    eva_BP, evc_BP, H_BP = hessian_compute(G, feat, ImDist, hessian_method="BP")  # ~ 91 sec
    # Un comment following lines to try other methods and see the speed
    # eva_BI_h, evc_BI_h, H_BI_h = hessian_compute(G, feat, ImDist, hessian_method="BackwardIter", cutoff=None)
    # eva_FI_h, evc_FI_h, H_FI_h = hessian_compute(G, feat, ImDist, hessian_method="ForwardIter", cutoff=None, EPS=1E-4) # ~ 88 sec

    plot_spectra([eva_BP], titstr="BigGAN full", save=False)
    plt.show()
    plt.cla()
    cc = np.corrcoef(H_BP.flatten(), H_FI.flatten())[0, 1]
    print("Correlation similarity of the Full Hessian matrix and the rank 20 approximation: %.4f" % cc)


if __name__ == '__main__':
    main()