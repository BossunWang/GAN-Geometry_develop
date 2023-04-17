import os
from os.path import join
from tqdm import tqdm
from time import time
import numpy as np
import torch
import matplotlib.pylab as plt
from core import get_full_hessian, hessian_compute, save_imgrid, show_imgrid, plot_spectra, hessian_summary_pipeline
from core.GAN_utils import DCGAN_wrapper, loadDCGAN, BigGAN_wrapper, loadBigGAN, upconvGAN
import lpips
from scipy.spatial.distance import pdist, squareform
import numpy.ma as ma
from scipy.stats import pearsonr
from core.hessian_analysis_tools import plot_layer_consistency_mat, plot_layer_consistency_example, plot_layer_amplif_curves, plot_layer_amplif_consistency, compute_plot_layer_corr_mat


def main():
    ImDist = lpips.LPIPS(net="squeeze", )
    DG = loadDCGAN()
    DG.cuda().eval()
    DG.requires_grad_(False)
    G = DCGAN_wrapper(DG)

    trials = 30
    savedir = join("hessian/DCGAN")
    os.makedirs(savedir, exist_ok=True)

    T0 = time()
    for triali in tqdm(range(trials)):
        feat = G.sample_vector(sampn=1)  # .detach().clone().cuda()
        eva_BP, evc_BP, H_BP = hessian_compute(G, feat, ImDist, hessian_method="BP")
        # eva_FI, evc_FI, H_FI = hessian_compute(G, feat, ImDist, hessian_method="ForwardIter", EPS=1E-4)
        np.savez(join(savedir, "Hess_BP_%d.npz" % triali), eva_BP=eva_BP, evc_BP=evc_BP, H_BP=H_BP,
                 feat=feat.cpu().detach().numpy())
    print("Total time %.2f sec" % (time() - T0))

    sumdir = join("hessian/DCGAN/summary")
    os.makedirs(sumdir, exist_ok=True)
    S = hessian_summary_pipeline(savedir, "DCGAN", sumdir, npzpatt="Hess_BP_(\d*).npz", featkey="feat", evakey='eva_BP',
                                 evckey='evc_BP', )

    plt.show()

    distmat_feat = squareform(pdist(S.feat_col, metric="euclidean"))  # cosine
    distmat_feat += np.diag(np.nan * np.ones((distmat_feat.shape[0])))
    plt.figure(figsize=[9, 7])
    plt.matshow(distmat_feat, fignum=0)
    plt.title("Distance between the reference vectors")
    plt.colorbar()
    plt.show()
    plt.figure(figsize=[9, 7])
    plt.matshow(S.corr_mat_lin, fignum=0)
    plt.title("Log scale Hessian consistency")
    plt.colorbar()
    plt.show()

    # cc = ma.corrcoef(ma.masked_invalid(S.corr_mat_lin.flatten()), ma.masked_invalid(distmat_feat.flatten()))[0,1]
    msk = np.isnan(distmat_feat.flatten())
    cval, pval = pearsonr(S.corr_mat_log.flatten()[~msk], distmat_feat.flatten()[~msk])
    print("Pearson Corr of the distance between the feature tensor and the log Hessian consistency %.3f (P=%.1e)" % (
    cval, pval))
    cval, pval = pearsonr(S.corr_mat_lin.flatten()[~msk], distmat_feat.flatten()[~msk])
    print("Pearson Corr of the distance between the feature tensor and the lin Hessian consistency %.3f (P=%.1e)" % (
    cval, pval))

    # Mechanism behind the Geometry
    L2dist_col = {}  # global dictionary to store the fetched activation

    def L2dist_hook(module, fea_in, fea_out):
        print("hooker on %s" % module.__class__)
        ref_feat = fea_out.detach().clone()
        ref_feat.requires_grad_(False)
        L2dist = torch.pow(fea_out - ref_feat, 2).sum()
        L2dist_col["dist"] = L2dist
        return None

    datadir = "hessian/DCGAN_layers"
    figdir = "hessian/DCGAN_layers/summary"
    os.makedirs(figdir, exist_ok=True)
    torch.cuda.empty_cache()

    feat = torch.randn(1, 120).cuda()
    feat.requires_grad_(True)

    """Initial layer, format_layer"""
    H1 = G.DCGAN.module.formatLayer.register_forward_hook(L2dist_hook)
    img = G.visualize(feat)
    H1.remove()
    T0 = time()
    H00 = get_full_hessian(L2dist_col["dist"], feat)
    eva00, evc00 = np.linalg.eigh(H00)
    print("Spent %.2f sec computing" % (time() - T0))
    np.savez(join(datadir, "eig_format_layer.npz"), H=H00, eva=eva00, evc=evc00)
    plt.plot(np.log10(eva00)[::-1])
    plt.title("format layer Spectra" % ())
    plt.xlim([0, len(evc00)])
    plt.savefig(join(figdir, "spectrum_format.png"))
    plt.show()

    """Following layers in the sequential network"""
    for name, layer in G.DCGAN.module.main.named_children():
        torch.cuda.empty_cache()
        H1 = layer.register_forward_hook(L2dist_hook)
        img = G.visualize(feat)
        H1.remove()
        T0 = time()
        H00 = get_full_hessian(L2dist_col["dist"], feat)
        eva00, evc00 = np.linalg.eigh(H00)
        print("Spent %.2f sec computing" % (time() - T0))
        np.savez(join(datadir, "eig_%s.npz" % name), H=H00, eva=eva00, evc=evc00)
        plt.plot(np.log10(eva00)[::-1])
        plt.title("Layer %s Spectra" % (name,))
        plt.xlim([0, len(evc00)])
        plt.savefig(join(figdir, "spectrum_%s.png" % (name)))
        plt.show()

    # Load up the data into a list
    layernames = ["format_layer"] + \
                 list(name for name, _ in G.DCGAN.module.main.named_children())
    eva_col, evc_col, H_col = [], [], []
    for name in layernames:
        data = np.load(join(datadir, "eig_%s.npz" % name))
        eva_col.append(data["eva"])
        evc_col.append(data["evc"])
        H_col.append(data["H"])

    compute_plot_layer_corr_mat(eva_col, evc_col, H_col, layernames, titstr="DCGAN", savestr="DCGAN", figdir=figdir)

    plot_layer_amplif_consistency(eva_col, evc_col, layernames, layeridx=[0, 1, 2, -1], titstr="DCGAN",
                                  savelabel="DCGAN", figdir=figdir)
    plot_layer_amplif_curves(eva_col, evc_col, H_col, layernames, savestr="DCGAN", figdir=figdir, maxnorm=False)
    plt.show()


if __name__ == '__main__':
    main()