import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import scipy.io as sio
import matplotlib
matplotlib.use('agg')

from scipy.sparse.linalg import svds
from sklearn import cluster
from sklearn.preprocessing import normalize
from munkres import Munkres
import os, random
import yaml
import transform
from PIL import Image
from sklearn.metrics.cluster import normalized_mutual_info_score


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.use_deterministic_algorithms(True)

class MLP(nn.Module):
    def __init__(self, convae, feature_dim):
        super(MLP, self).__init__()
        self.convencorder = convae

        self.feature_dim = feature_dim
        self.input_dim = 48
        self.input_dim1 = 1280
        self.input_dim2 = 192
        self.Projection_head = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim),
            nn.ReLU(),
            nn.Linear(self.input_dim, self.feature_dim),
        )
        self.low1_Projection_head = nn.Sequential(
            nn.Linear(self.input_dim1, self.input_dim1),
            nn.ReLU(),
            nn.Linear(self.input_dim1, self.feature_dim),
            # nn.ReLU(),
        )
        self.low2_Projection_head = nn.Sequential(
            nn.Linear(self.input_dim2, self.input_dim2),
            nn.ReLU(),
            nn.Linear(self.input_dim2, self.feature_dim),
            # nn.ReLU(),
        )

    def forward(self, x_i, x_j):
        z1, h_1, c1, z3, z5 = self.convencorder(x_i)

        z2, h_2, c2, z4, z6 = self.convencorder(x_j)
        z31 = z3.view(z3.size(0), -1)
        z41 = z4.view(z4.size(0), -1)
        z51 = z5.view(z5.size(0), -1)
        z61 = z6.view(z6.size(0), -1)
        C1 = torch.matmul(c1, c1.t())
        C2 = torch.matmul(c2, c2.t())

        h_i = h_1.view(h_1.size(0), -1)
        h_j = h_2.view(h_2.size(0), -1)
        z_i = F.normalize(self.Projection_head(h_i), dim=1)
        z_j = F.normalize(self.Projection_head(h_j), dim=1)

        z3_i = F.normalize(self.low2_Projection_head(z31), dim=1)
        z4_j = F.normalize(self.low2_Projection_head(z41), dim=1)

        z5_i = F.normalize(self.low1_Projection_head(z51), dim=1)
        z6_j = F.normalize(self.low1_Projection_head(z61), dim=1)
        z3c = torch.matmul(C1, z3_i)
        z4c = torch.matmul(C2, z4_j)
        z5c = torch.matmul(C1, z5_i)
        z6c = torch.matmul(C2, z6_j)
        z3c = z3c.view(z3_i.size())
        z4c = z4c.view(z4_j.size())
        z5c = z5c.view(z5_i.size())
        z6c = z6c.view(z6_j.size())
        return z_i, z_j, h_1, h_2, z1, z2, c1, c2, C1, C2, z3c, z4c, z5c, z6c, z3_i, z4_j, z5_i, z6_j
class InstanceLoss(nn.Module):
    def __init__(self, batch_size, temperature, device):
        super(InstanceLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))

        mask = mask.fill_diagonal_(0)

        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0

        mask = mask.bool()
        return mask

    def forward(self, z_i, z_j):
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)
        sim = torch.matmul(z, z.T) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size)

        sim_j_i = torch.diag(sim, -self.batch_size)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)

        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)

        loss = self.criterion(logits, labels)

        loss /= N
        # print("totalloss",loss)
        return loss
class StructuralContrastiveLoss(nn.Module):
    def __init__(self, batch_size, temperature, device):
        super(StructuralContrastiveLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def forward(self, z_i, z_j, Q):
        N = 2 * self.batch_size
        loss = 0.0
        # z_i = F.normalize(z_i, dim=1)    
        # z_j = F.normalize(z_j, dim=1)
        for i in range(self.batch_size):
            # Compute similarities for all pairs
            sim_i_i = torch.exp(torch.matmul(z_i[i], z_i.T) / self.temperature)
            sim_i_j = torch.exp(torch.matmul(z_i[i], z_j.T) / self.temperature)
            sim_j_i = torch.exp(torch.matmul(z_j[i], z_i.T) / self.temperature)
            sim_j_j = torch.exp(torch.matmul(z_j[i], z_j.T) / self.temperature)

            # Sum of all exponential similarity scores for normalization
            norm_factor1 = sim_i_i.sum() + sim_i_j.sum()
            norm_factor2 = sim_j_i.sum() + sim_j_j.sum()

            # Create a mask for positive samples where Q[i, j] == 1
            pos_mask = Q[i] == 1

            # Calculate the loss for positive samples
            loss_pos = -torch.log(sim_i_i[pos_mask] / norm_factor1) - torch.log(
                sim_i_j[pos_mask] / norm_factor1) - torch.log(sim_j_i[pos_mask] / norm_factor2) - torch.log(
                sim_j_j[pos_mask] / norm_factor2)

            # Sum the loss for all positive pairs
            loss += loss_pos.sum()

        loss /= N
        return loss
class ConvAE(nn.Module):
    def __init__(self, params):
        super(ConvAE, self).__init__()
        kernelSize = params["kernelSize"]
        numHidden = params["numHidden"]
        cte = params["cte"]
        numSubj = params["numSubj"]
        rankEs = params["rankE"]
        self.batchSize = numSubj * params["numPerSubj"]

        self.padEncL1 = nn.ZeroPad2d((1, 1, 1, 1))
        self.encL1 = nn.Conv2d(1, numHidden[0], kernel_size=kernelSize[0], stride=2)

        self.padEncL2 = nn.ZeroPad2d((1, 1, 2, 1))
        self.encL2 = nn.Conv2d(numHidden[0], numHidden[1], kernel_size=kernelSize[1], stride=2)
        self.padEncL2p = nn.ZeroPad2d((0, 0, -1, 0))

        self.padEncL3 = nn.ZeroPad2d((1, 1, 2, 1))
        self.encL3 = nn.Conv2d(numHidden[1], numHidden[2], kernel_size=kernelSize[2], stride=2)
        self.padEncL3p = nn.ZeroPad2d((0, 0, -1, 0))
        cc = np.zeros((self.batchSize, rankEs))
        self.C1 = nn.Parameter(Variable(torch.Tensor(cc), requires_grad=True))

    def forward(self, X):
        Z1 = F.relu(self.encL1(self.padEncL1(X)))
        Z2 = F.relu(self.padEncL2p(self.encL2(self.padEncL2(Z1))))
        Z3 = F.relu(self.padEncL3p(self.encL3(self.padEncL3(Z2))))

        Y = (torch.matmul(self.C1, torch.transpose(self.C1, 0, 1)) - torch.diag(
            torch.diag(torch.matmul(self.C1, torch.transpose(self.C1, 0, 1))))).mm(Z3.view(self.batchSize, -1))
        Y = Y.view(Z3.size())
        return Z3, Y, self.C1, Z2, Z1

def subspaceClusteringMLRDSC(images, params):
    numSubjects = params["numSubj"]
    numPerSubj = params["numPerSubj"]
    alpha = params["alpha"]
    lr = params["lr"]
    cte = params["cte"]
    seedValue = params["seedValue"]
    batchSize = numSubjects * numPerSubj
    regparams = params["regparams"]
    label = params["label"]
    rankEs = params["rankE"]
    gamma_reg, lambda_reg = params["lambda"], params["gamma"]

    x_i = images[:, 0:1].squeeze(2)
    x_j = images[:, 1:2].squeeze(2)
    coilSubjects = x_i
    coilSubjects = coilSubjects.astype(float)
    labelSubjects = label
    labelSubjects = labelSubjects - labelSubjects.min() + 1
    labelSubjects = np.squeeze(labelSubjects)
    X_i = Variable(torch.Tensor(coilSubjects).cuda(), requires_grad=False)

    coilSubjects = x_j
    coilSubjects = coilSubjects.astype(float)
    labelSubjects = label
    labelSubjects = labelSubjects - labelSubjects.min() + 1
    labelSubjects = np.squeeze(labelSubjects)
    X_j = Variable(torch.Tensor(coilSubjects).cuda(), requires_grad=False)
    if params["seedFlag"]:
        random.seed(seedValue)
        os.environ['PYTHONHASHSEED'] = str(seedValue)
        np.random.seed(seedValue)
        torch.manual_seed(seedValue)
        torch.cuda.manual_seed(seedValue)
        torch.cuda.manual_seed_all(seedValue)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    convencorder = ConvAE(params)
    model = MLP(convencorder, 64).cuda()

    save_dir = "pretrain-contrastive-ORL"
    load_path = os.path.join(save_dir, f"PreTrained-contrastive-ORL.pt")
    if os.path.exists(load_path):
        model_dict = torch.load(load_path)
        model.load_state_dict(model_dict)
        print(f"Model loaded from {load_path}")
    else:
        print(f"Checkpoint not found: {load_path}")

    c1_key = 'convencorder.C1'
    c1_tensor = model_dict[c1_key].cpu().numpy()

    model.convencorder.C1.data = (torch.Tensor(cte * np.random.randn(batchSize, rankEs))).cuda()
    model.convencorder.C1.data = torch.Tensor(c1_tensor).cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    temperature1 = 0.1
    temperature2 = 0.1
    loss_device = 'cuda'
    criterion_instance = InstanceLoss(batchSize, temperature1, loss_device).cuda()
    criterion_structral = StructuralContrastiveLoss(batchSize, temperature2, device).cuda()

    T = 4
    numEpochs = 152
    loss_epoch = 0
    best_accuClus = 0
    best_nmi = 0
    best_epoch = 0
    Theta = np.load('pretrain-contrastive-ORL/Theta.npy')
    for epoch in range(numEpochs + 1):

        z_i, z_j, y_i, y_j, z1, z2, C1, C2, c1, c2, z3c, z4c, z5c, z6c, z3_i, z4_j, z5_i, z6_j = model(X_i, X_j)

        Z3 = z3c.view(z3c.size(0), -1)
        Z4 = z4c.view(z4c.size(0), -1)
        Z5 = z5c.view(z5c.size(0), -1)
        Z6 = z6c.view(z6c.size(0), -1)

        loss_instance = criterion_instance(z_i, z_j)

        low_loss_instance1 = criterion_instance(Z3, Z4)
        low_loss_instance2 = criterion_instance(Z5, Z6)
        low_loss_instance = low_loss_instance1 + low_loss_instance2

        loss_structral = criterion_structral(z_i, z_j, Theta)
        z = torch.cat((z1, z2), dim=0)
        y = torch.cat((y_i, y_j), dim=0)
        z1 = torch.cat((z3_i, z4_j), dim=0)
        y1 = torch.cat((z3c, z4c), dim=0)
        z2 = torch.cat((z5_i, z6_j), dim=0)
        y2 = torch.cat((z5c, z6c), dim=0)

        expLoss = lambda_reg * (torch.norm(z - y, p=2) ** 2)
        expLoss1 = lambda_reg * (torch.norm(z1 - y1, p=2) ** 2)
        expLoss2 = lambda_reg * (torch.norm(z2 - y2, p=2) ** 2)
        expLossTotal = expLoss + expLoss1 + expLoss2

        regLoss = 1 * (torch.norm(C1, p=2) ** 2)
        tradeoff = 10000
        tra = 1e-1
        regparams = 1e-3

        loss = tradeoff * (10 * loss_instance + 10 * low_loss_instance + 1 * loss_structral) + 1 * (regparams * expLossTotal + tra * regLoss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_epoch += loss.item()


        if epoch > 0 and epoch % T == 0:
            print("Losses  " + "instanse Contrasitive: %.4f  %.4f %.4f   Structral Contrasitive: %.4f  Expression: %.4f %.4f %.4f Regularization: %.4f" % (
                loss_instance, low_loss_instance1, low_loss_instance2, loss_structral, expLoss, expLoss1, expLoss2, regLoss))
            mm1 = C1.detach().cpu().numpy()
            C = np.dot(mm1, mm1.T)
            params = {"post_proc": [3, 1]}
            Coef = thrC(C, alpha)

            yHat, LC = post_proC(Coef, labelSubjects.max(), params)

            errorClus = err_rate(labelSubjects, yHat)
            accuClus = 1 - errorClus
            nmi = normalized_mutual_info_score(labelSubjects, yHat)
            print("Accuracy after %d" % (epoch), "Iterations: %.4f" % accuClus, "NMI: %.4f" % nmi)

            s2_label_subjs = np.array(yHat)
            s2_label_subjs = s2_label_subjs - s2_label_subjs.min() + 1
            s2_label_subjs = np.squeeze(s2_label_subjs)
            Q = form_structure_matrix(s2_label_subjs, 40)
            Theta = form_Theta(Q)

            if accuClus >= best_accuClus and nmi >= best_nmi:
                best_accuClus = accuClus
                best_nmi = nmi
                best_epoch = epoch
    print("post", params, "best:epoch, acc,nmi:", best_epoch, best_accuClus, best_nmi)

    return (1 - accuClus)

def best_map(L1, L2):
    # L1 should be the labels and L2 should be the clustering number we got
    Label1 = np.unique(L1)
    nClass1 = len(Label1)
    Label2 = np.unique(L2)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1, nClass2)
    G = np.zeros((nClass, nClass))
    for i in range(nClass1):
        ind_cla1 = L1 == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = L2 == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i, j] = np.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:, 1]
    newL2 = np.zeros(L2.shape)
    for i in range(nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]
    return newL2

def thrC(C, ro):
    if ro < 1:
        N = C.shape[1]
        Cp = np.zeros((N, N))
        S = np.abs(np.sort(-np.abs(C), axis=0))
        Ind = np.argsort(-np.abs(C), axis=0)
        for i in range(N):
            cL1 = np.sum(S[:, i]).astype(float)
            stop = False
            csum = 0
            t = 0
            while (stop == False):
                csum = csum + S[t, i]
                if csum > ro * cL1:
                    stop = True
                    Cp[Ind[0:t + 1, i], i] = C[Ind[0:t + 1, i], i]
                t = t + 1
    else:
        Cp = C
    return Cp

def post_proC(C, K, params):
    # C: coefficient matrix, K: number of clusters, d: dimension of each subspace
    d = params["post_proc"][0]
    alpha = params["post_proc"][1]

    C = 0.5 * (C + C.T)
    C = C - np.diag(np.diag(C)) + np.eye(C.shape[0], C.shape[0])
    # r = d * K + 1
    r = min(d * K + 1, C.shape[0] - 1)
    U, S, _ = svds(C, r, v0=np.ones(C.shape[0]))
    U = U[:, ::-1]
    S = np.sqrt(S[::-1])
    S = np.diag(S)
    U = U.dot(S)
    U = normalize(U, norm='l2', axis=1)
    Z = U.dot(U.T)
    Z = Z * (Z > 0)
    L = np.abs(Z ** alpha)
    L = L / L.max()
    L = 0.5 * (L + L.T)
    spectral = cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',
                                          assign_labels='discretize')
    spectral.fit(L)
    grp = spectral.fit_predict(L) + 1
    return grp, L

def err_rate(gt_s, s):
    c_x = best_map(gt_s, s)
    err_x = np.sum(gt_s[:] != c_x[:])
    missrate = err_x.astype(float) / (gt_s.shape[0])
    return missrate

def form_structure_matrix(idx, K):
    Q = np.zeros((len(idx), K))
    for i, j in enumerate(idx):
        Q[i, j - 1] = 1
    return Q

def form_Theta(Q):
    Theta = np.zeros((Q.shape[0], Q.shape[0]))
    for i in range(Q.shape[0]):
        Qq = np.tile(Q[i], [Q.shape[0], 1])
        #         print("Qi",Q[i],"Qq",Qq)
        Theta[i, :] = 1 - 1 / 2 * np.sum(np.square(Q - Qq), 1)
    np.fill_diagonal(Theta, 0)
    return Theta

if __name__ == "__main__":
    args = yaml.load(open("ORL_config.yaml", 'r'), Loader=yaml.FullLoader)
    params = {}
    params["numSubj"] = args["dataset"]["numSubj"]
    params["numPerSubj"] = args["dataset"]["numPerSubj"]
    params["lr"] = args["training"]["lr"]
    params["lambda"] = args["training"]["lambda"]
    params["gamma"] = args["training"]["gamma"]
    params["cte"] = args["training"]["cte"]
    params["seedFlag"] = args["training"]["seedFlag"]
    params["seedValue"] = args["training"]["seedValue"]
    params["post_proc"] = args["training"]["post_proc"]
    params["dataPath"] = args["dataset"]["dataPath"]
    params["kernelSize"] = args["model"]["kernelSize"]
    params["numHidden"] = args["model"]["numHidden"]
    params["input_size"] = args["model"]["input_size"]
    params["rankE"] = args["training"]["rankE"] * params["numSubj"]
    params["indx"] = 0
    params["alpha"] = 0.25
    params["regparams"] = 1.0
    data = sio.loadmat(params["dataPath"])

    images = data['fea']
    images = np.reshape(images, [images.shape[0], 1, params["input_size"][0], params["input_size"][1]])
    params["label"] = data['gnd']

    transforms = transform.Transforms(size=(32, 32), s=0.5, blur=True, noise=False)
    augmented_images = []
    for image in images:
        pil_image = Image.fromarray(image[0])
        augmented_image = transforms(pil_image)
        augmented_images.append(augmented_image)
    augmented_images = np.array(augmented_images)

    errorMean = subspaceClusteringMLRDSC(augmented_images, params)

    print("====================================================")
    print('%d subjects:' % (params["numSubj"]), "Params: %d" % (params["indx"]))
    print('Mean: %.4f%%' % (errorMean * 100))