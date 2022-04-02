from __future__ import print_function
import argparse
import os
import random
import time

import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
from torch.autograd import Variable

import classifier2
import crUtil
import util
import model2
import numpy as np
import torch.nn.functional as F


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='CUB', help='CUB')
parser.add_argument('--k', type=int, default=1, help='k for knn')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate to train GANs ')
parser.add_argument('--syn_num', type=int, default=400, help='number features to generate per class')
parser.add_argument('--nz', type=int, default=312, help='size of the latent z vector')
parser.add_argument('--cr', type=float, default=1.0, help='the weight for the constrative loss')
parser.add_argument('--batch_size', type=int, default=512, help='input batch size')
parser.add_argument('--center_weight', type=float, default=1, help='the weight for the center loss')
parser.add_argument('--ngh', type=int, default=4096, help='size of the hidden units in generator')
parser.add_argument('--nepoch', type=int, default=501, help='number of epochs to train for')
parser.add_argument('--dataroot', default='../datasets', help='path to dataset')
parser.add_argument('--matdataset', default=True, help='Data in matlab format')
parser.add_argument('--image_embedding', default='res101')
parser.add_argument('--class_embedding', default='att')
parser.add_argument('--gzsl', action='store_true', default=True, help='enable generalized zero-shot learning')
parser.add_argument('--preprocessing', action='store_true', default=True,
                    help='enbale MinMaxScaler on visual features')
parser.add_argument('--standardization', action='store_true', default=False)
parser.add_argument('--validation', action='store_true', default=False, help='enable cross validation mode')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--resSize', type=int, default=2048, help='size of visual features')
parser.add_argument('--attSize', type=int, default=312, help='size of semantic features')
parser.add_argument('--lambda1', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
parser.add_argument('--cls_weight', type=float, default=1.0, help='weight of the classification loss')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--gpu', default='0', type=str, help='index of GPU to use')
parser.add_argument('--manualSeed', type=int, default=3483, help='manual seed')
parser.add_argument('--nclass_all', type=int, default=200, help='number of all classes')
parser.add_argument('--nclass_seen', type=int, default=150, help='number of seen classes')
parser.add_argument('--lr_dec_ep', type=int, default=12, help='lr decay for every n epoch')
parser.add_argument('--lr_dec_rate', type=float, default=0.95, help='lr decay rate')
parser.add_argument('--mapSize', type=int, default=2048, help='the size of mapping output')
parser.add_argument('--test_epoch', type=int, default=25, help='view result per test_epoch')
parser.add_argument('--use_classify', type=bool, default=False, help='use classify or not')
parser.add_argument('--temp', type=float, default=0.1)
parser.add_argument('--reg', type=float, default=1.0, help="the weight of regression loss")
parser.add_argument('--cls_epoch', type=int, default=80, help="the weight of regression loss")
parser.add_argument('--d', type=float, default=0.0001, help="the distance of attribute and visual feature")
parser.add_argument('--t2', type=float, default=0.1, help="the temperature of new contrastive loss")
parser.add_argument('--lema', type=float, default=1.0)
parser.add_argument('--begin_step', type=int, default=0, help="begin test step")
parser.add_argument('--syn_t', type=int, default=8, help="synthesis number of train")

opt = parser.parse_args()
print(opt)

os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

# cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# load data
data = util.DATA_LOADER(opt)
print("# of training samples: ", data.ntrain)

# initialize generator and discriminator
netG = model2.Generator(opt)
discriminator = model2.D2(opt)
RNet1 = model2.RNet1(opt)
Cls = model2.classifier(opt)
print(netG)
print(discriminator)
print(RNet1)

input_res = torch.FloatTensor(opt.batch_size, opt.resSize)
input_att = torch.FloatTensor(opt.batch_size, opt.attSize)
noise = torch.FloatTensor(opt.batch_size, opt.nz)
input_label = torch.LongTensor(opt.batch_size)

if opt.cuda:
    discriminator.cuda()
    netG.cuda()
    RNet1.cuda()
    Cls.cuda()
    input_res = input_res.cuda()
    noise, input_att = noise.cuda(), input_att.cuda()
    input_label = input_label.cuda()


def sample():
    batch_feature, batch_label, batch_att = data.next_batch(opt.batch_size)
    input_res.copy_(batch_feature)
    input_att.copy_(batch_att)
    input_label.copy_(util.map_label(batch_label, data.seenclasses))


def generate_syn_feature(netG, classes, attribute, num):
    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass * num, opt.resSize)
    syn_label = torch.LongTensor(nclass * num)
    syn_att = torch.FloatTensor(num, opt.attSize)
    syn_noise = torch.FloatTensor(num, opt.nz)
    if opt.cuda:
        syn_att = syn_att.cuda()
        syn_noise = syn_noise.cuda()
    with torch.no_grad():
        for i in range(nclass):
            iclass = classes[i]
            iclass_att = attribute[iclass]
            syn_att.copy_(iclass_att.repeat(num, 1))
            syn_noise = Variable(torch.randn(num, opt.nz)).cuda()
            output = netG(syn_noise, syn_att)
            syn_feature.narrow(0, i * num, num).copy_(output.data.cpu())
            syn_label.narrow(0, i * num, num).fill_(iclass)
    return syn_feature, syn_label


# setup optimizer
optimizerD = optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerR = optim.Adam(RNet1.parameters(), lr=0.0001, betas=(opt.beta1, 0.999))
optimizerC = optim.Adam(Cls.parameters(), lr=0.001, betas=(opt.beta1, 0.999))


def compute_per_class_acc_gzsl(test_label, predicted_label, target_classes):
    acc_per_class = 0
    for i in target_classes:
        idx = (test_label == i)
        if torch.sum(idx) == 0:
            acc_per_class += 0
        else:
            acc_per_class += float(torch.sum(test_label[idx] == predicted_label[idx])) / float(torch.sum(idx))
    acc_per_class /= float(target_classes.size(0))
    return acc_per_class


def calc_gradient_penalty(netD, real_data, fake_data, input_att):
    alpha = torch.rand(opt.batch_size, 1)
    alpha = alpha.expand(real_data.size())
    if opt.cuda:
        alpha = alpha.cuda()
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    if opt.cuda:
        interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)
    disc_interpolates = netD(interpolates)
    ones = torch.ones(disc_interpolates.size())
    if opt.cuda:
        ones = ones.cuda()
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda1
    return gradient_penalty


def KNNPredict(X_train, y_train, X_test, k=5):
    # sim = -1 * euclidean_distances(X_test.cpu().data.numpy(), X_train.cpu().data.numpy())
    # idx_mat = np.argsort(-1 * sim, axis=1)[:, 0: k]
    idx_mat2 = torch.argsort(torch.cdist(X_test, X_train), axis=1)[:, 0:k]
    preds2 = np.array([torch.argmax(torch.bincount(item)) for item in y_train[idx_mat2]])
    # preds2 = np.array([np.argmax(np.bincount(item)) for item in y_train[idx_mat]])
    return preds2


final_result = {
    "acc_unseen": 0,
    "acc_seen": 0,
    "H": 0
}

contras_criterion = crUtil.SupConLoss_clear2(opt.temp)
cent = torch.from_numpy(data.tr_cls_centroid).cuda()
cc=0
start = time.time()
for start_step in range(0, opt.nepoch):
    # start = time.time()
    for p in discriminator.parameters():  # reset requires_grad
        p.requires_grad = True  # they are set to False below in netG update
    for p in Cls.parameters():
        p.requires_grad = True
    # train D
    for iter_d in range(5):
        sample()
        discriminator.zero_grad()
        Cls.zero_grad()

        input_resv = Variable(input_res)
        input_attv = Variable(input_att)

        discTrue = discriminator(input_resv)
        criticD_real = discTrue.mean()

        # fake
        noise.normal_(0, 1)
        noisev = Variable(noise)
        fake = netG(noisev, input_attv)
        discFalse = discriminator(fake.detach())
        criticD_fake = discFalse.mean()


        gradient_penalty = calc_gradient_penalty(discriminator, input_res, fake.data, input_att)
        D_cost = criticD_fake - criticD_real + gradient_penalty

        RC_loss = F.cross_entropy(Cls(input_resv), input_label)
        D_cost +=  RC_loss * opt.cls_weight
        nn.utils.clip_grad_norm(discriminator.parameters(), max_norm=10, norm_type=2)
        D_cost.backward()
        optimizerD.step()
        optimizerC.step()

    # train G
    for p in discriminator.parameters():
        p.requires_grad = False
    for p in Cls.parameters():
        p.requires_grad = False

    netG.zero_grad()

    RNet1.zero_grad()
    input_attv = Variable(input_att)
    noise.normal_(0, 1)
    noisev = Variable(noise)
    fake = netG(noisev, input_attv)
    # a_r = RNet1(input_resv)
    # R_loss = (input_attv - a_g).pow(2).sum().sqrt() + (input_attv - a_r).pow(2).sum().sqrt()
    discTrue = discriminator(fake)
    discFalse = discriminator(input_resv)
    a_g = RNet1(fake)
    a_r = RNet1(input_resv)
    criticG_fake = discTrue.mean()

    R_loss = Variable(torch.Tensor([0.0])).cuda()
    countn = 0
    for i in range(data.ntrain_class):
        sample_idx = (input_label == i).data.nonzero().squeeze()
        if sample_idx.numel() == 0:

            R_loss += 0
        else:
            countn = countn + 1
            ag_cls = a_g[sample_idx, :]
            ar_cls = a_r[sample_idx, :]
            a_cls = input_attv[sample_idx, :]
            R_loss += (ar_cls.mean(dim=0) - a_cls.mean(dim=0)).pow(
                2).sum().sqrt() + (ag_cls.mean(dim=0) - a_cls.mean(dim=0)).pow(
                2).sum().sqrt()

    R_loss = R_loss / countn
    FC_loss = F.cross_entropy(Cls(fake), input_label)

    errG = - criticG_fake + R_loss * opt.reg + FC_loss * opt.cls_weight
    errG.backward()
    optimizerG.step()

    optimizerR.step()

    if (start_step + 1) % opt.lr_dec_ep == 0:
        for param_group in optimizerD.param_groups:
            param_group['lr'] = param_group['lr'] * opt.lr_dec_rate
        for param_group in optimizerG.param_groups:
            param_group['lr'] = param_group['lr'] * opt.lr_dec_rate

    log_text = 'Iter-{}; errD: {:.3f} errG:{:.3f} gp:{:.3f} , R_loss:{:.3f} RC:{:.3f}, FC:{:.3f}  ' \
        .format(start_step, D_cost.item(), errG.item(), gradient_penalty.item(), R_loss.item(), RC_loss.item(), FC_loss.item())
    print(log_text)


    # test
    if start_step != 0 and start_step % opt.test_epoch == 0 and start_step > opt.begin_step:
        netG.eval()

        syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, opt.syn_num)
        syn_feature2, syn_label2 = generate_syn_feature(netG, data.seenclasses, data.attribute, opt.syn_t)
        train_X = torch.cat((data.train_feature, syn_feature, syn_feature2), 0)
        train_Y = torch.cat((data.train_label, syn_label, syn_label2), 0)
        train_z = train_X.cuda()
        test_z_seen = data.test_seen_feature.cuda()


        if opt.use_classify == True:
            cls = classifier2.CLASSIFIER(train_X, train_Y, data, opt.nclass_all, True, 0.001, 0.5,
                                        opt.cls_epoch, opt.syn_num, True)
            acc_seen = cls.acc_seen
            acc_unseen = cls.acc_unseen

        else:


            test_z_unseen = data.test_unseen_feature.cuda()

            pred_Y_s = torch.from_numpy(KNNPredict(train_z, train_Y, test_z_seen, k=opt.k))
            pred_Y_u = torch.from_numpy(KNNPredict(train_z, train_Y, test_z_unseen, k=opt.k))
            acc_seen = compute_per_class_acc_gzsl(data.test_seen_label, pred_Y_s, data.seenclasses)
            acc_unseen = compute_per_class_acc_gzsl(data.test_unseen_label, pred_Y_u, data.unseenclasses)

        if acc_seen + acc_unseen > 0:
            H = 2 * acc_seen * acc_unseen / (acc_seen + acc_unseen)
        else: H = 0
        print('U: %.1f, S: %.1f, H: %.1f' % (acc_unseen * 100, acc_seen * 100, H * 100))

        if final_result["H"] < H:
            final_result["H"] = H
            final_result["acc_seen"] = acc_seen
            final_result["acc_unseen"] = acc_unseen
        netG.train()


print("result:")
print('%.1f, %.1f, %.1f' % (final_result["acc_unseen"] * 100, final_result["acc_seen"] * 100, final_result["H"] * 100))
print("time used:", time.time() - start)
