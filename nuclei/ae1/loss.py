import torch as T
from torch.autograd import Variable


def loss_ae(yp_batch, ms_batch):
    loss = Variable(T.zeros(len(yp_batch)).cuda(), requires_grad=False)
    for ix, (yp, ms) in enumerate(zip(yp_batch, ms_batch)):
        n_masks = ms.max()
        re = Variable(T.zeros(n_masks).cuda(), requires_grad=False)
        loss1 = Variable(T.zeros(n_masks).cuda(), requires_grad=False)

        for i in range(n_masks):
            pixel = yp[ms.cuda() == (i + 1)]
            re[i] = T.mean(pixel)
            loss1[i] = T.mean((pixel - re[i]) ** 2)
        A = re.expand(n_masks, n_masks)
        B = T.transpose(A, 0, 1)

        loss1 = T.mean(loss1)
        loss2 = T.mean(T.exp(-1/2 * (A - B) ** 2))
        loss[ix] = loss1 + loss2
    return T.mean(loss)