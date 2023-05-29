import math
import time

import torch
import torch.nn as nn
import transformers

from quant import *


DEBUG = False 

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class GPTQ:

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        # inp = inp.float()
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        # self.H += 2 / self.nsamples * inp.matmul(inp.t())
        self.H += inp.matmul(inp.t())

    def fasterquant(
        self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)

        H = self.H
        del self.H
        # 这三行代码是fasterquant函数中的一部分，它们用于计算和处理死亡权重（即权重为零的权重）。
        # 第一行代码dead = torch.diag(H) == 0计算出H矩阵的对角线元素是否为零。如果对角线元素为零，则表示该列（或行）的权重为零，即死亡权重。
        # 第二行代码H[dead, dead] = 1将H矩阵中死亡权重对应的对角线元素设置为1。这样做是为了避免在后面的计算中出现除以零的情况,以改善矩阵的可逆性
        # 第三行代码W[:, dead] = 0将权重矩阵中死亡权重对应的列设置为零。这样做是为了在后面的计算中忽略死亡权重。
        # 这三行代码是fasterquant函数中处理死亡权重的关键部分，这样可以过滤掉W矩阵与H矩阵非奇异(non-singular)部分无关的元素,从而改善后续矩阵运算的稳定性。
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)
        # 这段代码的作用是计算一个阻尼系数damp，然后将矩阵H的对角线上的元素加上damp。
        # 其中，percdamp是一个标量，表示阻尼系数的比例。
        # torch.mean(torch.diag(H))计算了矩阵H的对角线上的元素的平均值，即H的迹。diag是一个一维张量，包含了从0到self.columns-1的整数，用于选择矩阵H的对角线上的元素。
        # 最后，H[diag, diag] += damp将阻尼系数加到了矩阵H的对角线上的元素上 。
        # 通过在矩阵H的对角线上增加阻尼项来改善矩阵的条件,以方便后续的求逆运算。
        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        # 首先，函数使用torch.linalg.cholesky(H)对H矩阵进行Cholesky分解，将其分解为一个下三角矩阵。
        # 然后，函数使用torch.cholesky_inverse(H)计算H矩阵的逆。这一步是通过使用Cholesky分解的逆来计算H矩阵的逆实现的。
        # 接下来，函数再次使用torch.linalg.cholesky(H, upper=True)对H矩阵进行Cholesky分解，但这次将其分解为一个上三角矩阵。
        # 最后，函数将计算出的H矩阵的逆赋值给Hinv变量，以便在后面的计算中使用。
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H
        
        # 函数首先使用for i1 in range(0, self.columns, blocksize):循环按块处理权重矩阵。每次循环迭代时，它会计算出当前块的起始和结束位置（i1和i2），以及当前块的大小（count）。
        # 然后，函数使用W1 = W[:, i1:i2].clone()获取当前块的权重，并创建几个与当前块大小相同的零矩阵（Q1，Err1和Losses1）。这些矩阵将在后面的计算中用于存储量化后的权重、误差和损失。
        # 接下来，函数使用Hinv1 = Hinv[i1:i2, i1:i2]获取H矩阵逆的当前块对应的部分。这个子矩阵将在后面的计算中用于更新权重矩阵中未被量化的部分。
        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]
            
            # 函数首先使用for i in range(count):循环遍历当前块内的每一列权重。每次循环迭代时，它会获取当前列的权重（w）和H矩阵逆的对应元素（d）。
            # 然后，如果指定了组大小（即groupsize != -1），则函数会在每个组的开头调用find_params函数来重新计算量化参数。
            # 接下来，函数使用quantize函数对当前列的权重进行量化，并将量化后的权重存储在Q1矩阵中。然后，它计算量化误差并将其存储在Losses1矩阵中。
            # 最后，函数使用量化误差和H矩阵逆来更新权重矩阵中未被量化的部分。这样做是为了减少量化误差，并提高量化后模型的准确性。
            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if (i1 + i) % groupsize == 0:
                        self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)], weight=True)

                q = quantize(
                    w.unsqueeze(1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
                ).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1
            
            # 在对当前块内的权重进行量化后，函数首先使用Q[:, i1:i2] = Q1和Losses[:, i1:i2] = Losses1 / 2更新量化后的权重矩阵和损失矩阵。
            # 然后，函数使用W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])更新权重矩阵中未被量化的部分。这样做是为了减少量化误差，并提高量化后模型的准确性。
            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            if DEBUG:
                self.layer.weight.data[:, :i2] = Q[:, :i2]
                self.layer.weight.data[:, i2:] = W[:, i2:]
                print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                print(torch.sum(Losses))

        torch.cuda.synchronize()
        print('time %.2f' % (time.time() - tick))
        print('error', torch.sum(Losses).item())

        if actorder:
            invperm = torch.argsort(perm)
            Q = Q[:, invperm]

        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()
