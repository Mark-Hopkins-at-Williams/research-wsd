import torch
import torch.nn.functional as F
from reed_wsd.util import cudaify
from torch.autograd import Variable
import math

epsilon = 1e-7

class ConfidenceLoss(torch.nn.Module):
    def __init__(self):
        super(ConfidenceLoss, self).__init__()

    def notify(self, epoch):
        pass

def confidence_weighted_loss(confidence_x, confidence_y, nll_x, nll_y):
    confidence_pair = torch.stack([confidence_x, confidence_y], dim=-1)
    softmaxed_pair = F.softmax(confidence_pair, dim=-1)
    nll_pair = torch.stack([nll_x, nll_y], dim=-1)
    losses = torch.sum(nll_pair * softmaxed_pair, dim=-1)
    return losses

class SingleConfidenceLoss(ConfidenceLoss):
    def __init__(self):
        super(SingleConfidenceLoss, self).__init__()

    def __call__(self, output, confidence, gold):
        raise NotImplementedError("This feature has to be implemented in the child class.")

class PairwiseConfidenceLoss(ConfidenceLoss):

    def __call__(self, output_x, output_y, gold_x, gold_y, conf_x, conf_y):
        output_x = F.softmax(output_x.clamp(min=-25, max=25))
        output_y = F.softmax(output_y.clamp(min=-25, max=25))
        nll_x = F.cross_entropy(output_x, gold_x)
        nll_y = F.cross_entropy(output_y, gold_y)
        #gold_probs_x = output_x[list(range(output_x.shape[0])), gold_x]
        #gold_probs_y = output_y[list(range(output_y.shape[0])), gold_y]
        losses = confidence_weighted_loss(conf_x, conf_y, nll_x, nll_y)
        return losses.mean()

class CrossEntropyLoss(SingleConfidenceLoss):
    """
    this interface is for BEM training
    """    
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.CrossEntropyLoss()

    def __call__(self, output, confidence, gold):
        return self.loss(output, gold)

    def __str__(self):
        return "CrossEntropyLoss"

class NLLLoss(SingleConfidenceLoss):
    def __init__(self):
        super().__init__()
        self.loss = torch.nn.NLLLoss()

    def __call__(self, output, confidence, gold):
        return F.cross_entropy(output, gold, reduction='mean')

    def __str__(self):
        return "NLLLoss"

class AbstainingLoss(SingleConfidenceLoss):
    def __init__(self, alpha=0.5, warmup_epochs=3):
        super().__init__()
        self.alpha = 0.0
        self.warmup_epochs = warmup_epochs
        self.target_alpha = alpha
        self.notify(0)

    def notify(self, epoch):
        if epoch >= self.warmup_epochs:
            self.alpha = self.target_alpha

    def __call__(self, output, confidence, gold):
        output = F.softmax(output.clamp(min=-25, max=25), dim=1)
        label_ps = output[list(range(len(output))), gold]
        abstains = output[:,-1]
        losses = label_ps + (self.alpha * abstains)
        losses = torch.clamp(losses, min = 0.000000001)
        return -torch.mean(torch.log(losses))

class ConfidenceLoss4(SingleConfidenceLoss):
    def __init__(self, alpha=0.5, warmup_epochs=5):
        super().__init__()
        self.alpha = 0.0
        self.warmup_epochs = warmup_epochs
        self.target_alpha = alpha
        self.notify(0)

    def notify(self, epoch):
        if epoch >= self.warmup_epochs:
            self.alpha = self.target_alpha

    def __call__(self, output, confidence, gold):
        label_ps = output[list(range(len(output))), gold]
        label_ps_woa = output[:, :-1]
        label_ps_woa = F.normalize(label_ps_woa, p=1, dim=1)
        label_ps_woa = label_ps_woa[list(range(len(label_ps_woa))), gold]
        losses = label_ps_woa * (label_ps + (self.alpha * confidence))
        losses = torch.clamp(losses, min = 0.000000001)
        return -torch.mean(torch.log(losses))

    def __str__(self):
        return "ConfidenceLoss4_p0_" + str(self.p0)


"""
class DACLoss(SingleConfidenceLoss):
    
    def __init__(self, target_alpha, warmup_epochs,
                       total_epochs, alpha_init_factor, mu=0.05):
        self.target_alpha = target_alpha
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.alpha_init_factor = alpha_init_factor
        self.alpha = None
        self.mu = mu
        self.nll = NLLLoss()
        self.avg_beta = 0.

    def notify(self, epoch):
        self.epoch = epoch

    def _beta(self, output, gold):
        abs_prob = torch.clamp(output[:, -1], max = 1. - epsilon)
        label_ps = output[list(range(len(output))), gold]
        true_class_prob = 1 - abs_prob
        normalized_true_prob = label_ps / true_class_prob
        normalized_true_prob = torch.clamp(normalized_true_prob, min = epsilon)
        thresholds = true_class_prob * (- torch.log(normalized_true_prob))
        return thresholds.mean()

    def update_alpha(self, output, gold):
        if self.epoch < self.warmup_epochs:
            beta = self._beta(output, gold).detach()
            if self.epoch == 0:
                self.avg_beta = beta
            else:
                self.avg_beta = (1 - self.mu) * self.avg_beta + self.mu * beta
        if self.epoch == self.warmup_epochs:
            self.alpha = self.avg_beta / self.alpha_init_factor
            self.delta = ((self.target_alpha - self.alpha) /
                         (self.total_epochs - self.epoch))
        if self.epoch > self.warmup_epochs:
            self.alpha += self.delta

    def __call__(self, output, confidence, gold):
        self.update_alpha(output, gold)
        if self.epoch < self.warmup_epochs:
            return self.nll(output, confidence, gold)
        else:
            assert(self.alpha is not None)
            true_class_loss = self._beta(output, gold)
            abs_prob = torch.clamp(output[:, -1], max = 1. - epsilon)
            true_probs_sum = 1 - abs_prob
            true_probs_sum = torch.clamp(true_probs_sum, min = epsilon)
            abs_loss = self.alpha * torch.log(1 / true_probs_sum)
            abs_loss = abs_loss.mean()
            loss = true_class_loss + abs_loss
            return loss

"""


class DACLoss(SingleConfidenceLoss):
    def __init__(self, target_alpha, warmup_epochs, total_epochs,
		 alpha_init_factor=64.):
        super(DACLoss, self).__init__()
	#self.alpha = alpha
        self.learn_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.alpha_final = target_alpha
        self.alpha_init_factor = alpha_init_factor

        self.alpha_var = None

        self.alpha_thresh_ewma = None   #exponentially weighted moving average for alpha_thresh
        self.alpha_thresh = None #instantaneous alpha_thresh
        self.ewma_mu = 0.05 #mu parameter for EWMA; 
        self.curr_alpha_factor  = None #for alpha initiliazation
        self.alpha_inc = None #linear increase factor of alpha during abstention phase
        self.alpha_set_epoch = None
        self.vars = None

    def notify(self, e):
        self.epoch = e
    
    def _nll(self, output, gold):
        label_ps = output[list(range(len(output))), gold]
        label_ps = torch.clamp(label_ps, min = 0.000000001)
        losses = -torch.log(label_ps)
        return losses

    def _h_c(self, output, gold):
        abs_prob = torch.clamp(output[:, -1], max = 1. - epsilon)
        label_ps = output[list(range(len(output))), gold]
        true_class_prob = 1 - abs_prob
        true_class_prob = torch.clamp(true_class_prob, min=0.01)

        nan_tensor = torch.isnan(true_class_prob)
        assert( not (True in nan_tensor) )

        normalized_true_prob = label_ps / true_class_prob
        normalized_true_prob = torch.clamp(normalized_true_prob, min = epsilon)
        thresholds = (- torch.log(normalized_true_prob))
        return thresholds

    def clip(self):
        if self.vars is not None:
            torch.nn.utils.clip_grad_norm(self.vars, max_norm=1)
        

    def __call__(self, input_batch, confidence, target_batch):
        if self.epoch <= self.learn_epochs:
            #pdb.set_trace()
            loss = F.cross_entropy(input_batch, target_batch, reduction='none')
            #return loss.mean()
            h_c = F.cross_entropy(input_batch[:, :-1], target_batch).detach()
            p_out = torch.exp(F.log_softmax(input_batch,dim=1)).detach()
            p_out_abstain = p_out[:,-1].detach()
            #pdb.set_trace()

            #update instantaneous alpha_thresh
            self.alpha_thresh = Variable(((1. - p_out_abstain)*h_c).mean().data)
            #update alpha_thresh_ewma 
            if self.alpha_thresh_ewma is None:
                self.alpha_thresh_ewma = self.alpha_thresh #Variable(((1. - p_out_abstain)*h_c).mean().data)
            else:
                # self.alpha_thresh_ewma = Variable(self.ewma_mu*((1. - p_out_abstain)*h_c).mean().data + \
                # 	(1. - self.ewma_mu)*self.alpha_thresh_ewma.data)
                self.alpha_thresh_ewma = Variable(self.ewma_mu*self.alpha_thresh.data + \
                        (1. - self.ewma_mu)*self.alpha_thresh_ewma.data)
            return loss.mean()

        else:
            #calculate cross entropy only over true classes
            h_c = F.cross_entropy(input_batch[:,0:-1],target_batch,reduce='none')
            #probabilities of abstention  class
            p_out = torch.exp(F.log_softmax(input_batch,dim=1))
            p_out_abstain = torch.min(p_out[:,-1],
                                      cudaify(Variable(torch.tensor([1. - epsilon]))))
            # avoid numerical instability by upper-bounding 
            # p_out_abstain to never be more than  1 - eps since we have to
            # take log(1 - p_out_abstain) later.
            # pdb.set_trace()
            #p_out_abstain = torch.min(p_out_abstain,
            #        Variable(cudaify(torch.Tensor([1. - epsilon]))))

            #update instantaneous alpha_thresh
            self.alpha_thresh = Variable(((1. - p_out_abstain)*h_c).mean().data)

            #if (epoch == 5):
            #	pdb.set_trace()
            try:
            #update alpha_thresh_ewma
                if self.alpha_thresh_ewma is None:
                    self.alpha_thresh_ewma = self.alpha_thresh #Variable(((1. - p_out_abstain)*h_c).mean().data)
                else:
                    self.alpha_thresh_ewma = Variable(self.ewma_mu*self.alpha_thresh.data + \
                            (1. - self.ewma_mu)*self.alpha_thresh_ewma.data)


                if self.alpha_var is None: #hasn't been initialized. do it now
                    #we create a freshVariable here so that the history of alpha_var
                    #computation (which depends on alpha_thresh_ewma) is forgotten. This
                    #makes self.alpha_var a leaf variable, which will not be differentiated.
                    #aggressive initialization of alpha to jump start abstention
                    self.alpha_var = Variable(self.alpha_thresh_ewma.data /self.alpha_init_factor)
                    self.alpha_inc = (self.alpha_final - self.alpha_var.data)/(self.total_epochs - self.epoch)
                    self.alpha_set_epoch = self.epoch

                else:		
                    # we only update alpha every epoch
                    if self.epoch > self.alpha_set_epoch: 
                            self.alpha_var = Variable(self.alpha_var.data + self.alpha_inc)
                            self.alpha_set_epoch = self.epoch

                loss = (1. - p_out_abstain)*h_c - \
                self.alpha_var*torch.log(1. - p_out_abstain)

                self.vars = [h_c, p_out_abstain]
                return loss.mean()
            except RuntimeError as e:
                #pdb.set_trace()
                print(e)
