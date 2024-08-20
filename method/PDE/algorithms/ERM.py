from algorithms.single_model_algorithm import SingleModelAlgorithm
from models.initializer import initialize_model
from utils import move_to
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import torch
class ERM(SingleModelAlgorithm):
    def __init__(self, config, d_out, grouper, loss,
            metric, n_train_steps):
        model = initialize_model(config, d_out)
        # initialize module
        super().__init__(
            config=config,
            model=model,
            grouper=grouper,
            loss=loss,
            metric=metric,
            n_train_steps=n_train_steps,
        )
        self.use_unlabeled_y = config.use_unlabeled_y # Expect x,y,m from unlabeled loaders and train on the unlabeled y

    def process_batch(self, batch, unlabeled_batch=None):
        """
        Overrides single_model_algorithm.process_batch().
        ERM defines its own process_batch to handle if self.use_unlabeled_y is true.
        Args:
            - batch (tuple of Tensors): a batch of data yielded by data loaders
            - unlabeled_batch (tuple of Tensors or None): a batch of data yielded by unlabeled data loader
        Output:
            - results (dictionary): information about the batch
                - y_true (Tensor): ground truth labels for batch
                - g (Tensor): groups for batch
                - metadata (Tensor): metadata for batch
                - y_pred (Tensor): model output for batch 
                - unlabeled_g (Tensor): groups for unlabeled batch
                - unlabeled_metadata (Tensor): metadata for unlabeled batch
                - unlabeled_y_pred (Tensor): predictions for unlabeled batch for fully-supervised ERM experiments
                - unlabeled_y_true (Tensor): true labels for unlabeled batch for fully-supervised ERM experiments
        """
        x, y_true, metadata = batch
        x = move_to(x, self.device)
        y_true = move_to(y_true, self.device)
        g = move_to(self.grouper.metadata_to_group(metadata), self.device)

        outputs = self.get_model_output(x, y_true)

        results = {
            'g': g,
            'y_true': y_true,
            'y_pred': outputs,
            'metadata': metadata,
        }

        return results

    def process_batch_pgd(self, batch, unlabeled_batch=None):
        """
        Overrides single_model_algorithm.process_batch().
        ERM defines its own process_batch to handle if self.use_unlabeled_y is true.
        Args:
            - batch (tuple of Tensors): a batch of data yielded by data loaders
            - unlabeled_batch (tuple of Tensors or None): a batch of data yielded by unlabeled data loader
        Output:
            - results (dictionary): information about the batch
                - y_true (Tensor): ground truth labels for batch
                - g (Tensor): groups for batch
                - metadata (Tensor): metadata for batch
                - y_pred (Tensor): model output for batch
                - unlabeled_g (Tensor): groups for unlabeled batch
                - unlabeled_metadata (Tensor): metadata for unlabeled batch
                - unlabeled_y_pred (Tensor): predictions for unlabeled batch for fully-supervised ERM experiments
                - unlabeled_y_true (Tensor): true labels for unlabeled batch for fully-supervised ERM experiments
        """
        x, y_true, metadata = batch
        x = move_to(x, self.device)
        y_true = move_to(y_true, self.device)
        g = move_to(self.grouper.metadata_to_group(metadata), self.device)
        x_adv=pgd_test(self.model,x,y_true,epsilon=0.00196,random_init=0.0001,device=self.device)

        outputs = self.get_model_output(x_adv, y_true)

        results = {
            'g': g,
            'y_true': y_true,
            'y_pred': outputs,
            'metadata': metadata,
        }

        return results


    def objective(self, results,batch=None):
        if self.is_training:
            x, y_true, metadata = batch
            x = move_to(x, self.device)
            y_true = move_to(y_true, self.device)
            #labeled_loss = self.loss.compute(results['y_pred'], results['y_true'], return_dict=False)
            labeled_loss=trades_train(self.model, x, y_true, self.optimizer, self.device)
        else:
            labeled_loss = self.loss.compute(results['y_pred'], results['y_true'], return_dict=False)
        if self.use_unlabeled_y and 'unlabeled_y_true' in results:
            unlabeled_loss = self.loss.compute(
                results['unlabeled_y_pred'], 
                results['unlabeled_y_true'], 
                return_dict=False
            )
            lab_size = len(results['y_pred'])
            unl_size = len(results['unlabeled_y_pred'])
            return (lab_size * labeled_loss + unl_size * unlabeled_loss) / (lab_size + unl_size)
        else:
            return labeled_loss


def pgd_test(model,x,y,epsilon=0.00196,random_init=0.0001,device=None):
    model.eval()
    step_size = epsilon / 4
    X_pgd = Variable(x.data, requires_grad=True)

    random_noise = torch.FloatTensor(X_pgd.shape).uniform_(-random_init, random_init).to(device)
    X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(10):
        opt = torch.optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - x.data, -epsilon, epsilon)
        X_pgd = Variable(x.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1), requires_grad=True)
    x = X_pgd
    return x
from autoattack import AutoAttack
def AA_test(model,x,y,epsilon=0.00196,random_init=0.0001,device=None):
    model.eval()
    step_size = epsilon / 4

    adversary = AutoAttack(model, norm='Linf', eps=epsilon,
                           version='standard', device=device, verbose=False)
    adversary.attacks_to_run = ['apgd-ce']

    adv_images = adversary.run_standard_evaluation(x, y, bs=x.shape[0])
    adv_images.to(device)
    return adv_images



criterion_kl2 = nn.KLDivLoss(reduction='none')
criterion_kl = nn.KLDivLoss(size_average=False)
loss_trade = nn.CrossEntropyLoss()
def trades_train(model,x,y,optimizer,device):
    x_adv = trades(model, x, y, epsilon=0.00196, step_size=0.00196/ 4,
                        num_steps=10,
                        loss_fn='trades', category='trades', rand_init=True, device=device)


    model.train()

    # inputs=turn_batch2one_inf(inputs,label,0.3,net,20,Loss,optimizer,4)

    # x_adv = Variable(torch.clamp(x_adv, args.clmin, args.clmax), requires_grad=False)
    optimizer.zero_grad()
    outputs = model(x)
    loss_nat = loss_trade(outputs, y)
    loss_robust =(1.0 / x.shape[0]) *criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                                          F.softmax(model(x), dim=1))

    loss=loss_nat+6* loss_robust
    return loss





def trades(model, data, target, epsilon, step_size, num_steps,loss_fn,category,rand_init,device):
    model.eval()
    if category == "trades":
        x_adv = data.detach().clone() + 0.0001 * torch.randn(data.shape).to(device) if rand_init else data.detach()
    if category == "Madry":
        x_adv = data.detach() + torch.from_numpy(np.random.uniform(-epsilon, epsilon, data.shape)).float().to(device) if rand_init else data.detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    for k in range(num_steps):
        x_adv.requires_grad_()
        output = model(x_adv)
        model.zero_grad()
        with torch.enable_grad():
            if loss_fn == "cent":
                loss_adv = nn.CrossEntropyLoss(reduction="mean")(output, target)
            elif loss_fn == 'trades':
                loss_adv = nn.KLDivLoss(size_average=False)(F.log_softmax(output, dim=1),
                                       F.softmax(model(data.detach().clone()), dim=1))

        loss_adv.backward()
        eta = step_size * x_adv.grad.sign()
        x_adv = x_adv.detach() + eta
        x_adv = torch.min(torch.max(x_adv, data - epsilon), data + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    model.zero_grad()
    return x_adv.detach()


