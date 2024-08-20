import os, csv
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torchvision

from models import model_attributes,PreActResNet18
from data.data import dataset_attributes, shift_types, prepare_data, log_data
from utils import set_seed, Logger, CSVBatchLogger, log_args
from train import train
from zennit_show import zennit_show
from visual import gradcam


def main():
    parser = argparse.ArgumentParser()

    # Settings
    parser.add_argument('-d', '--dataset', choices=dataset_attributes.keys(), required=True)
    parser.add_argument('-s', '--shift_type', choices=shift_types, required=True)
    # Confounders
    parser.add_argument('-t', '--target_name')
    parser.add_argument('-c', '--confounder_names', nargs='+')
    # Resume
    parser.add_argument('--resume', default=False, action='store_true')  #is resume
    parser.add_argument('--pth_name', default='', type=str)              #model path
    # Label shifts
    parser.add_argument('--minority_fraction', type=float)
    parser.add_argument('--imbalance_ratio', type=float)
    # Data
    parser.add_argument('--fraction', type=float, default=1.0)
    parser.add_argument('--root_dir', default=None)
    parser.add_argument('--reweight_groups', action='store_true', default=False)   #need UW strategy
    parser.add_argument('--augment_data', action='store_true', default=False)      #need aug
    parser.add_argument('--val_fraction', type=float, default=0.1)                 #val fraction
    parser.add_argument('--class_num', default=2, type=int)                        #number of class
    # Objective
    parser.add_argument('--robust', default=False, action='store_true')           #need group-dro framework
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--generalization_adjustment', default="0.0")
    parser.add_argument('--automatic_adjustment', default=False, action='store_true')
    parser.add_argument('--use_normalized_loss', default=False, action='store_true')
    parser.add_argument('--btl', default=False, action='store_true')
    parser.add_argument('--hinge', default=False, action='store_true')

    # Model
    parser.add_argument(
        '--model',
        choices=model_attributes.keys(),
        default='resnet50')
    parser.add_argument('--train_from_scratch', action='store_true', default=False)

    # Optimization
    parser.add_argument('--n_epochs', type=int, default=4)
    parser.add_argument('--n_classes', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--scheduler', action='store_true', default=False)
    parser.add_argument('--weight_decay', type=float, default=5e-5)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--minimum_variational_weight', type=float, default=0)


    #log
    parser.add_argument('--log_dir', default='./logs')
    parser.add_argument('--log_dir_text', default='log.txt')
    parser.add_argument('--name_index', default='unit', type=str)
    parser.add_argument('--log_every', default=2000, type=int)
    parser.add_argument('--save_step', type=int, default=100)
    parser.add_argument('--save_best', action='store_true', default=True)
    parser.add_argument('--save_last', action='store_true', default=True)

    #training
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--show_progress', default=False, action='store_true')
    parser.add_argument('--train_type',  default="erm")   #training type range from[erm,trades,BAT,CFA,WAT,FAT,pgd]
    parser.add_argument('--test_type', default="erm")     #range from[erm,pgd,AA]
    parser.add_argument('--only_test', default=False, action='store_true')   #not training
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--early_stop', default=False, action='store_true')
    parser.add_argument('--is_combine', default=False, action='store_true')  #freeze model


    #adversarial attack
    parser.add_argument('--epsilon', default=0.03,type=float)
    parser.add_argument('--random_init', default=0.01, type=float)
    parser.add_argument('--attack_iters', default=10, type=int)
    parser.add_argument('--clmax', default=1.0, type=float)
    parser.add_argument('--clmin', default=0.0, type=float)


    #visualization
    parser.add_argument('--zennit_show',  default=False, action='store_true')

    #hyperparamters for other mdoels
    parser.add_argument('--beta', default=6.0, type=float)                 #trades
    parser.add_argument('--eta', default=0.1, type=float)
    parser.add_argument('--tau', default=0.03, type=float)
    parser.add_argument('--lr_tau', default=0.004, type=float)
    parser.add_argument('--lamda', default=0.03, type=float)
    parser.add_argument('--alpha_', default=0.03, type=float)
    parser.add_argument('--l2_norm', default=0.0, type=float)

    #GDRO-AT
    parser.add_argument('--limit_nat', default=False, action='store_true')            #nat disitributional constraint
    parser.add_argument('--limit_adv', default=False, action='store_true')            #adv disitributional constraint
    parser.add_argument('--train_grad', default=False, action='store_true')           #WULG
    parser.add_argument('--trades_new', default=False, action='store_true')           #GDRO-AT
    parser.add_argument('--limit_eps', default=0.2, type=float)                       #constraint eps
    parser.add_argument('--robust_step_size', default=0.01, type=float)               #update step size



    args = parser.parse_args()
    check_args(args)

    args.device=torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu")

    # BERT-specific configs copied over from run_glue.py
    if args.model == 'bert':
        args.max_grad_norm = 1.0
        args.adam_epsilon = 1e-8
        args.warmup_steps = 0

    if os.path.exists(args.log_dir) and args.resume:
        resume=True
        mode='a'
    else:
        resume=False
        mode='w'

    ## Initialize logs
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    args=set_output_dir(args)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    logger = Logger(args.log_dir+'/'+args.name_index+str(args.only_test)[0]+args.log_dir_text, mode)

    # Record args
    log_args(args, logger)
    set_seed(args.seed)

    # Data
    # Test data for label_shift_step is not implemented yet
    test_data = None
    test_loader = None
    if args.shift_type == 'confounder':
        train_data, val_data, test_data = prepare_data(args, train=True)
    elif args.shift_type == 'label_shift_step':
        train_data, val_data = prepare_data(args, train=True)
    loader_kwargs = {'batch_size':args.batch_size, 'num_workers':0, 'pin_memory':True}
    train_loader = train_data.get_loader(train=True, reweight_groups=args.reweight_groups, **loader_kwargs)
    val_loader = val_data.get_loader(train=False, reweight_groups=None, **loader_kwargs)
    if test_data is not None:
        test_loader = test_data.get_loader(train=False, reweight_groups=None, **loader_kwargs)
    data = {}
    data['train_loader'] = train_loader
    data['val_loader'] = val_loader
    data['test_loader'] = test_loader
    data['train_data'] = train_data
    data['val_data'] = val_data
    data['test_data'] = test_data
    n_classes = train_data.n_classes
    log_data(data, logger)


    ## Initialize model
    pretrained = not args.train_from_scratch
    if resume:
        model=torch.load(args.pth_name)
    elif model_attributes[args.model]['feature_type'] in ('precomputed', 'raw_flattened'):
        assert pretrained
        # Load precomputed features
        d = train_data.input_size()[0]
        model = nn.Linear(d, n_classes)
        model.has_aux_logits = False
    elif args.model == 'resnet50':
        model = torchvision.models.resnet50(pretrained=pretrained)
        d = model.fc.in_features
        model.fc = nn.Linear(d, n_classes)
    elif args.model == 'resnet34':
        model = torchvision.models.resnet34(pretrained=pretrained)
        d = model.fc.in_features
        model.fc = nn.Linear(d, n_classes)
    elif args.model == 'wideresnet50':
        model = torchvision.models.wide_resnet50_2(pretrained=pretrained)
        d = model.fc.in_features
        model.fc = nn.Linear(d, n_classes)
    elif args.model == 'pre18':
        model = PreActResNet18()
    elif args.model == 'bert':
        assert args.dataset == 'MultiNLI'
        from pytorch_transformers import BertConfig, BertForSequenceClassification
        config_class = BertConfig
        model_class = BertForSequenceClassification
        config = config_class.from_pretrained(
            'bert-base-uncased',
            num_labels=3,
            finetuning_task='mnli')
        model = model_class.from_pretrained(
            'bert-base-uncased',
            from_tf=False,
            config=config)
    else:
        raise ValueError('Model not recognized.')

    #visualization
    if args.zennit_show:
        args.model_name='res'
        args.image_path='zenuni/input_0.png'
        args.cuda=True
        args.eigen_smooth=True
        args.aug_smooth = True
        gradcam.show_visual_model(model,args)

    logger.flush()

    #training
    if not args.zennit_show:
        ## Define the objective
        if args.hinge:
            assert args.dataset in ['CelebA', 'CUB'] # Only supports binary
            def hinge_loss(yhat, y):
                # The torch loss takes in three arguments so we need to split yhat
                # It also expects classes in {+1.0, -1.0} whereas by default we give them in {0, 1}
                # Furthermore, if y = 1 it expects the first input to be higher instead of the second,
                # so we need to swap yhat[:, 0] and yhat[:, 1]...
                torch_loss = torch.nn.MarginRankingLoss(margin=1.0, reduction='none')
                y = (y.float() * 2.0) - 1.0
                return torch_loss(yhat[:, 1], yhat[:, 0], y)
            criterion = hinge_loss
        else:
            criterion = torch.nn.CrossEntropyLoss(reduction='none')

        train_csv_logger = CSVBatchLogger(os.path.join(args.log_dir, args.name_index+'train.csv'), train_data.n_groups, mode=mode)
        val_csv_logger =  CSVBatchLogger(os.path.join(args.log_dir, args.name_index+'val.csv'), train_data.n_groups, mode=mode)
        test_csv_logger =  CSVBatchLogger(os.path.join(args.log_dir, args.name_index+'test.csv'), train_data.n_groups, mode=mode)

        train(model, criterion, data, logger, train_csv_logger, val_csv_logger, test_csv_logger, args, epoch_offset=0)
    
        train_csv_logger.close()
        val_csv_logger.close()
        test_csv_logger.close()

def check_args(args):
    if args.shift_type == 'confounder':
        assert args.confounder_names
        assert args.target_name
    elif args.shift_type.startswith('label_shift'):
        assert args.minority_fraction
        assert args.imbalance_ratio


def set_output_dir(args):
    if not os.listdir(args.log_dir):
        n = 1
    else:
        n = len(next(os.walk(args.log_dir))[1]) + 1
    args.log_dir = os.path.join(args.log_dir,
                                 f'{n}-'
                                 f'{args.dataset}-'
                                 f'{args.train_type}-'
                                 f'{args.test_type}-re'
                                 f'{args.reweight_groups}-ro'
                                 f'{args.robust}-'
                                 f'{args.random_init}'
                                 f'{args.epsilon}')
    return args



if __name__=='__main__':
    main()
