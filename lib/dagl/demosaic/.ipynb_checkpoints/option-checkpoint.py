import argparse
import template

parser = argparse.ArgumentParser(description='EDSR and MDSR')

parser.add_argument('--debug', action='store_true',
                    help='Enables debug mode')
parser.add_argument('--template', default='.',
                    help='You can set various templates in option.py')

# Hardware specifications
parser.add_argument('--n_threads', type=int, default=5,
                    help='number of threads for data loading, 6')
parser.add_argument('--cpu', action='store_true',
                    help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=1,
                    help='number of GPUs')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed')
# '/home/ubuntu/Documents/MC/RNAN _New/DN_Gray/code/DIV2K/Val/DIV2K_HQ'
# Data specifications
parser.add_argument('--dir_data', type=str, default='.',
                    help='dataset directory')
parser.add_argument('--dir_demo', type=str, default='../test',
                    help='demo image directory')
parser.add_argument('--data_train', type=str, default='DIV2K',
                    help='train dataset name')
parser.add_argument('--data_test', type=str, default='DIV2K',
                    help='test dataset name')
parser.add_argument('--benchmark_noise', action='store_true',
                    help='use noisy benchmark sets')
parser.add_argument('--n_train', type=int, default=800,
                    help='number of training set')
parser.add_argument('--n_val', type=int, default=5,
                    help='number of validation set')
parser.add_argument('--offset_val', type=int, default=800,
                    help='validation index offest')
parser.add_argument('--ext', type=str, default='img',
                    help='dataset file extension, img')
parser.add_argument('--scale', default='1',
                    help='super resolution scale')
parser.add_argument('--patch_size', type=int, default=64,
                    help='output patch size')
parser.add_argument('--rgb_range', type=int, default=1.,
                    help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--noise', type=str, default='.',
                    help='Gaussian noise std.')
parser.add_argument('--chop', action='store_true',default=True,
                    help='enable memory-efficient forward')
parser.add_argument('--data_range', type=str, default='1-1000/1-100',
                    help='train/test data range')

# Model specifications
parser.add_argument('--model', default='DAGL',
                    help='model name')
parser.add_argument('--blocks',type=int,default=3)
parser.add_argument('--stages',type=int,default=6)
parser.add_argument('--act', type=str, default='relu',
                    help='activation function')
parser.add_argument('--pre_train', type=str, default='.',
                    help='pre-trained model directory')
parser.add_argument('--extend', type=str, default='.',
                    help='pre-trained model directory')
parser.add_argument('--n_resblocks', type=int, default=16,
                    help='number of residual blocks')
parser.add_argument('--n_feats', type=int, default=64,
                    help='number of feature maps')
parser.add_argument('--res_scale', type=float, default=1,
                    help='residual scaling')
parser.add_argument('--shift_mean', default=True,
                    help='subtract pixel mean from the input')
parser.add_argument('--precision', type=str, default='single',
                    choices=('single', 'half'),
                    help='FP precision for test (single | half)')

# Training specifications
parser.add_argument('--reset', action='store_true',
                    help='reset the training')
parser.add_argument('--test_every', type=int, default=1000,
                    help='do test per every N batches')
parser.add_argument('--epochs', type=int, default=1500,
                    help='number of epochs to train')
parser.add_argument('--batch_size', type=int, default=32,
                    help='input batch size for training')
parser.add_argument('--split_batch', type=int, default=1,
                    help='split the batch into smaller chunks')
parser.add_argument('--self_ensemble', action='store_true',
                    help='use self-ensemble method for test')
parser.add_argument('--test_only', action='store_true',
                    help='set this option to test the model')
parser.add_argument('--gan_k', type=int, default=1,
                    help='k value for adversarial loss')

# Optimization specifications
parser.add_argument('--lr', type=float, default=2e-4,
                    help='learning rate')
parser.add_argument('--lr_decay', type=int, default=200,
                    help='learning rate decay per N epochs')
parser.add_argument('--decay_type', type=str, default='step',
                    help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAM',
                    choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='ADAM beta1')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='ADAM beta2')
parser.add_argument('--epsilon', type=float, default=1e-8,
                    help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='weight decay')

# Loss specifications
parser.add_argument('--loss', type=str, default='1*MSE',
                    help='loss function configuration')
parser.add_argument('--skip_threshold', type=float, default='1e6',
                    help='skipping batch that has large error')

# Log specifications
parser.add_argument('--save_path',type=str,default='res_greccr2bv2_adp_thd3')
parser.add_argument('--save_inter',type=int,default=50)
parser.add_argument('--save', type=str, default='test',
                    help='file name to save')
parser.add_argument('--load', type=str, default='.',
                    help='file name to load')
parser.add_argument('--resume', type=int, default=0,
                    help='resume from specific checkpoint')
parser.add_argument('--print_model', action='store_true',
                    help='print model')
parser.add_argument('--save_models', action='store_true',
                    help='save all intermediate models')
parser.add_argument('--print_every', type=int, default=20,
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_results', action='store_true',default=True,
                    help='save output results')
parser.add_argument("--ensemble",action='store_true')
# For test
parser.add_argument("--logdir", type=str, default='res_greccr2bv2_adp_thd3_800_32b/model/model_best.pt')
parser.add_argument("--test_data", type=str, default='testsets/Kodak24', help='test on Set12 or Set68')
parser.add_argument("--test_noiseL", type=float, default=30, help='noise level used on test set')
parser.add_argument('--results_path',type=str,default='res_vis/res_adp_thd3_32b_urban100')
parser.add_argument('--save_image',action='store_true')

# For DN
parser.add_argument('--noise_level', default='30',
                    help='Gaussian noise std. added to the whole image')
parser.add_argument('--n_cab_1', type=int, default=20,
                    help='number of n1 channel attention blocks')
parser.add_argument('--reduction', type=int, default=16,
                    help='reduction in channel attention layer')
parser.add_argument('--n_resgroups', type=int, default=10,
                    help='number of residual groups')
parser.add_argument('--resnb_p', type=int, default=1,
                    help='pre-processing residual unit number')
parser.add_argument('--resnb_t', type=int, default=2,
                    help='residual unit number in trunk branch')                    
parser.add_argument('--resnb_r', type=int, default=1,
                    help='residual unit or conv layer number before downsample')
parser.add_argument('--stack_connect', action='store_true',
                    help='stacking connection between downsample and upsample')
parser.add_argument('--testpath', type=str, default='/media/yulun/10THD1/data/Denoising/AllGray/LQ',
                    help='dataset directory for testing')
parser.add_argument('--testset', type=str, default='Kodak24',
                    help='dataset name for testing')

args = parser.parse_args()
template.set_template(args)

args.scale = list(map(lambda x: int(x), args.scale.split('+')))
args.noise_level = list(map(lambda x: int(x), args.noise_level.split('+')))

if args.epochs == 0:
    args.epochs = 1e8

for arg in vars(args):
    if vars(args)[arg] == 'True':
        vars(args)[arg] = True
    elif vars(args)[arg] == 'False':
        vars(args)[arg] = False

