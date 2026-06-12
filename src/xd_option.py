import argparse

parser = argparse.ArgumentParser(description='VadCLIP')
parser.add_argument('--seed', default=234, type=int)

parser.add_argument('--embed-dim', default=512, type=int)
parser.add_argument('--visual-length', default=256, type=int)
parser.add_argument('--visual-width', default=512, type=int)
parser.add_argument('--visual-head', default=1, type=int)
parser.add_argument('--visual-layers', default=1, type=int)
parser.add_argument('--attn-window', default=64, type=int)
parser.add_argument('--prompt-prefix', default=10, type=int)
parser.add_argument('--prompt-postfix', default=10, type=int)
parser.add_argument('--classes-num', default=7, type=int)

parser.add_argument('--max-epoch', default=10, type=int)
parser.add_argument('--model-path', default='./XD_model/my_model_xd.pth')
parser.add_argument('--use-checkpoint', action='store_true', help='Use this flag to resume from checkpoint')
parser.add_argument('--checkpoint-path', default='./XD_model/checkpoint.pth')
parser.add_argument('--batch-size', default=96, type=int)
parser.add_argument('--train-list', default='list/xd_CLIP_rgb.csv')
parser.add_argument('--test-list', default='list/xd_CLIP_rgbtest.csv')
parser.add_argument('--gt-path', default='list/gt.npy')
parser.add_argument('--gt-segment-path', default='list/gt_segment.npy')
parser.add_argument('--gt-label-path', default='list/gt_label.npy')

parser.add_argument('--lr', default=1e-5)
parser.add_argument('--scheduler-rate', default=0.1)
parser.add_argument('--scheduler-milestones', nargs='+', type=int, default=[1])

xd_label_map = {
    'A':  'Normal videos without any anomaly', 
    'B1': 'Fighting and violence', 
    'B2': 'Shooting and gunfire', 
    'B4': 'Riot and crowd violence', 
    'B5': 'Abuse and torture', 
    'B6': 'Car Accident and crash', 
    'G':  'Explosion and bombing'
}

                                           
                                                                   
                                                                      

parser.add_argument('--branch-mode', default='dual', type=str,
                    choices=['dual', 'aux_only', 'dual_wo_csu'])

parser.add_argument('--lambda-align', default=1.0, type=float,
                    help='Weight for semantic alignment loss L_align')

parser.add_argument('--lambda-smooth', default=0.25, type=float,
                    help='Weight for temporal smoothness loss L_smooth')

                        
parser.add_argument('--v-enhance', default='dual', type=str,
                    choices=['dual', 'cnn'], 
                    help='Ablation for DualGraphEnhancement: cnn: baseline VisualEnhancement')

                      
parser.add_argument('--fusion-mode', default='dynamic', type=str,
                    choices=['dynamic', 'standard'], 
                    help='Ablation for fusion: dynamic (default) or standard (w only)')

                          
parser.add_argument('--prompt-level', default='optimal', type=str,
                    choices=['coarse', 'detailed', 'optimal'], 
                    help='Ablation for label_map prompt engineering')
