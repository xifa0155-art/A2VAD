import argparse


parser = argparse.ArgumentParser(description="AGST/AVad UCF-Crime training and testing")

parser.add_argument("--seed", default=234, type=int)

                                                                
                                                 
parser.add_argument("--embed-dim", default=512, type=int)
parser.add_argument("--visual-length", default=256, type=int)
parser.add_argument("--visual-width", default=512, type=int)
parser.add_argument("--visual-head", default=1, type=int)
parser.add_argument("--visual-layers", default=2, type=int)
parser.add_argument("--attn-window", default=8, type=int)
parser.add_argument("--prompt-prefix", default=10, type=int)
parser.add_argument("--prompt-postfix", default=10, type=int)
parser.add_argument("--classes-num", default=14, type=int)
parser.add_argument("--eval-interval", default=40, type=int)

parser.add_argument(
    "--visual-root",
    default="./Dataset/features/UCF_CLIP/UCFClipFeatures",
    type=str,
)
parser.add_argument("--audio-root", default=None, type=str)
parser.add_argument("--audio-dim", default=512, type=int)
parser.add_argument("--train-list", default="list/ucf_CLIP_rgb.csv", type=str)
parser.add_argument("--test-list", default="list/ucf_CLIP_rgbtest.csv", type=str)
parser.add_argument("--gt-path", default="list/gt_ucf.npy", type=str)
parser.add_argument("--stride", default=16, type=int)

parser.add_argument("--max-epoch", default=5, type=int)
parser.add_argument("--batch-size", default=64, type=int)
parser.add_argument("--num-workers", default=4, type=int)
parser.add_argument("--lr", default=2e-5, type=float)
parser.add_argument("--scheduler-rate", default=0.1, type=float)
parser.add_argument("--scheduler-milestones", nargs="+", type=int, default=[4])

parser.add_argument("--model-path", default="UCF_model/my_model_ucf.pth", type=str)
parser.add_argument("--checkpoint-path", default="UCF_model/checkpoint_ucf.pth", type=str)
parser.add_argument("--use-checkpoint", action="store_true")

parser.add_argument("--lambda-align", default=1.0, type=float)
parser.add_argument("--lambda-smooth", default=0.0, type=float)
parser.add_argument("--lambda-norm", default=0.0, type=float)

parser.add_argument("--video-minmax", action="store_true")

parser.add_argument(
    "--test-crop-agg",
    default="mean",
    type=str,
    choices=["mean", "max", "top3", "center_max_avg"],
)

parser.add_argument("--smooth-sigma", default=0.0, type=float)
parser.add_argument(
    "--smooth-target",
    default="snippet",
    type=str,
    choices=["none", "snippet", "frame"],
)

parser.add_argument(
    "--train-sampling",
    default="linspace",
    type=str,
    choices=["linspace", "jitter"],
)

parser.add_argument("--lambda-rank", default=0.0, type=float)

parser.add_argument(
    "--feature-norm",
    default="none",
    type=str,
    choices=["none", "l2", "standard"],
)

parser.add_argument(
    "--branch-mode",
    default="dual",
    type=str,
    choices=["dual", "aux_only", "dual_wo_csu"],
)

parser.add_argument(
    "--score-mode",
    default="semantic",
    type=str,
    choices=["aux", "semantic", "avg", "weighted", "max"],
)

parser.add_argument("--score-alpha", default=0.5, type=float,
                    help="For --score-mode weighted: alpha*aux + (1-alpha)*semantic.")


parser.add_argument(
    "--test-crop-ensemble",
    action="store_true",
    help="Use 10-crop test-time score averaging for UCF features.",
)

parser.add_argument(
    "--test-crop-num",
    default=10,
    type=int,
    help="Number of UCF crops to ensemble at test time.",
)

parser.add_argument(
    "--ucf-label-mode",
    default="multiclass",
    type=str,
    choices=["binary", "multiclass"],
)

parser.add_argument(
    "--loss-mode",
    default="simple",
    type=str,
    choices=["simple", "aux_only", "xd"],
    help="simple=CLAS2+lambda_align*CLASM; xd=old XD-style loss; aux_only=CLAS2 only.",
)

parser.add_argument("--topk-divisor", default=32, type=int)

parser.add_argument(
    "--crop-fusion",
    default="none",
    type=str,
    choices=["none", "mean", "max"],
    help="Fuse 10-crop UCF features at feature level. none: use single crop path; mean/max: load __0~__9 and fuse to one [T,512] feature.",
)

parser.add_argument(
    "--test-window-mode",
    default="chunk",
    type=str,
    choices=["chunk", "sliding"],
    help="chunk: non-overlap inference; sliding: overlapping temporal windows.",
)

parser.add_argument(
    "--test-window-stride",
    default=128,
    type=int,
    help="Temporal stride for sliding window inference.",
)

parser.add_argument(
    "--train-crop",
    default="all",
    type=str,
    choices=["all", "center", "random"],
    help="all: use all 10-crop rows; center: keep only __5.npy; random: one random crop per video per epoch.",
)

parser.add_argument("--no-balanced-batch", dest="balanced_batch", action="store_false")
parser.set_defaults(balanced_batch=True)


ucf_binary_label_map = {
    "Normal": "normal surveillance activity",
    "Anomaly": "abnormal surveillance event",
}

ucf_multiclass_label_map = {
    "Normal": "normal",
    "Abuse": "abuse",
    "Arrest": "arrest",
    "Arson": "arson",
    "Assault": "assault",
    "Burglary": "burglary",
    "Explosion": "explosion",
    "Fighting": "fighting",
    "RoadAccidents": "road accidents",
    "Robbery": "robbery",
    "Shooting": "shooting",
    "Shoplifting": "shoplifting",
    "Stealing": "stealing",
    "Vandalism": "vandalism",
}


def get_ucf_label_map(label_mode: str):
    if label_mode == "binary":
        return ucf_binary_label_map
    if label_mode == "multiclass":
        return ucf_multiclass_label_map
    raise ValueError(f"Unknown label_mode: {label_mode}")
