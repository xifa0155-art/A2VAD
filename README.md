# A2VAD

Code for **A2VAD: Audio-Aware Video Representation Learning for Weakly Supervised Video Anomaly Detection**.

## Environment

```bash
conda env create -f environment.yml
conda activate a2vad
```

## Data

This repository does not include datasets, extracted features, or model checkpoints.

Please prepare the extracted features and place them under:

```text
Dataset/features/
```

The file lists and ground-truth files are provided in `list/`.

## Training

Train on XD-Violence:

```bash
python src/xd_train.py
```

Train on UCF-Crime:

```bash
python src/ucf_train.py
```

## Evaluation

Evaluate on XD-Violence:

```bash
python src/xd_test.py --model-path ./XD_model/my_model_xd.pth
```

Evaluate on UCF-Crime:

```bash
python src/ucf_test.py --model-path ./UCF_model/my_model_ucf.pth
```

## Checkpoints

Model checkpoints are not included in this repository.

Please place checkpoints under:

```text
XD_model/
UCF_model/
```

## License

This project is released under the license in `LICENSE`.
