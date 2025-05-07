# VGGT Evaluation

This repository contains code to reproduce the evaluation results of VGGT.

## Table of Contents

- [Camera Pose Estimation on Co3D](#camera-pose-estimation-on-co3d)
  - [Dataset Preparation](#dataset-preparation)
  - [Running the Evaluation](#running-the-evaluation)
  - [Expected Results](#expected-results)
- [Checklist](#checklist)

## Camera Pose Estimation on Co3D

### Dataset Preparation

1. Download the Co3D dataset from the [official repository](https://github.com/facebookresearch/co3d)

2. Preprocess the Co3D dataset (takes approximately 5 minutes):

   ```bash
   python preprocess_co3d.py --category all --co3d_v2_dir /YOUR/CO3D/PATH --output_dir /YOUR/CO3D/ANNO/PATH
   ```

   Replace `/YOUR/CO3D/PATH` with the path to your downloaded Co3D dataset, and `/YOUR/CO3D/ANNO/PATH` with the desired output directory for the processed annotations.

### Running the Evaluation

0. Install vggt as a package, pip install -e .

1. Update the Co3D paths in `test_co3d.py`:
   - Set `CO3D_DIR` to your Co3D dataset directory
   - Set `CO3D_ANNOTATION_DIR` to your processed annotations directory

2. Run the evaluation script:

   ```bash
   python test_co3d.py
   ```

### Expected Results

After the evaluation completes, you should see results similar to:

```
apple          : 0.9029
backpack       : 0.9227
banana         : 0.8749
baseballbat    : 0.8301
baseballglove  : 0.8381
bench          : 0.9744
bicycle        : 0.9463
bottle         : 0.9126
bowl           : 0.8894
broccoli       : 0.8857
cake           : 0.8689
car            : 0.9087
carrot         : 0.8797
cellphone      : 0.7784
chair          : 0.9593
cup            : 0.8517
donut          : 0.9211
hairdryer      : 0.9347
handbag        : 0.9008
hydrant        : 0.9580
keyboard       : 0.8403
laptop         : 0.8631
microwave      : 0.8581
motorcycle     : 0.9616
mouse          : 0.9030
orange         : 0.8850
parkingmeter   : 0.9125
pizza          : 0.8448
plant          : 0.9510
stopsign       : 0.8851
teddybear      : 0.9344
toaster        : 0.9598
toilet         : 0.8088
toybus         : 0.9007
toyplane       : 0.8312
toytrain       : 0.8452
toytruck       : 0.8566
tv             : 0.9405
umbrella       : 0.9461
vase           : 0.9509
wineglass      : 0.8759
--------------------------------------------------
Mean AUC: 0.8949
```

Note that this evaluation implementation may differ slightly from the internal one used for the paper, while our reported AUC@30 value is 89.8%, which is slightly better than the value of 88.2% reported in the paper.

## Checklist

The following features are planned for future releases:

- [x] Camera pose estimation code on Co3D
- [x] VGGT+BA (Bundle Adjustment) on Co3D
- [ ] Evaluation on Re10K dataset
- [ ] Evaluation on IMC dataset
- [ ] Evaluation of multi-view depth estimation

---
