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
AUC of apple test set: 0.910711111111111
AUC of backpack test set: 0.9273721340388009
AUC of banana test set: 0.877037037037037
AUC of baseballbat test set: 0.8273544973544971
AUC of baseballglove test set: 0.860888888888889
AUC of bench test set: 0.9729037037037036
AUC of bicycle test set: 0.9424296296296296
AUC of bottle test set: 0.9137925925925926
AUC of bowl test set: 0.8927635327635328
AUC of broccoli test set: 0.8995987654320988
AUC of cake test set: 0.8799012345679013
AUC of car test set: 0.9006042884990255
AUC of carrot test set: 0.8702495974235104
AUC of cellphone test set: 0.7632740740740741
AUC of chair test set: 0.9613963388676036
AUC of cup test set: 0.8436296296296296
AUC of donut test set: 0.9254222222222225
AUC of hairdryer test set: 0.9327739984882844
AUC of handbag test set: 0.9097470641373081
AUC of hydrant test set: 0.9602814814814816
AUC of keyboard test set: 0.8423816221284578
AUC of laptop test set: 0.8724400871459694
AUC of microwave test set: 0.8678814814814813
AUC of motorcycle test set: 0.9686074074074076
AUC of mouse test set: 0.9119727891156462
AUC of orange test set: 0.8842222222222224
AUC of parkingmeter test set: 0.9535802469135802
AUC of pizza test set: 0.8480776014109348
AUC of plant test set: 0.9539009139009138
AUC of stopsign test set: 0.8870445956160243
AUC of teddybear test set: 0.9355314009661838
AUC of toaster test set: 0.9593037037037035
AUC of toilet test set: 0.8095785440613028
AUC of toybus test set: 0.9063247863247862
AUC of toyplane test set: 0.8142260208926876
AUC of toytrain test set: 0.8369444444444443
AUC of toytruck test set: 0.8628368794326243
AUC of tv test set: 0.9395061728395063
AUC of umbrella test set: 0.9398666666666665
AUC of vase test set: 0.9519400352733683
AUC of wineglass test set: 0.8819363222871994

****************************************************************************************************

Mean AUC: 0.8975667260043426
```

Note that this evaluation implementation may differ slightly from the internal one used for the paper, while our reported AUC@30 value is 89.8%, which is slightly better than the value of 88.2% reported in the paper.

## Checklist

The following features are planned for future releases:

- [x] Camera pose estimation code on Co3D
- [ ] VGGT+BA (Bundle Adjustment) on Co3D
- [ ] Evaluation on Re10K dataset
- [ ] Evaluation on IMC dataset
- [ ] Evaluation of multi-view depth estimation

---

