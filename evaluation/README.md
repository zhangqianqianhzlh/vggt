This repository currently contains the code to reproduce the results of VGGT (feedforward) on Co3D dataset.

## Camera pose estimation on Co3D

First download Co3D from the official website https://github.com/facebookresearch/co3d

Then proprocess the Co3D dataset (which should take around 5 minutes)

```
python preprocess_co3d.py --category all --co3d_v2_dir /YOUR/CO3D/PATH --output_dir /YOUR/CO3D/ANNO/PATH
```


Update your  /YOUR/CO3D/PATH and  /YOUR/CO3D/ANNO/PATH in test_co3d.py and run it, you will get results like:


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

Although this evaluation code is not exactly the same to our internal one, the result AUC@30=89.8% is slightly better than our reported number in paper AUC@30=88.2.



## Checklist

- [\tick] Camera pose estimation code on Co3D
- [ ] VGGT+BA for Co3D
- [ ] Evaluation for Re10K and IMC
- [ ] Evaluation for
