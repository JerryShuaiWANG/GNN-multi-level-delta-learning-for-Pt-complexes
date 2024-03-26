# Multi-level Δ-learning for Predicting Experimental Radiative Decay Rate Constant of Phosphorescent Platinum(II) Complexes


<p align='center'>
  <img src='figure/TOC.png' width="600px">
</p>

## Abstract
The utilization of phosphorescent metal complexes as emissive dopants for organic light-emitting diodes (OLEDs) has been the subject of intense research. Cyclometalated Pt(II) complexes are particularly popular triplet emitters due to their color-tunable emissions. To make them viable for practical applications as OLED emitters, it is essential to develop Pt(II) complexes with high radiative decay rate constant (*k*<sub>r</sub>). To this end, an efficient and accurate prediction tool for small experimental *k*<sub>r</sub> sample is highly desirable. In this work, **two new datasets are established including 526K Pt-complexes structures and 467 first-principles calculated structures with simulated *k*<sub>r</sub> values**. We propose a general yet powerful **multi-level Δ-learning** protocol achieving high accuracy in predicting *k*<sub>r</sub> values. The structure of the protocol is exemplified with two major parts: a GNN semi-supervised regression model for first-principles calculated *k*<sub>r</sub> and a supervised regression model for experimental *k*<sub>r</sub>. The former model can be utilized for high throughput virtual screening (HTVS) while the latter for highly accurate *k*<sub>r</sub> predictions. **The multi-level Δ-learning approach offers a way of evaluating k<sub>r</sub> from different accuracy levels with more enhanced precision.** Besides, this work **first solves the problem of metal-complex representation via GNN considering coordination bonds**. Among 526K structures dataset, 52 new Pt-structures are screened out and their accurate evaluation results are presented. We expect this protocol will become a valuable tool for small sampling problems, expediting the rapid development of novel OLED materials and offering guidance for the future advancement of ML models for metal-complex systems.

Phosphorescent metal complexes have been under intense investigations as emissive dopants for energy efficient organic light emitting diodes (OLEDs). Among them, cyclometalated Pt(II) complexes are widespread triplet emitters with color-tunable emissions. To render their practical applications as OLED emitters, it is in great need to develop Pt(II) complexes with high radiative decay rate constant (k<sub>r</sub>) and photoluminescence (PL) quantum yield. Thus, an efficient and accurate prediction tool is highly desirable. **Here, we develop a general protocol for accurate predictions of emission wavelength, radiative decay rate constant, and PL quantum yield for phosphorescent Pt(II) emitters based on the combination of first-principles quantum mechanical method, machine learning (ML) and experimental calibration**. A new dataset concerning phosphorescent Pt(II) emitters is constructed, with more than two hundred samples collected from the literature. Features containing pertinent electronic properties of the complexes are chosen. Our results demonstrate that **ensemble learning models** combined with stacking-based approaches exhibit the best performance, where the values of **squared correlation coefficients (R<sup>2</sup>), mean absolute error (MAE), and root mean square error (RMSE) are 0.96, 7.21 nm and 13.00 nm for emission wavelength prediction, and 0.81, 0.11 and 0.15 for PL quantum yield prediction. For radiative decay rate constant (k<sub>r</sub>), the obtained value of R<sup>2</sup> is 0.67 while MAE and RMSE are 0.21 and 0.25 (both in log scale)**, respectively. **The accuracy of the protocol is further confirmed using 24 recently reported Pt(II) complexes**, which demonstrates its reliability for a broad palette of Pt(II) emitters. We expect this protocol will become a valuable tool, accelerating the rational design of novel OLED materials with desired properties.



## Requirements

Python: 3.7.0

rdkit=2020.09.1.0

scikit-learn=0.23.2

hyperopt=0.2.7

Detials can be seen at requirements.txt

## Dataset

The dataset directory should look like this:
```bash
datasets
├── data in training and independent testing
│   ├── optimized_T1_structures.zip
│   └── all_exp_data_with_ref.csv / all_features_with_all_exp_data_with_ref.csv
└──  data in external testing
│   ├── external_test_samples_T1_structures.zip
│   └── external_test_features_with_exp_data_with_ref.csv  
└──  division method: ks_sampling.py    

```

## Usage

1.You could download the traning data and testing data at the directory you have interest (for instance, emission-wavelength) or spilt it based on the raw_data

2.Run python main_for_test.py and main_for_stacking.py to test the model performance before and after stacking repectively.

## Results

```bash
# Default results directory is:
./_/

```

## Note

- raw_data includes the optimized T1 structures in .xyz format and the features with experimental data for all the samples occurred in the paper. 
- All the code and data for three properties of Pt emitters can be seen in the directories respectively.
- The dataset, model, and code are for non-commercial research purposes only.
- If there are any questions, please contace me freely.
- Relative paths are updated so that it is convenient to get the correct data.
