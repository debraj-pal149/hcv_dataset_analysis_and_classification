# hcv_dataset_analysis_and_classification
We analyse the HCV dataset which contains laboratory values of blood donors and Hepatitis C patients and demographic values like age. We also build a classification model for eligible blood donors and otherwise Hepatitis C patients based on their clinical records. 


The target attribute for classification is Category (blood donors vs. Hepatitis C (including its progress ('just' Hepatitis C, Fibrosis, Cirrhosis).

All attributes except Category and Sex are numerical. The laboratory data are the attributes 5-14. 1) X (Patient ID/No.) 2) Category (diagnosis) (values: '0=Blood Donor', '0s=suspect Blood Donor', '1=Hepatitis', '2=Fibrosis', '3=Cirrhosis') 3) Age (in years) 4) Sex (f,m) 5) ALB 6) ALP 7) ALT 8) AST 9) BIL 10) CHE 11) CHOL 12) CREA 13) GGT 14) PROT

The Category labels are 0 and 0s for subjects who are viable blood donors. Category labels 1=Hepatitis, 2=Fibrosis and 3=Cirrhosis represent subjects with different stages of Hepatitis-C who are not viable blood donors. The task at hand is to fit a classification model with high degree of accuracy which can predict which subjects are viable blood donors or Hepatitis C patients based on a number of available laboratory paramaters. The project done for my undergraduate course also involves analysis of feature mean differences using ANOVA.
