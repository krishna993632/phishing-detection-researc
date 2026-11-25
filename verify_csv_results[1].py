"""Independent arithmetic verification of the supplied phishing result archive.
Run from a directory containing results.csv and confusion_matrices.csv.
It does not rerun machine-learning model training or independently confirm source-data provenance.
"""
import pandas as pd
from pathlib import Path
D=Path(__file__).resolve().parent
if not (D/'results.csv').exists():
    D=Path.cwd()
a=pd.read_csv(D/'results.csv')
b=pd.read_csv(D/'confusion_matrices.csv')
c=a.merge(b, on=['seed','condition','model'], validate='one_to_one')
assert len(c)==20 and not a.duplicated(['seed','condition','model']).any()
for _,r in c.iterrows():
    tn,fp,fn,tp=[int(r[k]) for k in ('tn','fp','fn','tp')]
    n=tn+fp+fn+tp
    assert n==2211 and tp+fn==980 and tn+fp==1231
    expected={'accuracy':(tp+tn)/n,'precision_phishing':tp/(tp+fp),
              'recall_phishing':tp/(tp+fn),'f1_phishing':2*tp/(2*tp+fp+fn)}
    for key,v in expected.items():assert abs(v-r[key])<1e-12,(key,r['seed'],r['model'])
    assert fp==r['false_positive_phishing'] and fn==r['false_negative_phishing']
print('PASS: all 20 result rows agree with confusion matrices.')
print(a.groupby(['condition','model'])[['accuracy','recall_phishing','precision_phishing','f1_phishing']].mean().to_string())
