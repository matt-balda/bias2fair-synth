import pandas as pd
import numpy as np
from aif360.datasets import BinaryLabelDataset
from aif360.algorithms.preprocessing import DisparateImpactRemover
from aif360.sklearn.preprocessing import LearnedFairRepresentations

# Dummy data
df = pd.DataFrame({
    'feature1': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    'race': [0, 1, 0, 1, 0, 1], # sensitive attribute
    'two_year_recid': [1, 0, 1, 0, 1, 0] # target
})

print("Original DF:\n", df)

# DIRemover expects BinaryLabelDataset
bld = BinaryLabelDataset(df=df, label_names=['two_year_recid'], protected_attribute_names=['race'])
dir_remover = DisparateImpactRemover(repair_level=1.0)
bld_repaired = dir_remover.fit_transform(bld)
df_repaired, _ = bld_repaired.convert_to_dataframe()

print("\nDIRemover Repaired DF:\n", df_repaired)

# LFR uses aif360.sklearn.preprocessing
# LFR requires the dataframe to have a multi-index or we can just pass prot_attr
X = df.drop(columns=['two_year_recid'])
X = X.set_index('race', drop=False)
y = df['two_year_recid']

# Note: LFR needs some categorical / continuous features handling maybe?
try:
    lfr = LearnedFairRepresentations(prot_attr='race')
    X_lfr, y_lfr = lfr.fit_transform(X, y)
    print("\nLFR Transform X:\n", X_lfr)
    print("\nLFR Transform y:\n", y_lfr)
except Exception as e:
    import traceback
    traceback.print_exc()

