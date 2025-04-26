## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:

```
Developed by: SHAHIN J
Register no: 212223040190
```
```
import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df
# ORDINAL ENCODING
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
![image](https://github.com/user-attachments/assets/ff502dcf-a01a-433c-bc70-ef9951183a0a)

df['bo2']=e1.fit_transform(df[["ord_2"]])
df
![image](https://github.com/user-attachments/assets/a7afcdac-a3f4-4eed-8ba7-a1bc9c4f09a4)

# Label Encoder (Orders in Alphabetical order)
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
![image](https://github.com/user-attachments/assets/976092a1-7f66-447b-91bb-2488545d9a2f)

# ONE HOT ENCODING
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
df2=pd.concat([df2,enc],axis=1)
df
![image](https://github.com/user-attachments/assets/8ec5bab5-3f58-4897-94fa-68420cc498a8)

pd.get_dummies(df2,columns=["nom_0"])
![image](https://github.com/user-attachments/assets/649ccc3d-aa48-4cbf-80a8-5e56e4d019b9)

pip install --upgrade category_encoders
![image](https://github.com/user-attachments/assets/c1ca022d-3566-4778-b881-94351af5f522)

from category_encoders import BinaryEncoder
df = pd.read_csv("/content/data.csv")
df
![image](https://github.com/user-attachments/assets/519d2dfc-be4f-49bc-95fb-60755c9d916b)

be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
df
dfb=pd.concat([df,nd],axis=1)
dfb
![image](https://github.com/user-attachments/assets/83adca19-276d-42a0-bcb1-cc4c9ab894a8)

from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
![image](https://github.com/user-attachments/assets/d6db7748-51a0-44ec-b0bf-5ec5501fc418)

import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
![image](https://github.com/user-attachments/assets/10127f58-d6fc-4b2b-9509-e3e9f9416f0f)

df.skew()
![image](https://github.com/user-attachments/assets/5b3ff74b-8862-4467-b49d-56593b84e688)

np.log(df["Highly Positive Skew"])
![image](https://github.com/user-attachments/assets/2f9323d9-4847-42c5-a2b9-bdbc60868e25)

np.reciprocal(df["Moderate Positive Skew"])
![image](https://github.com/user-attachments/assets/ab95bf6a-86c1-47b4-bdbb-6b154fc8c1b3)

np.sqrt(df["Highly Positive Skew"])
![image](https://github.com/user-attachments/assets/1ad2b943-3e31-4e1c-9a60-0ff896c99068)

np.square(df["Highly Positive Skew"])
![image](https://github.com/user-attachments/assets/4428d192-63d9-46a5-a9d2-e798ec076a47)

# POWER TRANSFORMATION
# BOX_COX
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
![image](https://github.com/user-attachments/assets/ab2ba06a-5175-48f4-ad4e-e498319aa968)

df.skew()
![image](https://github.com/user-attachments/assets/23404996-049d-40b0-9e71-e9b6d0268ecd)

# YEO JOHNSON
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
![image](https://github.com/user-attachments/assets/0e976595-7a0b-4de6-b191-623b24a75c2f)

# QUANTILE TRANSFORMATION
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
![image](https://github.com/user-attachments/assets/47474b33-e694-4bd6-b16d-d8a9cb265720)

import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
![image](https://github.com/user-attachments/assets/1ec3f7e0-e516-427a-a378-8e9f0d3246ac)

sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
![image](https://github.com/user-attachments/assets/b7170c53-1186-4673-9258-4fe2d7e3f178)

from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
![image](https://github.com/user-attachments/assets/76afa6e1-ad9e-490c-a360-aa3b77f3721d)

df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
![image](https://github.com/user-attachments/assets/164fa91b-22d0-48f6-9a86-6a6e754fc925)

```
# RESULT:
Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.

       
