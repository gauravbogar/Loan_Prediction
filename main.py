import os
import pickle
import warnings

import numpy as np
import pandas as pd
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")


current_path = os.getcwd()
assets_path = os.path.join(current_path, "assets")
data = os.path.join(assets_path, "loan.csv")
pickle_path = os.path.join(assets_path, "loan.pkl")
df = pd.read_csv(data)
df = df.dropna()
x = df.drop("bad_loan", axis=1)
y = df["bad_loan"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

num_feat = [feature for feature in x.columns if x[feature].dtype != "O"]
cat_feat = [feature for feature in x.columns if x[feature].dtype == "O"]

col_transformer = ColumnTransformer(
    transformers=[
        ("ss", StandardScaler(), num_feat),
        ("ordinal", OrdinalEncoder(), cat_feat),
    ],
    remainder="drop",
)


model = BalancedBaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    sampling_strategy="auto",
    replacement=False,
    random_state=0,
)

pipe = Pipeline([("preprocessing", col_transformer), ("model", model)])
pipe.fit(x_train, y_train)
# filename = "loan.pkl"
# pickle.dump(pipe, open(filename, "wb"))
with open(pickle_path, "wb") as f:
    pickle.dump(pipe, f)
print("Pickle File created at {}".format(pickle_path))

