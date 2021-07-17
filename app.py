"""Main application logic."""
import ast
import base64

import numpy as np
import pandas as pd
import streamlit as st

import utils

st.write("""
## [Accuracy on the Line: On the Strong Correlation Between Out-of-Distribution and In-Distribution Generalization](https://arxiv.org/abs/2107.04649)
[John Miller](https://people.eecs.berkeley.edu/~miller_john/),
[Rohan Taori](https://www.rohantaori.com/),
[Aditi Raghunathan](https://stanford.edu/~aditir/),
[Shiori Sagawa](https://cs.stanford.edu/~ssagawa/),
[Pang Wei Koh](https://koh.pw/),
[Vaishaal Shankar](http://vaishaal.com/),
[Percy Liang](https://cs.stanford.edu/~pliang/),
[Yair Carmon](https://www.cs.tau.ac.il/~ycarmon/),
[Ludwig Schmidt](https://people.csail.mit.edu/ludwigs/)

### Plotting
Visualize and download evaluation data from our testbed below (options in the left sidebar).
"""
)

df = pd.read_csv("results.csv",
    converters={
        "hyperparameters": ast.literal_eval,
        "test_accuracy_ci": ast.literal_eval,
        "shift_accuracy_ci": ast.literal_eval
    })
df["model_types"] = df.apply(utils.get_model_type, axis=1)

train_sets = sorted(df.train_set.unique())
train_set = st.sidebar.selectbox(
    "Train dataset:", train_sets, index=train_sets.index("cifar10-train"))

df_train = df[df.train_set == train_set]
test_sets = sorted(df_train.test_set.unique())
shift_sets = sorted(df_train.shift_set.unique())
if "cifar10-train" == train_set:
    test_default_idx = test_sets.index("cifar10-test")
    shift_default_idx = shift_sets.index("cifar10.2-test")
else:
    test_default_idx, shift_default_idx = 0, 0
test_set = st.sidebar.selectbox(
    "Test dataset (x-axis):", test_sets, index=test_default_idx)
shift_set = st.sidebar.selectbox(
    "Shift dataset (y-axis):", shift_sets, index=shift_default_idx)

selected_df = df_train[
    (df_train.test_set == test_set)
    & (df_train.shift_set == shift_set)
]

scaling = st.sidebar.selectbox(
    "Axis scaling:", ["probit", "logit", "linear"], index=0)

if st.sidebar.checkbox(f"Show only a subset of models?", value=False):
    model_types = list(selected_df.model_types.unique())
    types_to_show = set(st.sidebar.multiselect(f"Models to show", options=model_types))
    if len(types_to_show):
        selected_df = selected_df[selected_df.model_type.isin(types_to_show)]

st.plotly_chart(utils.plot(selected_df, scaling=scaling))

"To visualize only a subset of model types (e.g. just Linear Models), check the box to show only a subset of models in the left sidebar."

"Click the link below to download the raw data for this plot as a csv"
# Encode data for download
df_to_download = selected_df.to_csv(index=False)
b64 = base64.b64encode(df_to_download.encode()).decode()

linkname = f"train:{train_set}_test:{test_set}.csv"
link = f'<a href="data:file/txt;base64,{b64}" download="{linkname}"> Download data as csv</a>'
st.markdown(link, unsafe_allow_html=True)

st.write("""
### Citation
```
@inproceedings{miller2021accuracy,
    title={Accuracy on the Line: On the Strong Correlation Between Out-of-Distribution and In-Distribution Generalization},
    author={Miller, John and Taori, Rohan and Raghunathan, Aditi and Sagawa, Shiori and Koh, Pang Wei and Shankar, Vaishaal and Liang, Percy and Carmon, Yair and Schmidt, Ludwig},
    booktitle={International Conference on Machine Learning (ICML)},
    year={2021},
    note={\\url{https://arxiv.org/abs/2007.00644}},
}
```
""")
