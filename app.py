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

@st.cache
def load_data():
    df = pd.read_csv("results.csv",
        converters={
            "hyperparameters": ast.literal_eval,
            "test_accuracy_ci": ast.literal_eval,
            "shift_accuracy_ci": ast.literal_eval,
            "test_macro_f1_ci": ast.literal_eval,
            "shift_macro_f1_ci": ast.literal_eval,
            "test_worst_region_accuracy_ci": ast.literal_eval,
            "shift_worst_region_accuracy_ci": ast.literal_eval,
        })
    df["model_type"] = df.apply(utils.get_model_type, axis=1)
    return df

df = load_data()

# TODO: Add YCB-Objects
universe = st.sidebar.selectbox(
    "Dataset universe", ["CIFAR-10", "WILDS", "YCB-Objects"], index=0)

if universe == "CIFAR-10":
    shift_type = st.sidebar.selectbox(
        "Distribution shift type", [
            "Dataset reproduction",
            "Benchmark shift",
            "Synthetic perturbations",
        ],
        index=0
    )
    train_sets = ["cifar10-train"]
    if shift_type == "Dataset reproduction":
        test_sets = ["cifar10-test"]
        shift_sets = ["cifar10.2-test", "cifar10.1-v6", "cifar10.2-all"]
    elif shift_type == "Benchmark shift":
        test_sets = ["cifar10-test", "cifar10-test-STL10classes"]
        shift_sets = ["cinic10", "STL10"]
    else:
        test_sets = ["cifar10-test"]
        shift_sets = sorted([ts for ts in df.shift_set.unique() if "cifar10c" in ts])
    metrics = ["accuracy"]
elif universe == "WILDS":
    task = st.sidebar.selectbox(
        "Task", [
            "FMoW", "IWildCam", "Camelyon17"
        ],
        index=0
    )
    if task == "FMoW":
        train_sets = ["FMoW-train"]
        test_sets = ["FMoW-id_val", "FMoW-id_test"]
        shift_sets = ["FMoW-ood_val", "FMoW-ood_test"]
        metrics = ["accuracy", "worst_region_accuracy"]
    elif task == "IWildCam":
        train_sets = ["IWildCamOfficialV2-train"]
        test_sets = ["IWildCamOfficialV2-id_val", "IWildCamOfficialV2-id_test"]
        shift_sets = ["IWildCamOfficialV2-ood_val", "IWildCamOfficialV2-ood_test"]
        metrics = ["accuracy", "macro_f1"]
    elif task == "Camelyon17":
        train_sets = ["Camelyon17-train"]
        test_sets = ["Camelyon17-id_val", "Camelyon17-id_test"]
        shift_sets = ["Camelyon17-ood_val", "Camelyon17-ood_test"]
        metrics = ["accuracy"]
elif universe == "YCB-Objects":
    train_sets = ["YCB Train 50k examples", "YCB Train 100k examples"]
    test_sets = ["YCB ID Test"]
    shift_sets = ["YCB OOD Test"]
    metrics = ["accuracy"]


train_set = st.sidebar.selectbox(
    "Train dataset:", train_sets, index=0)
test_set = st.sidebar.selectbox(
    "Test dataset (x-axis):", test_sets, index=0)
shift_set = st.sidebar.selectbox(
    "Shift dataset (y-axis):", shift_sets, index=0)

selected_df = df[
    (df.train_set == train_set)
    & (df.test_set == test_set)
    & (df.shift_set == shift_set)
]

scaling = st.sidebar.selectbox(
    "Axis scaling:", ["probit", "logit", "linear"], index=0)

metric = st.sidebar.selectbox(
    "Metric: ", metrics, index=0)

if st.sidebar.checkbox(f"Show only a subset of models?", value=False):
    model_types = list(selected_df.model_type.unique())
    types_to_show = set(st.sidebar.multiselect(f"Models to show", options=model_types))
    if len(types_to_show):
        selected_df = selected_df[selected_df.model_type.isin(types_to_show)]

st.plotly_chart(utils.plot(selected_df, scaling=scaling, metric=metric))

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
