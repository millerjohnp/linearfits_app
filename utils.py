"""Plotting and downloading utilities."""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
import statsmodels.api as sm

import streamlit as st

COLOR_MAP = {
    "Neural Network": "blue",
    "ImageNet Pretrained Network": "cyan",
    "RandomFeatures": "green",
    "RandomForest": "orange",
    "KNN": "purple",
    "Linear Model": "yellow",
    "AdaBoost": "pink",
}

def get_model_type(row):
    family = row.model_family
    hparams = row.hyperparameters
    if family in ["AdaBoost", "KNN", "RandomFeatures", "RandomForest"]:
        return family
    if family in ["LogisticRegression", "RidgeClassifier", "SVM", "SGDClassifier"]:
        return "Linear Model"
    if "pretrained" in hparams and hparams["pretrained"]:
        return "ImageNet Pretrained Network"
    return "Neural Network"


def rescale(data, scaling=None):
    """Rescale the data."""
    if scaling == "probit":
        return norm.ppf(data)
    elif scaling == "logit":
        return np.log(data / (1 - data))
    elif scaling == "linear":
        return data
    raise NotImplementedError


def linear_fit(x, y):
    """Returns bias and slope from regression y on x."""
    x = np.array(x)
    y = np.array(y)
    
    invalid_idx = np.isnan(x) | np.isinf(x) | np.isnan(y) | np.isinf(y)
    x = x[~invalid_idx]
    y = y[~invalid_idx]

    covs = sm.add_constant(x, prepend=True)
    model = sm.OLS(y, covs)
    result = model.fit()
    return result.params, result.rsquared


def plot(df, scaling="probit", metric="accuracy"):
    """Generate an interactive scatter plot."""
    test_sets = df.test_set.unique()
    shift_sets = df.shift_set.unique()
    assert len(test_sets) == 1
    assert len(shift_sets) == 1
    test_set = test_sets[0]
    shift_set = shift_sets[0]

    title = f"{test_set} vs. {shift_set} {metric}"
    fig = make_subplots(
        rows=1, cols=1, subplot_titles=((f"{title} ({scaling} scaling)"), ),
    )
    traces = []
    for label, color in COLOR_MAP.items():
        traces.append(
            go.Scatter(x=[None], y=[None], mode='markers',
                marker=dict(size=8, color=color),
                showlegend=True, name=label))

    def map_colors(row):
        family = row.model_family
        hparams = row.hyperparameters
        if family in ["AdaBoost", "KNN", "RandomFeatures", "RandomForest"]:
            return COLOR_MAP[family]
        if family in ["LogisticRegression", "RidgeClassifier", "SVM", "SGDClassifier"]:
            return COLOR_MAP["Linear Model"]
        if "pretrained" in hparams and hparams["pretrained"]:
            return COLOR_MAP["ImageNet Pretrained Network"]
        return COLOR_MAP["Neural Network"]
    
    def get_name(row):
        model_name = row.model_family + "<br>"
        model_name += "<br>".join([f"{key}={val}" for key, val in row.hyperparameters.items()])
        return model_name

    # Generate the main scatter plot
    traces.extend(
        scatter_plot(
            xs=df[f"test_{metric}"],
            x_errs=list(df[f"test_{metric}_ci"].values),
            ys=df[f"shift_{metric}"],
            y_errs=list(df[f"shift_{metric}_ci"].values),
            model_names=list(df.apply(get_name, axis=1)),
            scaling=scaling,
            colors=df.model_type.apply(lambda x: COLOR_MAP[x]),
        )
    )
    
    metric_min, metric_max = 0.01, 0.99 # Avoid numerical issues
    traces.append(
        go.Scatter(
            mode="lines",
            x=rescale(np.arange(metric_min, metric_max + 0.01, 0.01), scaling),
            y=rescale(np.arange(metric_min, metric_max + 0.01, 0.01), scaling),
            name="y=x",
            line=dict(color="black", dash="dashdot")
        )
    )

    for trace in traces:
        fig.add_trace(trace, row=1, col=1)

    ax_range = [rescale(metric_min, scaling), rescale(metric_max, scaling)]
    fig.update_xaxes(title_text=f"{test_set} {metric}", range=ax_range, row=1, col=1)
    fig.update_yaxes(title_text=f"{shift_set} {metric}", range=ax_range, row=1, col=1)
    tickmarks = np.array([0.1, 0.25, 0.5, 0.7, 0.8, 0.9, 0.95, metric_max])
    ticks = dict(
        tickmode="array",
        tickvals=rescale(tickmarks, scaling),
        ticktext=[f"{mark:.2f}" for mark in tickmarks],
    )
    fig.update_layout(width=1000, height=700, xaxis=ticks, yaxis=ticks)
    return fig


def scatter_plot(
    xs,
    ys,
    x_errs,
    y_errs,
    model_names=None,
    scaling="linear",
    label="",
    colors="blue",
    fitlabel="Linear Fit",
    fitcolor="red",
):
    """Scatter plot Xs against Ys, optionally scaling the data."""
    xs =  np.array(xs).reshape(-1, 1)
    ys = np.array(ys).reshape(-1, 1)
    x_errs = np.asarray(x_errs)
    y_errs = np.asarray(y_errs)

    scaled_xs = rescale(xs, scaling)
    scaled_ys = rescale(ys, scaling)

    if x_errs is not None:
        scaled_x_errs = rescale(x_errs, scaling)
        x_delta = np.abs(scaled_xs - scaled_x_errs)
    else:
        x_delta = np.zeros((scaled_xs.shape[0], 2))

    if y_errs is not None:
        scaled_y_errs = rescale(y_errs, scaling)
        y_delta = np.abs(scaled_ys - scaled_y_errs)
    else:
        y_delta = np.zeros((scaled_ys.shape[0], 2))

    def label_point(i):
        x = xs[i, 0]
        y = ys[i, 0]
        label = f"{model_names[i]} <br>" if model_names is not None else ""
        if x_errs is not None:
            label += f"X: {x:.3f} ({x_errs[i, 0]:.3f}, {x_errs[i, 1]:.3f}) <br>"
        else:
            label += f"X: {x:.3f} <br>"
        if y_errs is not None:
            label += f"Y: {y:.3f} ({y_errs[i, 0]:.3f}, {y_errs[i, 1]:.3f}) <br>"
        else:
            label += f"Y: {y:.3f} <br>"
        return label

    traces = []
    traces.append(
        go.Scatter(
            x=scaled_xs.flatten(),
            y=scaled_ys.flatten(),
            hoverinfo="text",
            name=label,
            mode="markers",
            error_x=dict(
                type="data", symmetric=False, arrayminus=x_delta[:, 0], array=x_delta[:, 1], color="grey",
            ),
            error_y=dict(
                type="data", symmetric=False, arrayminus=y_delta[:, 0], array=y_delta[:, 1], color="grey",
            ),
            marker=dict(color=colors, size=6),
            text=[label_point(i) for i in range(len(scaled_xs))],
            showlegend=False,
        )
    )

    # Add linear fit
    (bias, slope), r2 = linear_fit(scaled_xs, scaled_ys)
    sample_pts = rescale(np.arange(0.0, 1.0, 0.01), scaling)
    traces.append(
        go.Scatter(
            mode="lines",
            x=sample_pts,
            y=slope * sample_pts + bias,
            name=f"{fitlabel} (Slope: {slope:.2f}, R2: {r2:.2f})",
            line=dict(color=fitcolor),
        )
    )

    return traces
