from cProfile import label
from statistics import variance
import ML_Model as mlm
import Data_into_Matrix as DiM

import copy
import os
import random
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib

import bitsandbobs as bnb
import pickle
import seaborn as sns

matplotlib.rcParams["axes.prop_cycle"] = matplotlib.cycler(
    "color", bnb.plt.get_default_colors()
)
matplotlib.rcParams["axes.labelcolor"] = "black"
matplotlib.rcParams["axes.edgecolor"] = "black"
matplotlib.rcParams["xtick.color"] = "black"
matplotlib.rcParams["ytick.color"] = "black"
matplotlib.rcParams["xtick.labelsize"] = 8
matplotlib.rcParams["ytick.labelsize"] = 8
matplotlib.rcParams["xtick.major.pad"] = 2  # padding between text and the tick
matplotlib.rcParams["ytick.major.pad"] = 2  # default 3.5
matplotlib.rcParams["lines.dash_capstyle"] = "round"
matplotlib.rcParams["lines.solid_capstyle"] = "round"
matplotlib.rcParams["font.size"] = 8
matplotlib.rcParams["axes.titlesize"] = 8
matplotlib.rcParams["axes.labelsize"] = 8
matplotlib.rcParams["legend.fontsize"] = 8
matplotlib.rcParams["legend.facecolor"] = "#D4D4D4"
matplotlib.rcParams["legend.framealpha"] = 0.8
matplotlib.rcParams["legend.frameon"] = True
matplotlib.rcParams["axes.spines.right"] = False
matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["figure.figsize"] = [3.4, 2.7]  # APS single column
matplotlib.rcParams["figure.dpi"] = 200
matplotlib.rcParams["savefig.facecolor"] = (0.0, 0.0, 0.0, 0.0)  # transparent figure bg
matplotlib.rcParams["axes.facecolor"] = (1.0, 0.0, 0.0, 0.0)


PICKLE_DIRECTORY_NAME = mlm.PICKLE_DIRECTORY_NAME
FIGURE_DIRECTORY_PATH = "Figures/"
if not os.path.isdir(FIGURE_DIRECTORY_PATH):
    os.mkdir(FIGURE_DIRECTORY_PATH)


def plot_pca_evr(pickle_names=["pca/pca_last_pickle.pkl"], **kwargs):
    """
    """
    kwargs.setdefault("title", "")
    kwargs.setdefault("x_label", "principal component")
    kwargs.setdefault("y_label", "explained variance ratio")
    kwargs.setdefault("figsize", [3.4, 2.7])
    kwargs.setdefault("linewidth", 2)
    kwargs.setdefault("legend", ["SuperCon database", "reduced dataset"])
    kwargs.setdefault("xlim", [-1,90])

    evr = []
    x_ax = []
    for i in pickle_names:
        with open(PICKLE_DIRECTORY_NAME + i, "rb") as fid:
            pickle_data_loaded = pickle.load(fid)
        evr.append(pickle_data_loaded["evr"])
        x = []
        for i in range(0, len(pickle_data_loaded["evr"])):
            x.append(i+1)
        x_ax.append(x)
    
    fig, ax = plt.subplots(figsize=kwargs["figsize"])
    ax.set_xlabel(kwargs["x_label"])
    ax.set_ylabel(kwargs["y_label"])
    ax.set_title(kwargs["title"])

    for i in range(0, len(evr)):
        ax.plot(x_ax[i], evr[i], label=kwargs["legend"][i])
    ax.legend(loc="best")
    plt.xlim(kwargs["xlim"])
    plt.show()
    


def plot_pca(pickle_name="pca/pca_last_pickle.pkl", **kwargs):
    """
    MAIN FUNCTION: Plotting function to 'vpc'

    DESCRIPTION:
    Plots scatter diagram of first two principal components.

    NEW KEYWORD-ARGUMENTS:
    - colormap: The colormap that is used for the colorbar
    - num_discreet_color:   If (= 0): continuos color scheme will be used (default)
                            If (> 0): Determines number of discreet colors
    - colorbar_label: label for colorbar

    KNOWN KEYWORD-ARGUMENTS: (see seaborn and matplotlib documentation for explanation)
    - titel
    - x_label
    - y_label
    - marker
    - figsize
    - s (markersize)
    """
    kwargs.setdefault("title", "")
    kwargs.setdefault("x_label", "principal component 1")
    kwargs.setdefault("y_label", "principal component 2")
    kwargs.setdefault("colormap", "seismic")
    kwargs.setdefault("num_discreet_color", 0)
    kwargs.setdefault("marker", "o")
    kwargs.setdefault("colorbar_label", "$T_{c}$ (in K)")
    kwargs.setdefault("figsize", [3.4, 2.7])
    kwargs.setdefault("s", 20)
    kwargs.setdefault("manual_limit", False)
    kwargs.setdefault("xlim", [0, 1])

    with open(PICKLE_DIRECTORY_NAME + pickle_name, "rb") as fid:
        pickle_data_loaded = pickle.load(fid)

    comparted_ccm = pickle_data_loaded["ccm"]
    comparted_cT = pickle_data_loaded["cT"]
    min_cT = pickle_data_loaded["min"]
    max_cT = pickle_data_loaded["max"]

    if kwargs["num_discreet_color"] == 0:
        color = kwargs["colormap"]
    else:
        color = cm.get_cmap(kwargs["colormap"], kwargs["num_discreet_color"])

    fig, ax = plt.subplots(figsize=kwargs["figsize"])
    ax.set_xlabel(kwargs["x_label"])
    ax.set_ylabel(kwargs["y_label"])
    ax.set_title(kwargs["title"])

    for i in range(0, len(comparted_ccm)):
        X = comparted_ccm[i]
        y = comparted_cT[i]

        x_ax = []
        y_ax = []
        for j in range(0, len(X)):
            x_ax.append(X[j][0])
            y_ax.append(X[j][1])

        im = ax.scatter(
            x_ax, y_ax, c=y, cmap=color, vmin=min_cT, vmax=max_cT, marker=kwargs["marker"], s=kwargs["s"]
        )

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(kwargs["colorbar_label"])

    plt.show()


def bar_show_values(axs, decimal, orient, space, last = None):
    """
    SIDE FUNCTION:

    DESCRIPTION:
    Helps out in 'plot_pca_variance'. Prints values on top of bars.

    ARGUMENTS:
    - axs: axis of plot
    - decimal: number of decimal units
    - orient: orientation of bars
    - space: distance of value from bar
    - last: value of last bar. If None: last value is set like the others
    """
    decimal = '{:.' + str(decimal) + 'f}'
    def _single(ax):
        if orient == "v":
            for p in range(0, len(ax.patches)):
                _x = ax.patches[p].get_x() + ax.patches[p].get_width() / 2
                _y = ax.patches[p].get_y() + ax.patches[p].get_height() + float(space)
                if _y < 0:
                    _y = float(space)
                value = decimal.format(ax.patches[p].get_height()) 
                if last != None and p == len(ax.patches)-1:
                    value = decimal.format(last)
                ax.text(_x, _y, value, ha="center")
        elif orient == "h":
            for p in range(0, len(ax.patches)):
                _x = ax.patches[p].get_x() + ax.patches[p].get_width() + float(space)
                if _x < 0:
                    _x = float(space)
                _y = ax.patches[p].get_y() + ax.patches[p].get_height() - (ax.patches[p].get_height()*0.5)
                value = decimal.format(ax.patches[p].get_width())
                if last != None and p == len(ax.patches)-1:
                    value = decimal.format(last)
                ax.text(_x, _y, value, ha="left")

    if isinstance(axs, np.ndarray):
        for idx, ax in np.ndenumerate(axs):
            _single(ax)
    else:
        _single(axs)


def plot_pca_variance(
    pickle_name="pca/pca_last_pickle.pkl", 
    plot_bars = 7, 
    variant = 3, 
    **kwargs
    ):
    """
    MAIN FUNCTION: Plotting function to 'vpc'

    DESCRIPTION:
    Plots for some features the magnitude of importance on the two principal components
    as bar plot.

    ARGUMENTS:
    - pickle_name: location from which the data will be collected
    - plot_bars: number of features to be displayed as bar
    - variant: way of visualizing data:
        - variant = 1:  bars are plotted in order of highest feature value
        - variant = 2:  bars are plotted in order of highest absolute feature value
        - variant = 3:  bars are plotted in order like 'variant = 2'. The difference is
                        that the absolute values are plotted instead of the values itself
                        (like in 'variant = 1' and 'variant = 2')
        - variant = None: All available variants are plotted

    NEW KEYWORD-ARGUMENTS:
    - plot_remaining:   If 'True': remaining features are combined and plotted
                        If 'False': just the first "plot_bars" features are plotted
    - plot_remaining_reduced: if 'True': average of the remaining features is plotted
    - name_rest: label name of features that will be shown combined
    - rotation: rotation of feature labels
    - add_text: If 'True': add values of bars on top of them
        - decimal: number of decimal units displayed on top of bars
        - space: distance of values from bars
        - remaining_average: If 'True': value of last bar with remaining features is 
                                        average of them
    - grid: If 'True': plot grid
    - colormap: This time a vector with two colormap definitions to distinguish both
                bar plots

    KNOWN KEYWORD-ARGUMENTS: (see seaborn and matplotlib documentation for explanation)
    - orient
    - title
    - x_label
    - y_label
    - figsize
    - edgecolor
    - grid
    """
    kwargs.setdefault("colormap", ["Blues_d", "Reds_d"]) #flag
    diverging_colors_1 = sns.color_palette(kwargs["colormap"][0], plot_bars) #"Blues_d"
    diverging_colors_2 = sns.color_palette(kwargs["colormap"][1], plot_bars)
    kwargs.setdefault("orient", "v")
    kwargs.setdefault("title", "")
    if kwargs["orient"] == "v":
        kwargs.setdefault("x_label", "feature")
        kwargs.setdefault("y_label", "magnitude of value in PC")
    else:
        kwargs.setdefault("x_label", "magnitude of importance")
        kwargs.setdefault("y_label", "feature")
    kwargs.setdefault("name_rest", "other")
    kwargs.setdefault("figsize", [3.4, 2.7])
    kwargs.setdefault("rotation", 45)
    kwargs.setdefault("add_text", True)
    kwargs.setdefault("decimal", 2)
    kwargs.setdefault("space", 0.025)
    kwargs.setdefault("remaining_average", True)
    kwargs.setdefault("grid", True)
    kwargs.setdefault("plot_remaining", True)
    kwargs.setdefault("plot_remaining_reduced", True)
    kwargs.setdefault("edgecolor", "black")

    if not kwargs["plot_remaining"]:
        kwargs["remaining_average"] = False
        plot_bars+=1

    with open(PICKLE_DIRECTORY_NAME + pickle_name, "rb") as fid:
        pickle_data_loaded = pickle.load(fid)

    components_one = pickle_data_loaded["ordered_comp_one"]
    corresponding_name_one = pickle_data_loaded["ordered_name_one"]
    components_two = pickle_data_loaded["ordered_comp_two"]
    corresponding_name_two = pickle_data_loaded["ordered_name_two"]

    components_one_abs = pickle_data_loaded["ordered_comp_one_abs"]
    corresponding_name_one_abs = pickle_data_loaded["ordered_name_one_abs"]
    components_two_abs = pickle_data_loaded["ordered_comp_two_abs"]
    corresponding_name_two_abs = pickle_data_loaded["ordered_name_two_abs"]

    plot_comp_one = components_one[len(components_one):len(components_one)-plot_bars:-1]
    plot_name_one = corresponding_name_one[len(components_one):len(components_one)-plot_bars:-1]

    plot_comp_two = components_two[len(components_two):len(components_two)-plot_bars:-1]
    plot_name_two = corresponding_name_two[len(components_two):len(components_two)-plot_bars:-1]

    plot_comp_one_abs = components_one_abs[len(components_one):len(components_one)-plot_bars:-1]
    plot_name_one_abs = corresponding_name_one_abs[len(components_one):len(components_one)-plot_bars:-1]

    plot_comp_two_abs = components_two_abs[len(components_two):len(components_two)-plot_bars:-1]
    plot_name_two_abs = corresponding_name_two_abs[len(components_two):len(components_two)-plot_bars:-1]

    len_sum_one = len(components_one)-plot_bars+1
    sum_comp_one = sum(components_one[0:len_sum_one])
    #print(np.sqrt(np.sum(np.square(components_one))))
    if kwargs["plot_remaining"]:
        if kwargs["plot_remaining_reduced"]:
            plot_comp_one.append(sum_comp_one/len_sum_one)
        else:
            plot_comp_one.append(sum_comp_one)
        if kwargs["remaining_average"]:
            plot_name_one.append(kwargs["name_rest"] + " " + str(len_sum_one) + " avg.")
        else:
            plot_name_one.append(kwargs["name_rest"] + " " + str(len_sum_one))

    len_sum_two = len(components_two)-plot_bars+1
    sum_comp_two = sum(components_two[0:len_sum_two])
    if kwargs["plot_remaining"]:
        if kwargs["plot_remaining_reduced"]:
            plot_comp_two.append(sum_comp_two/len_sum_two)
        else:
            plot_comp_two.append(sum_comp_two)
        if kwargs["remaining_average"]:
            plot_name_two.append(kwargs["name_rest"] + " " + str(len_sum_two) + " avg.")

            plot_name_one_abs.append(kwargs["name_rest"] + " " + str(len_sum_one) + " avg.")
            plot_name_two_abs.append(kwargs["name_rest"] + " " + str(len_sum_two) + " avg.")
        else:
            plot_name_two.append(kwargs["name_rest"] + " " + str(len_sum_two))

            plot_name_one_abs.append(kwargs["name_rest"] + " " + str(len_sum_one))
            plot_name_two_abs.append(kwargs["name_rest"] + " " + str(len_sum_two))

    #print(len(plot_comp_one), len(components_one[0:len(components_one)-plot_bars]), len(components_one))

    if variant == 1 or variant == None:
        print("PCA feature #1:")
        #print("Average remaining:" + str(sum_comp_one/len_sum_one))
        fig, ax = plt.subplots(figsize=kwargs["figsize"])
        if kwargs["orient"] == "h":
            ax = sns.barplot(x=plot_comp_one, y=plot_name_one, palette=diverging_colors_1)
        else:
            ax = sns.barplot(y=plot_comp_one, x=plot_name_one, palette=diverging_colors_1)
            for item in ax.get_xticklabels():
                item.set_rotation(kwargs["rotation"])
        #ax.bar_label(ax.containers[0])
        if kwargs["add_text"]:
            if kwargs["remaining_average"]:
                bar_show_values(ax, orient=kwargs["orient"], space=kwargs["space"], decimal=kwargs["decimal"], last=sum_comp_one/len_sum_one)
            else:
                bar_show_values(ax, orient=kwargs["orient"], space=kwargs["space"], decimal=kwargs["decimal"])
        ax.set(xlabel=kwargs["x_label"], ylabel=kwargs["y_label"]+" 1")
        ax.grid(kwargs["grid"])
        ax.set_title(kwargs["title"])
        #ax.legend(kwargs["label"], loc=kwargs["loc"])
        plt.show()

        print("PCA feature #2:")
        #print("Average remaining:" + str(sum_comp_two/len_sum_two))
        fig, ax = plt.subplots(figsize=kwargs["figsize"])
        if kwargs["orient"] == "h":
            ax = sns.barplot(x=plot_comp_two, y=plot_name_two, palette=diverging_colors_2)
        else:
            ax = sns.barplot(y=plot_comp_two, x=plot_name_two, palette=diverging_colors_2)
            for item in ax.get_xticklabels():
                item.set_rotation(kwargs["rotation"])
        if kwargs["add_text"]:
            if kwargs["remaining_average"]:
                bar_show_values(ax, orient=kwargs["orient"], space=kwargs["space"], decimal=kwargs["decimal"], last=sum_comp_two/len_sum_two)
            else:
                bar_show_values(ax, orient=kwargs["orient"], space=kwargs["space"], decimal=kwargs["decimal"])
        ax.set(xlabel=kwargs["x_label"], ylabel=kwargs["y_label"]+" 2")
        ax.grid(kwargs["grid"])
        ax.set_title(kwargs["title"])
        #ax.legend(kwargs["label"], loc=kwargs["loc"])
        plt.show()

    if variant == 2 or variant == None:
        sum_comp_one_abs = sum(components_one_abs[0:len_sum_one])
        sum_comp_two_abs = sum(components_two_abs[0:len_sum_two])
        if kwargs["plot_remaining"]:
            if kwargs["plot_remaining_reduced"]:
                plot_comp_one_abs.append(sum_comp_one_abs/len_sum_one)
                plot_comp_two_abs.append(sum_comp_two_abs/len_sum_two)
            else:
                plot_comp_one_abs.append(sum_comp_one_abs)
                plot_comp_two_abs.append(sum_comp_two_abs)

        print("PCA feature #1:")
        #print("Average remaining:" + str(sum_comp_one_abs/len_sum_one))
        fig, ax = plt.subplots(figsize=kwargs["figsize"])
        if kwargs["orient"] == "h":
            ax = sns.barplot(x=plot_comp_one_abs, y=plot_name_one_abs, palette=diverging_colors_1)
        else:
            ax = sns.barplot(y=plot_comp_one_abs, x=plot_name_one_abs, palette=diverging_colors_1)
            for item in ax.get_xticklabels():
                item.set_rotation(kwargs["rotation"])
        if kwargs["add_text"]:
            if kwargs["remaining_average"]:
                bar_show_values(ax, orient=kwargs["orient"], space=kwargs["space"], decimal=kwargs["decimal"], last=sum_comp_one_abs/len_sum_one)
            else:
                bar_show_values(ax, orient=kwargs["orient"], space=kwargs["space"], decimal=kwargs["decimal"])
        ax.set(xlabel=kwargs["x_label"], ylabel=kwargs["y_label"]+" 1")
        ax.grid(kwargs["grid"])
        ax.set_title(kwargs["title"])
        #ax.legend(kwargs["label"], loc=kwargs["loc"])
        plt.show()

        print("PCA feature #2:")
        #print("Average remaining:" + str(sum_comp_two_abs/len_sum_two))
        fig, ax = plt.subplots(figsize=kwargs["figsize"])
        if kwargs["orient"] == "h":
            ax = sns.barplot(x=plot_comp_two_abs, y=plot_name_two_abs, palette=diverging_colors_2)
        else:
            ax = sns.barplot(y=plot_comp_two_abs, x=plot_name_two_abs, palette=diverging_colors_2)
            for item in ax.get_xticklabels():
                item.set_rotation(kwargs["rotation"])
        if kwargs["add_text"]:
            if kwargs["remaining_average"]:
                bar_show_values(ax, orient=kwargs["orient"], space=kwargs["space"], decimal=kwargs["decimal"], last=sum_comp_two_abs/len_sum_two)
            else:
                bar_show_values(ax, orient=kwargs["orient"], space=kwargs["space"], decimal=kwargs["decimal"])
        ax.set(xlabel=kwargs["x_label"], ylabel=kwargs["y_label"]+" 2")
        ax.grid(kwargs["grid"])
        ax.set_title(kwargs["title"])
        #ax.legend(kwargs["label"], loc=kwargs["loc"])
        plt.show()
        #print(plot_comp_one_abs[-1])
        if kwargs["plot_remaining"]:
            del plot_comp_one_abs[-1]
            del plot_comp_two_abs[-1]
    
    if variant == 3 or variant == None:
        sum_comp_one_abs = sum([abs(value) for value in components_one_abs[0:len_sum_one]])
        sum_comp_two_abs = sum([abs(value) for value in components_two_abs[0:len_sum_two]])
        if kwargs["plot_remaining"]:
            if kwargs["plot_remaining_reduced"]:
                plot_comp_one_abs.append(sum_comp_one_abs/len_sum_one)
                plot_comp_two_abs.append(sum_comp_two_abs/len_sum_two)
            else:
                plot_comp_one_abs.append(sum_comp_one_abs)
                plot_comp_two_abs.append(sum_comp_two_abs)
        plot_comp_one_this = [abs(value) for value in plot_comp_one_abs]
        plot_comp_two_this = [abs(value) for value in plot_comp_two_abs]

        print("PCA feature #1:")
        #print("absolute average remaining:" + str(sum_comp_one_abs/len_sum_one))
        fig, ax = plt.subplots(figsize=kwargs["figsize"])
        #print(len(plot_comp_one_this), len(plot_name_one_abs))
        if kwargs["orient"] == "h":
            ax = sns.barplot(x=plot_comp_one_this, y=plot_name_one_abs, palette=diverging_colors_1)
            ax.set(xlabel="absolute " + kwargs["x_label"], ylabel=kwargs["y_label"]+" 1")
        else:
            ax = sns.barplot(y=plot_comp_one_this, x=plot_name_one_abs, palette=diverging_colors_1)
            ax.set(xlabel=kwargs["x_label"], ylabel=kwargs["y_label"]+" 1")
            for item in ax.get_xticklabels():
                item.set_rotation(kwargs["rotation"])
        if kwargs["add_text"]:
            if kwargs["remaining_average"]:
                bar_show_values(ax, orient=kwargs["orient"], space=kwargs["space"], decimal=kwargs["decimal"], last=sum_comp_one_abs/len_sum_one)
            else:
                bar_show_values(ax, orient=kwargs["orient"], space=kwargs["space"], decimal=kwargs["decimal"])
        ax.grid(kwargs["grid"])
        ax.set_title(kwargs["title"])
        #ax.legend(kwargs["label"], loc=kwargs["loc"])
        plt.show()

        print("PCA feature #2:")
        #print("absolute average remaining:" + str(sum_comp_two_abs/len_sum_two))
        fig, ax = plt.subplots(figsize=kwargs["figsize"])
        if kwargs["orient"] == "h":
            ax = sns.barplot(x=plot_comp_two_this, y=plot_name_two_abs, palette=diverging_colors_2)
            ax.set(xlabel="absolute " + kwargs["x_label"], ylabel=kwargs["y_label"]+" 2")
        else:
            ax = sns.barplot(y=plot_comp_two_this, x=plot_name_two_abs, palette=diverging_colors_2)
            ax.set(xlabel=kwargs["x_label"], ylabel=kwargs["y_label"]+" 2")
            for item in ax.get_xticklabels():
                item.set_rotation(kwargs["rotation"])
        if kwargs["add_text"]:
            if kwargs["remaining_average"]:
                bar_show_values(ax, orient=kwargs["orient"], space=kwargs["space"], decimal=kwargs["decimal"], last=sum_comp_two_abs/len_sum_two)
            else:
                bar_show_values(ax, orient=kwargs["orient"], space=kwargs["space"], decimal=kwargs["decimal"])
        ax.grid(kwargs["grid"])
        ax.set_title(kwargs["title"])
        #ax.legend(kwargs["label"], loc=kwargs["loc"])
        plt.show()


def plot_distribution(
    pickle_name = "distribution/dist_last_pickle.pkl", 
    remove_text=True,
    move_text_x=0,
    move_text_y=0, 
    **kwargs
    ):
    """
    MAIN FUNCTION: Plotting function to: 'cT_distribution'

    DESCRIPTION:
    Plots histogram of temperatures.

    ARGUMENTS:
    - pickle_name: location from which the data will be collected
    - remove_text: If 'True':   removes the text which explains what range of temperature
                                values are considered in the distribution
    - move_text_x:    value that lets you move the text along x axis
    - move_text_y:    value that lets you move the text along y axis

    NEW KEYWORD-ARGUMENTS:
    - manual_limit: If 'True': applies manual limit to axis by settings

    KNOWN KEYWORD-ARGUMENTS: (see seaborn and matplotlib documentation for explanation)
    - title
    - x_label
    - y_label
    - figsize
    - label
    - y_scale
    - ylim
    - xlim
    - grid

    - edgecolor
    - kde
    - binwidth
    - loc
    - common_bins
    
    -bw_adjust                  
    - alpha             
    - multiple
    """

    kwargs.setdefault("title", "")
    kwargs.setdefault("x_label", "$T_{c}$ (in K)")
    kwargs.setdefault("y_label", "count")
    kwargs.setdefault("label", ["remaining", "Fe & Se", "Fe & As", "Cu & O"])
    kwargs.setdefault("edgecolor", "black")
    kwargs.setdefault("kde", True)
    kwargs.setdefault("bw_adjust", 2)
    kwargs.setdefault("binwidth", 0.25)
    kwargs.setdefault("common_bins", True)
    kwargs.setdefault("multiple", "layer")
    kwargs.setdefault("alpha", 0.5)
    kwargs.setdefault("loc", "best")
    kwargs.setdefault("figsize", [3.4, 2.7])
    kwargs.setdefault("y_scale", "linear")
    kwargs.setdefault("ylim", [0.5, 10**4])
    kwargs.setdefault("xlim", [0, 60])
    kwargs.setdefault("manual_limit", False)
    kwargs.setdefault("grid", True)

    sns_args = dict(
        kde=kwargs["kde"],
        binwidth=kwargs["binwidth"],
        edgecolor=kwargs["edgecolor"],
        label=kwargs["label"],
        common_bins=kwargs["common_bins"],
        multiple=kwargs["multiple"],
        alpha=kwargs["alpha"]
    )

    with open(PICKLE_DIRECTORY_NAME + pickle_name, "rb") as fid:
        pickle_data_loaded = pickle.load(fid)
    cT = pickle_data_loaded["cT"]
    min_cT = pickle_data_loaded["min"]

    fig, ax = plt.subplots(figsize=kwargs["figsize"])
    ax = sns.histplot(cT, **sns_args)
    ax.set(xlabel=kwargs["x_label"], ylabel=kwargs["y_label"])
    ax.set_yscale(kwargs["y_scale"])
    if kwargs["manual_limit"]:
        plt.ylim(kwargs["ylim"])
        plt.xlim(kwargs["xlim"])
    ax.grid(kwargs["grid"])
    ax.legend(kwargs["label"], loc=kwargs["loc"])
    if not remove_text:
        ax.text(
            0.7 + move_text_x,
            0.9 + move_text_y,
            "$T_{c}$ > " + str(min_cT) + " K",
            fontsize=10,
            bbox=dict(facecolor="black", alpha=0.15),
            transform=ax.transAxes,
        )
    ax.set_title(kwargs["title"])
    plt.show()


def plot_feat_importance(
    pickle_name="ML_feat_importance/mlfi_last_pickle.pkl", 
    plot_bars = 7, 
    variant = 3, 
    **kwargs
    ):
    """
    MAIN FUNCTION: Plotting function to 'vpc'

    DESCRIPTION:
    Plots for some features the magnitude of importance on the two principal components
    as bar plot.

    ARGUMENTS:
    - pickle_name: location from which the data will be collected
    - plot_bars: number of features to be displayed as bar
    - variant: way of visualizing data:
        - variant = 1:  bars are plotted in order of highest feature value
        - variant = 2:  bars are plotted in order of highest absolute feature value
        - variant = 3:  bars are plotted in order like 'variant = 2'. The difference is
                        that the absolute values are plotted instead of the values itself
                        (like in 'variant = 1' and 'variant = 2')
        - variant = None: All available variants are plotted

    NEW KEYWORD-ARGUMENTS:
    - plot_remaining:   If 'True': remaining features are combined and plotted
                        If 'False': just the first "plot_bars" features are plotted
    - plot_remaining_reduced: if 'True': average of the remaining features is plotted
    - name_rest: label name of features that will be shown combined
    - rotation: rotation of feature labels
    - add_text: If 'True': add values of bars on top of them
        - decimal: number of decimal units displayed on top of bars
        - space: distance of values from bars
        - remaining_average: If 'True': value of last bar with remaining features is 
                                        average of them
    - grid: If 'True': plot grid
    - colormap: This time a vector with two colormap definitions to distinguish both
                bar plots

    KNOWN KEYWORD-ARGUMENTS: (see seaborn and matplotlib documentation for explanation)
    - orient
    - title
    - x_label
    - y_label
    - figsize
    - edgecolor
    - grid
    """
    kwargs.setdefault("colormap", "Blues_d") #flag
    diverging_colors_1 = sns.color_palette(kwargs["colormap"], plot_bars) #"Blues_d"
    kwargs.setdefault("orient", "v")
    kwargs.setdefault("title", "")
    if kwargs["orient"] == "v":
        kwargs.setdefault("x_label", "feature")
        kwargs.setdefault("y_label", "feature importance")
    else:
        kwargs.setdefault("x_label", "feature importance")
        kwargs.setdefault("y_label", "feature")
    kwargs.setdefault("name_rest", "other")
    kwargs.setdefault("figsize", [3.4, 2.7])
    kwargs.setdefault("rotation", 45)
    kwargs.setdefault("add_text", True)
    kwargs.setdefault("decimal", 3)
    kwargs.setdefault("space", 0.005)
    kwargs.setdefault("remaining_average", True)
    kwargs.setdefault("grid", False)
    kwargs.setdefault("plot_remaining", True)
    kwargs.setdefault("plot_remaining_reduced", True)
    kwargs.setdefault("edgecolor", "black")

    if not kwargs["plot_remaining"]:
        kwargs["remaining_average"] = False
        plot_bars+=1

    with open(PICKLE_DIRECTORY_NAME + pickle_name, "rb") as fid:
        pickle_data_loaded = pickle.load(fid)
    
    ordered_imp = pickle_data_loaded["feat_imp"]
    ordered_std = pickle_data_loaded["std_feat"]
    ordered_names = pickle_data_loaded["names"]
    print("Average standard deviation: " + str(sum(ordered_std)/len(ordered_std)))

    plot_imp = ordered_imp[len(ordered_imp):len(ordered_imp)-plot_bars:-1]
    plot_name = ordered_names[len(ordered_imp):len(ordered_imp)-plot_bars:-1]

    len_sum_one = len(ordered_imp)-plot_bars+1
    sum_comp_one = sum(ordered_imp[0:len_sum_one])
    #print(np.sqrt(np.sum(np.square(components_one))))
    if kwargs["plot_remaining"]:
        if kwargs["plot_remaining_reduced"]:
            plot_imp.append(sum_comp_one/len_sum_one)
            plot_name.append(kwargs["name_rest"] + " " + str(len_sum_one) + " avg.")
        else:
            plot_imp.append(sum_comp_one)
            plot_name.append(kwargs["name_rest"] + " " + str(len_sum_one))

    fig, ax = plt.subplots(figsize=kwargs["figsize"])
    if kwargs["orient"] == "h":
        ax = sns.barplot(x=plot_imp, y=plot_name, palette=diverging_colors_1)
    else:
        ax = sns.barplot(y=plot_imp, x=plot_name, palette=diverging_colors_1)
        for item in ax.get_xticklabels():
            item.set_rotation(kwargs["rotation"])
    #ax.bar_label(ax.containers[0])
    if kwargs["add_text"]:
        if kwargs["remaining_average"]:
            bar_show_values(ax, orient=kwargs["orient"], space=kwargs["space"], decimal=kwargs["decimal"], last=sum_comp_one/len_sum_one)
        else:
            bar_show_values(ax, orient=kwargs["orient"], space=kwargs["space"], decimal=kwargs["decimal"])
    ax.set(xlabel=kwargs["x_label"], ylabel=kwargs["y_label"])
    ax.grid(kwargs["grid"])
    ax.set_title(kwargs["title"])
    #ax.legend(kwargs["label"], loc=kwargs["loc"])
    plt.show()


def plot_continuous_hp(
    pickle_name="hp_visualize/hpv_last_pickle.pkl",
    plot_computation_time = False,
    manual_limit = False,
    x_names = [],
    **kwargs
    ):
    """
    MAIN FUNCTION: Plotting function to 'continuous_hp_evaluation'

    DESCRIPTION:
    Plots errorbar plot of accuracy over one or multiple hyperparameter.

    ARGUMENTS:
    - pickle_name: location from which the data will be collected
    - plot_computation_time: If 'True': also plot computation time

    NEW KEYWORD-ARGUMENTS:
    - time_label: label for computation time if included
    - time_y_label: y_axis label for computation time if included  

    KNOWN KEYWORD-ARGUMENTS: (see seaborn and matplotlib documentation for explanation)
    - title
    - x_label
    - y_label
    - label

    - figsize
    - linewidth
    - linestyle
    - marker
    - markersize
    
    - elinewidth
    - ecolor
    - capthick
    - capsize
    """
    #apply default settings
    kwargs.setdefault("title", "")
    kwargs.setdefault("x_label", "default67")
    kwargs.setdefault("y_label", "mean absolute error")
    kwargs.setdefault("label", "accuracy")
    kwargs.setdefault("time_label", "computation time")
    kwargs.setdefault("time_y_label", "time (in s)")
    kwargs.setdefault("figsize", [3.4, 2.7])
    kwargs.setdefault("linewidth", 2)
    kwargs.setdefault("linestyle", "--")
    kwargs.setdefault("marker", "o")
    kwargs.setdefault("markersize", 6)
    kwargs.setdefault("capsize", 2.5)
    kwargs.setdefault("capthick", None)
    kwargs.setdefault("ecolor", None)
    kwargs.setdefault("elinewidth", None)
    kwargs.setdefault("ylim", [0.47,1.3])
    #markerfacecolor, markeredgecolor, color

    mpl_keywords = dict(
        linewidth=kwargs["linewidth"],
        linestyle=kwargs["linestyle"],
        marker=kwargs["marker"],
        markersize=kwargs["markersize"],
        label=kwargs["label"],
        capsize=kwargs["capsize"],
        capthick=kwargs["capthick"],
        ecolor=kwargs["ecolor"],
        elinewidth=kwargs["elinewidth"]
    )

    #import data:
    with open(PICKLE_DIRECTORY_NAME + pickle_name, "rb") as fid:
        data_loaded = pickle.load(fid)
    hyperparameters = data_loaded["hyper_para"]
    hyper_values = data_loaded["hyper_val"]
    acc_params = data_loaded["acc"]
    time_params = data_loaded["time"]
    std_params = data_loaded["standard_deviation"]
    var_params = data_loaded["variance"]

    #plot
    if plot_computation_time:
        for i in range(0, len(hyperparameters)):
            fig, ax = plt.subplots(figsize=kwargs["figsize"])
            ax.set_title(kwargs["title"])
            lns1 = ax.errorbar(
                hyper_values[i], 
                acc_params[hyperparameters[i]], 
                yerr=std_params[hyperparameters[i]],
                **mpl_keywords
                )
            ax2 = ax.twinx()
            lns2 = ax2.plot(
                hyper_values[i],
                time_params[hyperparameters[i]],
                linewidth=kwargs["linewidth"],
                linestyle=kwargs["linestyle"],
                marker=kwargs["marker"],
                label=kwargs["time_label"]
            )

            if kwargs["x_label"] == "default67":
                ax.set_xlabel('"' + hyperparameters[i] + '"')
            else:
                ax.set_xlabel(kwargs["x_label"])
            ax.set_ylabel(kwargs["y_label"])

            ax2.set_ylabel(kwargs["time_y_label"])
            lns = lns1 + lns2
            labs = [l.get_label() for l in lns]
            ax.legend(lns, labs, loc="best")
    else:
        for i in range(0, len(hyperparameters)):
            fig, ax = plt.subplots(figsize=kwargs["figsize"])
            ax.set_title(kwargs["title"])
            ax.errorbar(
                hyper_values[i], 
                acc_params[hyperparameters[i]], 
                yerr=std_params[hyperparameters[i]],
                **mpl_keywords
                )

            if len(x_names) == 0:
                if kwargs["x_label"] == "default67":
                    ax.set_xlabel('"' + hyperparameters[i] + '"')
                else:
                    ax.set_xlabel(kwargs["x_label"])
            else:
                ax.set_xlabel(x_names[i])
            ax.set_ylabel(kwargs["y_label"])

            ax.set_title(kwargs["title"])
            if manual_limit:
                plt.ylim(kwargs["ylim"])
            plt.show()


def plot_accuracy_training_size(
    pickle_names=["accuracy_size/as_last_pickle.pkl"], 
    manual_limit=False,
    **kwargs
    ):
    """
    MAIN FUNCTION: Plotting function to 'accuracy_training_size'

    DESCRIPTION:
    Plots errorbar plot for accuracy depending on size of training dataset for one
    or multiple datasets

    ARGUMENTS:
    - pickle_names: list of path locations to stored data

    NEW KEYWORD-ARGUMENTS:
    - label: this time a list of labels for different datasets
    - legend_title: title for legend

    KNOWN KEYWORD-ARGUMENTS: (see seaborn and matplotlib documentation for explanation)
    - title
    - x_label
    - y_label
    - figsize
    - xlim
    - ylim

    - marker
    - markersize
    - linestyle
    - linewidth
    
    - elinewidth
    - ecolor
    - capthick
    - capsize
    """
    #apply default settings:
    kwargs.setdefault("title", "")
    kwargs.setdefault("x_label", "training set size")
    kwargs.setdefault("y_label", "mean absolute error")
    kwargs.setdefault("figsize", [3.4, 2.7])
    kwargs.setdefault("linewidth", 2)
    kwargs.setdefault("linestyle", "--")
    kwargs.setdefault("marker", "o")
    kwargs.setdefault("markersize", 6)
    kwargs.setdefault("capsize", 2.5)
    kwargs.setdefault("capthick", None)
    kwargs.setdefault("ecolor", None)
    kwargs.setdefault("elinewidth", None)
    kwargs.setdefault("label", [])
    kwargs.setdefault("legend_title", "")
    kwargs.setdefault("xlim", [0,10000])
    kwargs.setdefault("ylim", [0.5,0.9])
    #markerfacecolor, markeredgecolor, color

    mpl_keywords = dict(
        linewidth=kwargs["linewidth"],
        linestyle=kwargs["linestyle"],
        marker=kwargs["marker"],
        markersize=kwargs["markersize"],
        capsize=kwargs["capsize"],
        capthick=kwargs["capthick"],
        ecolor=kwargs["ecolor"],
        elinewidth=kwargs["elinewidth"]
    )

    #load data:
    fig, ax = plt.subplots(figsize=kwargs["figsize"])
    ax.set_title(kwargs["title"])
    for i in range(0, len(pickle_names)):
        with open(PICKLE_DIRECTORY_NAME + pickle_names[i], "rb") as fid:
            pickle_data_loaded = pickle.load(fid)
        training_size = pickle_data_loaded["training"]
        accuracies = pickle_data_loaded["acc"]
        std_errors = pickle_data_loaded["standard_deviation"]
        variances = pickle_data_loaded["variance"]

        if len(kwargs["label"]) == 0:
            ax.errorbar(
                training_size, 
                accuracies, 
                yerr=std_errors,
                **mpl_keywords
                        )
        else:
            ax.errorbar(
                training_size, 
                accuracies, 
                yerr=std_errors,
                label=kwargs["label"][i],
                **mpl_keywords
                        )
    if len(kwargs["label"]) != 0:
        ax.legend(loc='best', title=kwargs["legend_title"])
    ax.set_xlabel(kwargs["x_label"])
    ax.set_ylabel(kwargs["y_label"])
    ax.set_title(kwargs["title"])
    if manual_limit:
        plt.xlim(kwargs["xlim"])
        plt.ylim(kwargs["ylim"])
    
    plt.show()


def plot_predicted_actual(
    pickle_name="predicted_actual/pa_last_pickle.pkl",
    show_kde=False,
    relative_reduce=1.0,
    remove_zero = False,
    remove_above = 200,
    manual_limit = False,
    **kwargs
    ):
    """
    MAIN FUNCTION: Plotting function to 'predicted_actual_filter_compare'

    DESCRIPTION:
    Plots scatter plot of predicted vs actual values of temperature. Also adds error 
    tube around perfect line (predicted values = actual values) and prints number
    of datapoints inside to console.

    ARGUMENTS:
    - pickle_name: location from which the data will be collected
    - with_errorbars: if 'True': scatter points are plotted with error bars
    - relative_reduce:  (float between 0 and 1) determines how much of original data
                        is shown in the plot. See: 'randmoly_delete_from_list' for
                        the point of this
    - remove_zero:  remove datappoints with temperature=0 (not very professional so don't 
                    touch this)
    
    NEW KEYWORD-ARGUMENTS:
    - perfect_label: label for perfect line (see description above)
    - line_alpha: alpha for perfect line (see description above)
    - scatter_alpha: alpha for scatter points
    - add_tube: if 'True': add cool tube around perfect line (see description above)
    - tube_label: label for the tube
    - tube_height: distance of tube from perfect line in y and -y direction

    KNOWN KEYWORD-ARGUMENTS: (see seaborn and matplotlib documentation for explanation)
    - title
    - x_label
    - y_label
    - figsize
    - label

    - marker
    - markersize
    - linestyle
    - linewidth
    
    - elinewidth
    - ecolor
    - capthick
    - capsize
    """
    #apply default settings:
    kwargs.setdefault("title", "")
    kwargs.setdefault("x_label", "actual $T_c$ (in K)")
    kwargs.setdefault("y_label", "predicted $T_c$ (in K)")
    kwargs.setdefault("figsize", [3.4, 2.7])
    kwargs.setdefault("linewidth", 2)
    kwargs.setdefault("linestyle", "--")
    kwargs.setdefault("marker", "o")
    kwargs.setdefault("markersize", 6)
    kwargs.setdefault("capsize", 1.5)
    kwargs.setdefault("capthick", None)
    kwargs.setdefault("ecolor", None)
    kwargs.setdefault("elinewidth", None)
    kwargs.setdefault("line_alpha", 0.9)
    kwargs.setdefault("scatter_alpha", 0.9)
    kwargs.setdefault("label", "data")
    kwargs.setdefault("tube_label", "error tube: ")
    kwargs.setdefault("add_tube", True)
    kwargs.setdefault("perfect_label", "pred. $T_c=$ act. $T_c$")
    kwargs.setdefault("levels", [0.5, 0.7])
    kwargs.setdefault("xlim", [0,40])
    kwargs.setdefault("ylim", [0,40])
    kwargs.setdefault("fill", False)
    kwargs.setdefault("colormap", 'viridis_r')
    #facecolor, marker

    mpl_keywords = dict(
        marker=kwargs["marker"],
        markersize=kwargs["markersize"],
        capsize=kwargs["capsize"],
        capthick=kwargs["capthick"],
        ecolor=kwargs["ecolor"],
        elinewidth=kwargs["elinewidth"]
    )
    #load data:
    with open(PICKLE_DIRECTORY_NAME + pickle_name, "rb") as fid:
        pickle_data_loaded = pickle.load(fid)
    predicted_values = pickle_data_loaded["pred"]
    actual_values = pickle_data_loaded["actual"]
    std_errors = pickle_data_loaded["standard_deviation"]
    variances = pickle_data_loaded["variance"]

    deletion_counter = 0
    for i in range(0, len(actual_values)):
        if actual_values[i-deletion_counter] > remove_above:
            del actual_values[i-deletion_counter]
            del predicted_values[i-deletion_counter]
            deletion_counter += 1

    if remove_zero:
        del_counter = 0
        for i in range(0, len(actual_values)):
            if actual_values[i-del_counter] == 0:
                del actual_values[i-del_counter]
                del predicted_values[i-del_counter]
                del std_errors[i-del_counter]
                del variances[i-del_counter]
                del_counter += 1
    #remove items
    use_for_plot=int((1-relative_reduce)*len(predicted_values))
    for j in range(0, use_for_plot):
        rand = random.randint(0, len(predicted_values)-1)
        predicted_values.pop(rand)
        actual_values.pop(rand)
        std_errors.pop(rand)
    
    #add tube
    global_min = min(actual_values)
    global_max = max(actual_values)

    kwargs.setdefault("tube_height", 0.1*global_max)

    tube_val = kwargs["tube_height"]
    tube_distance = tube_val*np.sqrt(2)/2

    fig, ax = plt.subplots(figsize=kwargs["figsize"])

    line_x = np.linspace(global_min, global_max+0.01*global_max, 10)
    line_y = copy.deepcopy(line_x)

    # ax.set_title(this_title)
    inside_tube = 0
    outside_tube = 0
    #variant 1
    """
    for i in range(0, len(predicted_values)):
        if abs(np.sqrt(actual_values[i]**2+actual_values[i]**2)-np.sqrt(predicted_values[i]**2+actual_values[i]**2)) < tube_distance:
            inside_tube += 1
        else:
            outside_tube += 1
    """
    #variant 2
    for i in range(0, len(predicted_values)):
        if predicted_values[i] > actual_values[i]:
            if predicted_values[i] <= actual_values[i]+tube_val:
                inside_tube += 1
            else:
                outside_tube += 1
        else:
            if predicted_values[i] >= actual_values[i]-tube_val:
                inside_tube += 1
            else:
                outside_tube += 1
    if show_kde:
        data = dict()
        data["kde"]=actual_values
        data["bla"]=predicted_values
        # sns.kdeplot(
        #     actual_values,
        #     predicted_values,
        #     fill=kwargs["fill"],
        #     levels=kwargs["levels"],
        #     legend=True
        # )
        sns.kdeplot(
            data=data,
            x="kde",
            y="bla",
            fill=kwargs["fill"],
            levels=kwargs["levels"],
            legend=True
        )
        sns.kdeplot(
            data=data,
            x="kde",
            y="bla",
            fill=True,
            levels=kwargs["levels"],
            cmap = kwargs["colormap"]
        )
        if manual_limit:
            ax.plot([-10,-10],[-10,-10], color="black", label="kde")
    else:
        ax.scatter(
            actual_values,
            predicted_values,
            marker=kwargs["marker"],
            s=kwargs["markersize"],
            alpha=kwargs["scatter_alpha"],
            label=kwargs["label"],
            linewidth=0
        )
    ax.plot(
        line_x,
        line_y,
        linewidth=kwargs["linewidth"],
        linestyle=kwargs["linestyle"],
        color="red",
        alpha=kwargs["line_alpha"],
        label=kwargs["perfect_label"]
    )
    if kwargs["add_tube"]:
        ax.fill_between(line_x, line_y - tube_val, line_y + tube_val, alpha=0.2, color="red", label=kwargs["tube_label"] + str(round(tube_val,3)) + " K")
        cut_y_axes = True

    ax.legend(loc="best")
    if cut_y_axes:
        plt.ylim([global_min-0.02*global_max, global_max+0.02*global_max])
        plt.xlim([global_min-0.02*global_max, global_max+0.02*global_max])
    if manual_limit:
        plt.ylim(kwargs["ylim"])
        plt.xlim(kwargs["xlim"])
    ax.set_xlabel(kwargs["x_label"])
    ax.set_ylabel(kwargs["y_label"])
    ax.set_title(kwargs["title"])

    print("Inside tube: " + str(inside_tube))
    print("Outside tube: " + str(outside_tube))
    print("All datapoints: " + str(len(predicted_values)))
    percentage = round(inside_tube*100/len(predicted_values),2)
    print(str(percentage) + " % of datapoints are inside error bar")
    
    plt.show()


def plot_pred_act_kde(
    pickle_name="predicted_actual/pa_last_pickle.pkl",
    remove_above = 200,
    manual_limit = False,
    reverse = False,
    **kwargs
    ):
    """
    bla
    """
    kwargs.setdefault("title", "")
    kwargs.setdefault("x_label", "actual $T_c$ (in K)")
    kwargs.setdefault("y_label", "count")
    kwargs.setdefault("edgecolor", "black")
    kwargs.setdefault("kde", True)
    if not reverse:
        kwargs.setdefault("label", ["outside error tube", "inside error tube"])
    else:
        kwargs.setdefault("label", ["inside error tube", "outside error tube"])
    kwargs.setdefault("bw_adjust", 2)
    kwargs.setdefault("binwidth", 2)
    kwargs.setdefault("common_bins", True)
    kwargs.setdefault("multiple", "stack") #layer
    kwargs.setdefault("alpha", 0.3)
    kwargs.setdefault("loc", "best")
    kwargs.setdefault("figsize", [3.4, 2.7])
    kwargs.setdefault("y_scale", "log")
    kwargs.setdefault("grid", False)
    kwargs.setdefault("xlim", [0, 1.0])
    kwargs.setdefault("ylim", [0, 1.0])
    kwargs.setdefault("tube_height", 1.46)
    tube_val = kwargs["tube_height"]

    sns_args = dict(
        kde=kwargs["kde"],
        binwidth=kwargs["binwidth"],
        edgecolor=kwargs["edgecolor"],
        common_bins=kwargs["common_bins"],
        label=kwargs["label"],
        multiple=kwargs["multiple"],
        alpha=kwargs["alpha"]
    )

    with open(PICKLE_DIRECTORY_NAME + pickle_name, "rb") as fid:
        pickle_data_loaded = pickle.load(fid)
    predicted_values = pickle_data_loaded["pred"]
    actual_values = pickle_data_loaded["actual"]
    std_errors = pickle_data_loaded["standard_deviation"]
    variances = pickle_data_loaded["variance"]

    deletion_counter = 0
    for i in range(0, len(actual_values)):
        if actual_values[i-deletion_counter] > remove_above:
            del actual_values[i-deletion_counter]
            del predicted_values[i-deletion_counter]
            deletion_counter += 1
    
    inside_tube = []
    outside_tube = []
    bla = 0
    
    for i in range(0, len(predicted_values)):
        if predicted_values[i] > actual_values[i]:
            if predicted_values[i] <= actual_values[i]+tube_val:
                inside_tube.append(actual_values[i])
                bla += 1
            else:
                outside_tube.append(actual_values[i])
        else:
            if predicted_values[i] >= actual_values[i]-tube_val:
                inside_tube.append(actual_values[i])
                bla += 1
            else:
                outside_tube.append(actual_values[i])

    fig, ax = plt.subplots(figsize=kwargs["figsize"])
    if reverse:
        ax = sns.histplot([outside_tube, inside_tube], **sns_args)
    else:
        ax = sns.histplot([inside_tube, outside_tube], **sns_args)
    ax.set(xlabel=kwargs["x_label"], ylabel=kwargs["y_label"])
    ax.set_yscale(kwargs["y_scale"])
    if manual_limit:
        plt.ylim(kwargs["ylim"])
        plt.xlim(kwargs["xlim"])
    ax.grid(kwargs["grid"])
    ax.legend(kwargs["label"], loc="best")

    plt.show()
 

def plot_best_features_temperature(
    pickle_name="best_features/bf_last_pickle.pkl",
    mark_composition = ["MgB2"],
    mix_properties = False,
    mgb2 = False,
    **kwargs
    ):
    """
    """
    kwargs.setdefault("title", "")
    kwargs.setdefault("y_label", "$T_c$ (in K)")
    kwargs.setdefault("figsize", [3.4, 2.7])
    #kwargs.setdefault("linewidth", 2)
    #kwargs.setdefault("linestyle", "--")
    kwargs.setdefault("marker", "o")
    kwargs.setdefault("markersize", 6)
    #kwargs.setdefault("line_alpha", 0.9)
    kwargs.setdefault("alpha", 0.4)

    with open(PICKLE_DIRECTORY_NAME + pickle_name, "rb") as fid:
        pickle_data_loaded = pickle.load(fid)
    best_features = pickle_data_loaded["collect_features"]
    temperatures = pickle_data_loaded["temperature"]

    #the code in this condition disgusts myself but i was too lazy to come up with something nice
    if mgb2:
        copy_features = copy.deepcopy(best_features)
        del best_features
        best_features = dict()
        copy_temp = copy.deepcopy(temperatures)
        del temperatures
        temperatures = [[],[]]

        for key in copy_features:
            best_features[key] = [[],[]]
        
        for i in range(0, len(copy_temp)):
            try:
                ratio = copy_features["Mg"][i]/copy_features["B"][i]
            except:
                ratio = 0
            #if ratio < 0.6 and ratio > 0.4:
            if ratio > 0:
                for key in copy_features:
                    best_features[key][1].append(copy_features[key][i])
                temperatures[1].append(copy_temp[i])
            else:
                for key in copy_features:
                    best_features[key][0].append(copy_features[key][i])
                temperatures[0].append(copy_temp[i])
        
        del best_features["Mg"]
        del best_features["B"]
        marker = ["o", "D"]

        label = ["remaining", "MgB$_2$-related"]
        apply_legend = True
    else:
        for key in best_features:
            best_features[key] = [best_features[key]]
        temperatures = [temperatures]
        apply_legend = False
        label = ["MgB2-family", "remaining"]
        marker = [kwargs["marker"], kwargs["marker"]]

    scatter_keywords = dict(
        s=kwargs["markersize"],
        alpha=kwargs["alpha"]
    )

    for key in best_features:
        fig, ax = plt.subplots(figsize=kwargs["figsize"])
        for i in range(0, len(best_features[key])):
            ax.scatter(
                best_features[key][i],
                temperatures[i],
                label=label[i],
                marker=marker[i],
                edgecolors='none',
                **scatter_keywords)
    
        ax.set_xlabel(key)
        ax.set_ylabel(kwargs["y_label"])
        ax.set_title(kwargs["title"])
        ax.set_ylim([None, 45])
        if apply_legend:
            ax.legend(loc="best")
        plt.show()
    """
    if mix_properties:
        for key in best_features:
            for sec_key in best_features:
                if key != sec_key:
                    fig, ax = plt.subplots(figsize=kwargs["figsize"])
                    ax.scatter(
                        best_features[key],
                        best_features[sec_key],
                        **scatter_keywords)
    
                    ax.set_xlabel(key)
                    ax.set_ylabel(sec_key)
                    ax.set_title(kwargs["title"])
                    plt.show()
    """

def plot_best_features(
    pickle_name="best_features/bf_last_pickle.pkl",
    temperature_threshold = 0,
    variant = 0, 
    remove_text=False, 
    move_text_y=0,
    move_text_x=0, 
    append_zeros=True,
    manual_limit=False,
    **kwargs
    ):
    """
    MAIN FUNCTION: Plotting function to: 'best_feature_candidates'

    DESCRIPTION:
    Plots histogram of the values the collected features have above a certain temperature.
    Also prints to console what features had to be normalized and their minimum and
    maximum value.

    ARGUMENTS:
    - pickle_name: location from which the data will be collected
    - temperature_threshold:    only features that led to temperatures equal or higher 
                                than this value will be plotted
    - append_zeros: if 'True':  also show counts or probability from feature values that
                                equal zero. If you are looking at chemical elements only,
                                this means that you will see how often the element was
                                not inside a chemical composition that led to temperatures
                                defined by 'temperature_threshold'
    - variant: (0,1,2,None) determines y-axis of plot:
        - variant=0:    show probability of occuring regarding to total datapoints with
                        transition temperature above temperature threshold
        - variant=1:    show counts normalalized to 1
        - variant=2:    show counts
        - variant=None: show all variants
    - remove_text: If 'True':   removes the text which states the temperature threshold
    - move_text_y:    value that lets you move the text along y axis
    - move_text_x:    value that lets you move the text along x axis
    - manual_limit: limit y to 0-1
    - mark_composition: 

    KNOWN KEYWORD-ARGUMENTS: (see seaborn and matplotlib documentation for explanation)
    - title
    - figsize
    - x_label
    - y_label
    - label
    - y_scale
    - ylim
    - xlim
    - grid

    - edgecolor
    - kde
    - binwidth
    - loc
    - common_bins
    - bw_adjust                  
    - alpha             
    - multiple
    """
    kwargs.setdefault("title", "")
    kwargs.setdefault("x_label", "relative number of atoms")
    kwargs.setdefault("y_label", "probability")
    kwargs.setdefault("edgecolor", "black")
    kwargs.setdefault("kde", True)
    kwargs.setdefault("bw_adjust", 2)
    kwargs.setdefault("binwidth", 0.05)
    kwargs.setdefault("common_bins", True)
    kwargs.setdefault("multiple", "layer")
    kwargs.setdefault("alpha", 0.5)
    kwargs.setdefault("loc", "best")
    kwargs.setdefault("figsize", [3.4, 2.7])
    kwargs.setdefault("y_scale", "linear")
    kwargs.setdefault("grid", False)
    kwargs.setdefault("xlim", [0, 1.0])
    kwargs.setdefault("ylim", [0, 1.0])

    with open(PICKLE_DIRECTORY_NAME + pickle_name, "rb") as fid:
        pickle_data_loaded = pickle.load(fid)
    best_features = pickle_data_loaded["collect_features"]
    temperatures = pickle_data_loaded["temperature"]

    deletion_counter = 0
    for i in range(0, len(temperatures)):
        if temperatures[i] < temperature_threshold:
            for key in best_features:
                del best_features[key][i-deletion_counter]
            deletion_counter += 1
    
    total_threshold = len(temperatures)-deletion_counter

    print_altered = False
    altered_names = []
    altered_min = []
    altered_max = []

    labels = []
    ratios = []
    for key in best_features:
        labels.append(key)
        if key in DiM.all_used_feature_names:
            print_altered = True
            altered_names.append(key)
            this_min = min(best_features[key])
            this_max = max(best_features[key])
            altered_min.append(this_min)
            altered_max.append(this_max)
            this_ratio = []
            for i in range(0, len(best_features[key])):
                this_ratio.append(best_features[key][i]/(this_max))
            ratios.append(this_ratio)
        else:
            ratios.append(best_features[key])
    if not append_zeros:
        for i in range(0, len(ratios)):
            deletion_counter = 0
            for j in range(0, len(ratios[i])):
                if ratios[i][j-deletion_counter] == 0:
                    del ratios[i][j-deletion_counter]
                    deletion_counter += 1
    
    if print_altered:
        print("Normalized features to fit range 0 to 1:")
        for i in range(0, len(altered_names)):
            print("   - " + altered_names[i] + ": min=" + str(altered_min[i]) + " ; max=" + str(altered_max[i]))

    sns_args = dict(
        kde=kwargs["kde"],
        binwidth=kwargs["binwidth"],
        edgecolor=kwargs["edgecolor"],
        common_bins=kwargs["common_bins"],
        multiple=kwargs["multiple"],
        alpha=kwargs["alpha"]
    )

    sns_hue = []
    sns_weights = []
    sns_values = []
    for i in range(0, len(ratios)):
        for j in range(0, len(ratios[i])):
            sns_values.append(ratios[i][j])
            sns_hue.append(labels[i])
            sns_weights.append(1/total_threshold)

    if variant == 0 or variant == None:
        data = {"x": sns_values, "weights": sns_weights, "features": sns_hue}
        fig, ax = plt.subplots(figsize=kwargs["figsize"])
        ax = sns.histplot(data=data, x="x", weights="weights", hue="features", **sns_args)
        ax.set(xlabel=kwargs["x_label"], ylabel=kwargs["y_label"])
        ax.set_yscale(kwargs["y_scale"])
        ax.grid(kwargs["grid"])
        if manual_limit:
            plt.ylim(kwargs["ylim"])
            plt.xlim(kwargs["xlim"])
        else:
            plt.xlim([0, 1.0])
        if not remove_text:
            ax.text(
                0.7 + move_text_x,
                0.9 + move_text_y,
                "$T_{c} \geq$" + str(temperature_threshold) + " K",
                fontsize=10,
                bbox=dict(facecolor="black", alpha=0.15),
                transform=ax.transAxes,
            )
        if variant == None:
            plt.show()

    if variant == 1 or variant == None:
        data = {"x": sns_values, "weights": sns_weights, "features": sns_hue}
        fig, ax = plt.subplots(figsize=kwargs["figsize"])
        ax = sns.histplot(data=data, x="x", hue="features", stat="probability", **sns_args)
        ax.set(xlabel=kwargs["x_label"], ylabel=kwargs["y_label"])
        ax.set_yscale(kwargs["y_scale"])
        ax.grid(kwargs["grid"])
        if manual_limit:
            plt.ylim(kwargs["ylim"])
            plt.xlim(kwargs["xlim"])
        else:
            plt.xlim([0, 1.0])
        if not remove_text:
            ax.text(
                0.7 + move_text_x,
                0.9 + move_text_y,
                "$T_{c} \geq$" + str(temperature_threshold) + " K",
                fontsize=10,
                bbox=dict(facecolor="black", alpha=0.15),
                transform=ax.transAxes,
            )
        if variant == None:
            plt.show()
    
    if variant == 2 or variant == None:
        data = {"x": sns_values, "weights": sns_weights, "features": sns_hue}
        fig, ax = plt.subplots(figsize=kwargs["figsize"])
        ax = sns.histplot(data=data, x="x", hue="features", **sns_args)
        kwargs["y_label"]="count"
        ax.set(xlabel=kwargs["x_label"], ylabel=kwargs["y_label"])
        ax.set_yscale(kwargs["y_scale"])
        ax.grid(kwargs["grid"])
        if manual_limit:
            plt.ylim(kwargs["ylim"])
            plt.xlim(kwargs["xlim"])
        else:
            plt.xlim([0, 1.0])
        if not remove_text:
            ax.text(
                0.7 + move_text_x,
                0.9 + move_text_y,
                "$T_{c} \geq$" + str(temperature_threshold) + " K",
                fontsize=10,
                bbox=dict(facecolor="black", alpha=0.15),
                transform=ax.transAxes,
            )
        if variant == None:
            plt.show()

    if append_zeros:
        for i in range(0, len(ratios)):
            deletion_counter = 0
            for j in range(0, len(ratios[i])):
                if ratios[i][j-deletion_counter] == 0:
                    del ratios[i][j-deletion_counter]
                    deletion_counter += 1
    print("Total datapoints above " + str(temperature_threshold) + " K: " + str(total_threshold))
    for i in range(0, len(ratios)):
        print(
            "Appearance of " + str(labels[i]) + ": " + str(len(ratios[i])) + " ("
            + str(round(len(ratios[i])/total_threshold, 3)) + ")"
            )
    #plt.show()
    return ax


def plot_predicted_distribution(
    name_or_directory="Predictions",
    kwargs_names = ["blank", "min", "sug"],
    temperature_threshold=0,
    remove_text=True,
    move_text=0,
    **kwargs
):
    """
    MAIN FUNCTION

    Plots Histogramm of critical temperatures

    ARGUMENTS:
    - name_or_directory:    Path to either a single file or directory.
    - kwargs_name:  If path leads to directory, data will be collected from files which
                    name contains 'kwargs_name'.
    - temperature_threshold:    only temperatures higher than this will be used in
                                distribution
    - revert_log:   if 'True':  if temperatures were calculated in log values they will
                                be transformed back to Kelvin
    
    ATTENTION:  revert_log requires hyperparameter settings to be stored in: 'hp_settings'
                directory within pickle data directory.

    - remove_text: If 'True':   removes the text which states the temperature threshold
    - move_text:    value that lets you move the text along x axis

    KNOWN KEYWORD-ARGUMENTS: (see seaborn and matplotlib documentation for explanation)
    - title
    - figsize
    - x_label
    - y_label
    - y_scale
    - ylim
    - grid

    - edgecolor
    - kde
    - binwidth
    - loc
    - common_bins
    - bw_adjust                  
    - alpha             
    - multiple
    """
    kwargs.setdefault("title", "")
    kwargs.setdefault("x_label", "$T_{c}$ (in K)")
    kwargs.setdefault("y_label", "count")
    kwargs.setdefault("y_scale", "linear")
    kwargs.setdefault("figsize", [3.4, 2.7])

    kwargs.setdefault("edgecolor", "black")
    kwargs.setdefault("kde", True)
    kwargs.setdefault("bw_adjust", 2)
    kwargs.setdefault("binwidth", 2)
    kwargs.setdefault("common_bins", True)
    kwargs.setdefault("multiple", "layer")
    kwargs.setdefault("alpha", 0.5)
    kwargs.setdefault("loc", "best")
    kwargs.setdefault("grid", True)
    kwargs.setdefault("ylim", [1,None])
    kwargs.setdefault("xlim", [0,None])
    kwargs.setdefault("legend", ["no properties", "property set 1", "property set 2"])

    sns_args = dict(
        kde=kwargs["kde"],
        binwidth=kwargs["binwidth"],
        edgecolor=kwargs["edgecolor"],
        common_bins=kwargs["common_bins"],
        multiple=kwargs["multiple"],
        alpha=kwargs["alpha"]
    )

    data_files = []
    temperatures = []
    for i in range(0, len(kwargs_names)):
        data_files.append([])
        temperatures.append([])
    if os.path.isdir(PICKLE_DIRECTORY_NAME + name_or_directory):
        all_data_files = os.listdir(PICKLE_DIRECTORY_NAME + name_or_directory)
        for i in range(0, len(all_data_files)):
            for j in range(0, len(kwargs_names)):
                if kwargs_names[j] in all_data_files[i]:
                    data_files[j].append(all_data_files[i])
        data_directory = name_or_directory + "/"
    else:
        data_files = [[name_or_directory]]
        data_directory = ""

    for i in range(0, len(data_files)):
        for j in range(0, len(data_files[i])):
            with open(PICKLE_DIRECTORY_NAME + data_directory + data_files[i][j], "rb") as fid:
                pickle_data_loaded = pickle.load(fid)
            this_temperatures = pickle_data_loaded["pred"]
            std_error = pickle_data_loaded["standard_deviation"]
            var_pred = pickle_data_loaded["variance"]
            for temps in this_temperatures:
                temperatures[i].append(temps)
            del this_temperatures
            del std_error
            del var_pred

    temps = []
    for j in range(0, len(temperatures)):
        thisi = 0
        deletion_counter = 0
        original_len = len(temperatures[j])
        for i in range(0, len(temperatures[j])):
            if temperatures[j][i] < temperature_threshold:
                thisi += 1
        temps.append(thisi)
        print(thisi*100/original_len)
                
        print(str(len(temperatures[j])) + " of " + str(original_len) + " datapoints remain after temperature filter.")
    print(temps)
    
    #plotting
    data = dict()
    data["x"] = []
    data["added properties"] = []
    for i in range(0, len(temperatures)):
        for j in range(0, len(temperatures[i])):
            data["x"].append(temperatures[i][j])
            data["added properties"].append(kwargs["legend"][i])
    del temperatures

    ax = sns.histplot(data, x="x", hue="added properties", **sns_args)
    ax.set(xlabel=kwargs["x_label"], ylabel=kwargs["y_label"])
    ax.grid(kwargs["grid"])
    if not remove_text:
        ax.text(
            0.7 + move_text,
            0.9,
            "$T_{c} \geq$" + str(temperature_threshold) + " K",
            fontsize=10,
            bbox=dict(facecolor="black", alpha=0.15),
            transform=ax.transAxes,
        )
    #ax.legend(kwargs["legend"], loc="best")
    #ax.legend(loc="best")
    ax.set_yscale(kwargs["y_scale"])
    plt.ylim(kwargs["ylim"])
    plt.xlim(kwargs["xlim"])
    plt.show()


def plot_regression_tree_behaviour(
    pickle_name="miscellaneous/reg_tree_last_pickle.pkl", 
    load_from_pickle=False,
    num_for_average = 10,
    **kwargs
    ):
    """
    MAIN FUNCTION

    DESCRIPTION:
    Only function in this document that calculates and plots data. Demonstrates Regression
    Tree and Random Forest behaviour on a made up one dimensional example of a simple
    sin curve with normal dostribution and some noise.

    ARGUMENTS:
    - pickle_name: location to where data will be stored or collected from
    - load_from_pickle: If 'True': load data from location, don't recalculate
    - num_for_average: How often values will be calculated for mean

    NEW KEYWORD-ARGUMENTS:
    - scatter_s: scatter size for sin curve
    - scatter_color: color for sin curve
    - scatter_edgecolor: scatter edgecolor for sin curve
    - reduced_linewidth:    linewidth of second Regression tree equals linewidth of first
                            regression tree - reduced_linewidth

    KNOWN KEYWORD-ARGUMENTS:
    - figsize
    - linewidth
    - linestyle
    - marker
    - markersize
    - alpha
    """
    #check if directory exists:
    if not os.path.isdir(PICKLE_DIRECTORY_NAME + mlm.get_directory(pickle_name)):
        os.mkdir(PICKLE_DIRECTORY_NAME + mlm.get_directory(pickle_name))

    kwargs.setdefault("figsize", [3.4, 2.7])
    kwargs.setdefault("linewidth", 2)
    kwargs.setdefault("linestyle", "--")
    kwargs.setdefault("marker", "o")
    kwargs.setdefault("markersize", 6)
    kwargs.setdefault("scatter_s", 20)
    kwargs.setdefault("scatter_color", "gray")
    kwargs.setdefault("scatter_edgecolor", "black")
    kwargs.setdefault("reduced_linewidth", 0)
    kwargs.setdefault("alpha", 0.7)

    if load_from_pickle:
        with open(PICKLE_DIRECTORY_NAME + pickle_name, "rb") as fid:
            data_loaded = pickle.load(fid)

        X = data_loaded["X"]
        y = data_loaded["y"]
        values_1_tree = data_loaded["values_1_tree"]
        values_2_tree = data_loaded["values_2_tree"]

        max_depth = data_loaded["depth"]
        train_tree = data_loaded["train_tree"]
        test_tree = data_loaded["test_tree"]

        estimators = data_loaded["estimators"]
        train_forest = data_loaded["train_forest"]
        test_forest = data_loaded["test_forest"]

        X_test = data_loaded["X_test"]
        large_depth = data_loaded["large_depth"]

    else:
        X_o = np.arange(0, 2*np.pi, 0.1)
        y = np.sin(X_o)
        y = y + np.random.normal(scale=0.01, size=len(y))
        X = []
        for i in X_o:
            X.append([i])
        X = np.array(X)
        

        theoretical_max_depth = np.log(len(y))/np.log(2)
        large_depth = int(theoretical_max_depth) + 7
        print("theoretical maximum depth of tree: " + str(theoretical_max_depth))

        #create noise
        for i in range(0, len(y)):
            if random.randrange(0, 4) == 1:
                if random.randrange(0, 2) == 1:
                    y[i] = y[i] + random.uniform(0.2,0.6)
                else:
                    y[i] = y[i] - random.uniform(0.2,0.6)

        # Predict
        X_test = np.arange(0.0, 2*np.pi, 0.01)[:, np.newaxis]

        values_1_tree = []
        values_2_tree = []

        max_depth = np.arange(1, large_depth)
        estimators = [1, 2, 3, 5, 7, 10, 13, 16, 20, 25]

        acc_train_tree = []
        acc_test_tree = []

        acc_train_forest = []
        acc_test_forest = []

        for i in range(0, num_for_average):
            # Fit regression model
            regr_1_tree = DecisionTreeRegressor(max_depth=2)
            regr_2_tree = DecisionTreeRegressor(max_depth=large_depth)

            regr_1_tree.fit(copy.deepcopy(X), copy.deepcopy(y))
            regr_2_tree.fit(copy.deepcopy(X), copy.deepcopy(y))

            #y_1 = regr_1.predict(X_test)
            #y_2 = regr_2.predict(X_test)

            values_1_tree.append(regr_1_tree.predict(X_test))
            values_2_tree.append(regr_2_tree.predict(X_test))
            """
            X_train, X_tst, y_train, y_tst = train_test_split(
                copy.deepcopy(X), 
                copy.deepcopy(y), 
                test_size=0.3
            )
            """
            X_train = copy.deepcopy(X)
            y_train = copy.deepcopy(y)
            X_tst = X_test
            y_tst = np.sin(X_test)

            this_train_forest = []
            this_test_forest = []

            for j in estimators:
                regr_forest = RandomForestRegressor(bootstrap=True, n_estimators=j)
                regr_forest.fit(X_train, y_train)

                this_train_forest.append(mean_absolute_error(y_train, regr_forest.predict(X_train)))
                this_test_forest.append(mean_absolute_error(y_tst, regr_forest.predict(X_tst)))
            
            acc_train_forest.append(this_train_forest)
            acc_test_forest.append(this_test_forest)

            this_train_tree = []
            this_test_tree = []

            for j in range(1, large_depth):
                regr = DecisionTreeRegressor(max_depth=j)
                regr.fit(X_train, y_train)

                this_train_tree.append(mean_absolute_error(y_train, regr.predict(X_train)))
                this_test_tree.append(mean_absolute_error(y_tst, regr.predict(X_tst)))
            
            acc_train_tree.append(this_train_tree)
            acc_test_tree.append(this_test_tree)

        values_1_tree = np.mean(np.array(values_1_tree), axis=0)
        values_2_tree = np.mean(np.array(values_2_tree), axis=0)

        train_tree = np.mean(np.array(acc_train_tree), axis=0)
        test_tree = np.mean(np.array(acc_test_tree), axis=0)

        train_forest = np.mean(np.array(acc_train_forest), axis=0)
        test_forest = np.mean(np.array(acc_test_forest), axis=0)

        data_for_fig = dict()
        data_for_fig["depth"] = max_depth
        data_for_fig["train_tree"] = train_tree
        data_for_fig["test_tree"] = test_tree

        data_for_fig["estimators"] = estimators
        data_for_fig["train_forest"] = train_forest
        data_for_fig["test_forest"] = test_forest

        data_for_fig["X"] = X
        data_for_fig["y"] = y
        data_for_fig["values_1_tree"] = values_1_tree
        data_for_fig["values_2_tree"] = values_2_tree

        data_for_fig["large_depth"] = large_depth
        data_for_fig["X_test"] = X_test

        with open(PICKLE_DIRECTORY_NAME + pickle_name, "wb") as fid:
            pickle.dump(data_for_fig, fid)
    
    print("Best accuracy achieved by forest: " + str(min(test_forest)))
    print("Best accuracy achieved by tree: " + str(min(test_tree)))

    plt.figure(figsize=kwargs["figsize"])

    #plt.plot(max_depth, acc_mean_train, color = 'steelblue', linewidth = 2, linestyle = '--', marker = 'o', markerfacecolor = 'dodgerblue', markeredgecolor = 'k', label="training data")
    #plt.plot(max_depth, acc_mean_test, color = 'orangered', linewidth = 2, linestyle = '--', marker = 'o', markerfacecolor = 'indianred', markeredgecolor = 'k', label = "test data")
    plt.plot(
        max_depth, 
        train_tree, 
        marker = kwargs["marker"], 
        markersize=kwargs["markersize"], 
        linewidth=kwargs["linewidth"], 
        linestyle=kwargs["linestyle"], 
        label="training data"
        )
    plt.plot(
        max_depth, 
        test_tree, 
        marker = kwargs["marker"], 
        markersize=kwargs["markersize"], 
        linewidth=kwargs["linewidth"], 
        linestyle=kwargs["linestyle"],
        label = "test data"
        )
    plt.xlabel("max_depth")
    plt.ylabel("mean absolute error")
    plt.legend()
    plt.show()

    plt.figure(figsize=kwargs["figsize"])
    plt.plot(
        estimators, 
        train_forest, 
        marker = kwargs["marker"], 
        markersize=kwargs["markersize"], 
        linewidth=kwargs["linewidth"], 
        linestyle=kwargs["linestyle"], 
        label="training data"
        )
    plt.plot(
        estimators, 
        test_forest, 
        marker = kwargs["marker"], 
        markersize=kwargs["markersize"], 
        linewidth=kwargs["linewidth"], 
        linestyle=kwargs["linestyle"], 
        label="test data"
        )
    plt.xlabel("n_estimators")
    plt.ylabel("mean absolute error")
    plt.legend()
    plt.show()

    plt.figure(figsize=kwargs["figsize"])
    plt.scatter(
        X, y, 
        label="training data", 
        s=kwargs["scatter_s"], 
        color=kwargs["scatter_color"],
        edgecolor=kwargs["scatter_edgecolor"]
        )
    plt.plot(
        X_test, 
        values_1_tree, 
        label="max_depth=2", 
        linewidth=kwargs["linewidth"], 
        alpha=kwargs["alpha"]
        )
    plt.plot(
        X_test, 
        values_2_tree, 
        label="max_depth="+str(large_depth), 
        linewidth=kwargs["linewidth"]-kwargs["reduced_linewidth"],
        alpha=kwargs["alpha"]
        )
    
    #plt.scatter(X, y, s=20, edgecolor="black", c="grey", label="real data")
    #plt.plot(X_test, y_1, color="orangered", label="max_depth=2", linewidth=1.7)
    #plt.plot(X_test, y_2, color="steelblue", label="max_depth=6", linewidth=1.7)

    #plt.plot(X_test, y_1, color="cornflowerblue", label="max_depth=2", linewidth=2)
    #plt.plot(X_test, y_2, color="yellowgreen", label="max_depth=5", linewidth=2)
    plt.xlabel("feature")
    plt.ylabel("response")
    plt.legend()
    plt.show()