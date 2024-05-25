import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

# COLORS
## Discreet palette
BLUE = "#7fa1d2"
GREEN = "#4aa19e"
RED = "#c95b5a"
PURPLE = "#956a89"
ORANGE = "#e09049"
YELLOW = "#f8d451"


## Background color
BG_WHITE = "#fbf9f4"

## Light palette
LIGHT_BLUE = "#d2dcf1"
LIGHT_GREEN = "#d7e6bd"
LIGHT_RED = "#f5c3b1"
LIGHT_PURPLE = "#dccee5"
LIGHT_ORANGE = "#f9da97"
LIGHT_YELLOW = "#f9ed89"

# FUNCTIONS

def standardize_colnames(df):
    """
    Formats column names of a dataframe: removes capital letters and spaces

    """
    dict = {}
    for col_name in df.columns:
        key = col_name
        value = col_name.lower().replace(" ", "_")
        dict.update({key: value})
    df.rename(columns=dict, inplace=True)
    return df


def missing_tab(df):
    df_m = pd.DataFrame(df.isna().sum().sort_values(ascending=False)).reset_index()
    df_m = df_m.rename(columns={"index": "column_name", 0: "no_missing"})
    tot_obs = df.shape[0]
    df_m["percent_missing"] = round(df_m["no_missing"] / tot_obs * 100, 2)
    res = df_m[df_m.no_missing > 0]
    return res


def plot_na(df, myfont: str = "Bahnschrift"):
    na_df = missing_tab(df)

    fig, ax = plt.subplots(figsize=(30, 15))

    # General parameters for fiqure and axes
    fig.patch.set_facecolor(BG_WHITE)
    ax.set_facecolor(BG_WHITE)
    ## Title
    fig.suptitle(
        "Number of Missing Values in a Dataframe",
        fontfamily=myfont,
        fontweight="bold",
        fontsize=25,
    )

    # Prepare values
    labels = na_df.column_name.values
    vals = na_df.no_missing.values
    p_vals = na_df.percent_missing

    # Graph
    ax.hlines(y=labels, xmin=0, xmax=vals, linewidth=2, color=BLUE)
    ax.plot(vals, labels, "o", color=BLUE)
    ax.grid(axis="x", linestyle=":", linewidth=1)

    # Additional labels
    for c in range(len(labels)):
        ax.text(
            vals[c] * (1 + 0.01),
            labels[c],
            "{} ({}%)".format(vals[c], p_vals[c]),
            fontsize=15,
            fontname=myfont,
            va="center",
            zorder=10,  # to make sure the line is on top
        )

    # Axes visibility
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_lw(1.5)

    # Axes parameters
    ax.tick_params(axis="both", which="major", labelsize=15)
    for tick in ax.get_xticklabels():
        tick.set_fontname(myfont)
    for tick in ax.get_yticklabels():
        tick.set_fontname(myfont)

    return fig, ax


def get_count_percent_tab(df, colname: str):
    t = (
        df.groupby(colname)
        .agg(number_of_entries=(colname, "count"))
        .reset_index()
        .sort_values("number_of_entries")
    )
    t = t.assign(
        percent=round(t.number_of_entries / t.number_of_entries.sum() * 100, 2)
    )
    return t


def unq_cat_df(df, ctg_cols):
    """
    Returns dataframe with uniqe values of columns
    """
    cat_cols = df[ctg_cols]
    unique_values = {}
    for col in ctg_cols:
        unique_values[col] = df[col].unique()

    # Find the length of the longest unique list
    max_length = max(len(vals) for vals in unique_values.values())

    # Pad lists to equal lengths
    for col in unique_values:
        unique_values[col] = list(unique_values[col]) + [" "] * (
            max_length - len(unique_values[col])
        )

    # Create DataFrame from dictionary
    unique_df = pd.DataFrame(unique_values)

    return unique_df


def bplots_custom(
    df,
    cols,
    gtitlesize: int = 25,
    gticksize: int = 150,
    textlabsize: int = 15,
    gentitle: str = None,
    gentitlesize: int = 50,
    nrows: int = 1,
    ncols: int = 1,
    perclabsize: int = 150,
    myfont: str = "Bahnschrift",
):
    """
    By default creates boxplot for a num feature. Supports creating multiple boxplots on one figure
    """
    if isinstance(cols, str):
        cols = [cols]
    elif isinstance(cols, list):
        pass
    fig = plt.figure(figsize=(30, 25))
    fig.patch.set_facecolor(BG_WHITE)
    boxprops = dict(linewidth=0, color=LIGHT_BLUE, linestyle=None)
    medianprops = dict(linewidth=0)

    # Create subset of dataframe
    subdf = df[cols]

    for i in enumerate(subdf.columns):
        num_info = subdf.describe()
        num_info = num_info.iloc[1:, :]
        # Get values for labels
        median = {}
        median[i[1]] = round(num_info.loc["50%", i[1]], 5)

        q75 = {}
        q75[i[1]] = round(num_info.loc["75%", i[1]], 5)

        q25 = {}
        q25[i[1]] = round(num_info.loc["25%", i[1]], 5)

        ax = plt.subplot(nrows, ncols, i[0] + 1)
        ax.set_facecolor(BG_WHITE)
        sb.boxplot(
            y=i[1], data=subdf, boxprops=boxprops, medianprops=medianprops, color=RED
        )
        ax.grid(linestyle=":", linewidth=2, axis="y")

        ax.set_title(
            f"Boxplot for {i[1]}",
            fontsize=gtitlesize,
            fontfamily=myfont,
            fontweight="bold",
        )
        # Add median marker
        ax.scatter(i[1], q75[i[1]], s=perclabsize, color=RED, marker="d", zorder=3)
        vertical_offset = median[i[1]] * 0.05

        # Add q75 value label.
        ax.text(
            i[1],
            q75[i[1]] + q75[i[1]] * 0.05,
            f"Q75 = {q75[i[1]]}",
            horizontalalignment="left",
            size=textlabsize,
            color="#121212",
            weight="semibold",
            fontname=myfont,
        )
        if median[i[1]] != q75[i[1]]:
            # Add median marker
            ax.scatter(
                i[1], median[i[1]], s=perclabsize, color=GREEN, marker="d", zorder=3
            )
            ax.text(
                i[1],
                median[i[1]] + median[i[1]] * 0.02,
                f"M = {median[i[1]]}",
                horizontalalignment="center",
                size=textlabsize,
                color="#121212",
                weight="semibold",
                fontname=myfont,
            )
        if median[i[1]] != q25[i[1]]:
            # Add q25 marker
            ax.scatter(
                i[1], q25[i[1]], s=perclabsize, color=GREEN, marker="d", zorder=3
            )
            ax.text(
                i[1],
                q25[i[1]] - q25[i[1]] * 0.02,
                f"Q25 = {q25[i[1]]}",
                horizontalalignment="left",
                size=textlabsize,
                color="#121212",
                weight="semibold",
                fontname=myfont,
            )
    if gentitle != None:
        fig.suptitle(
            gentitle,
            fontname=myfont,
            fontsize=gentitlesize,
            weight="bold",
            ha="center",
        )

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    return fig, ax


# Create the plot
def plot_count_perc(
    df,
    cols,
    gtitlesize: int = 250,
    gticksize: int = 150,
    textlabsize: int = 125,
    gentitle: str = None,
    gentitlesize: int = 300,
    nrows: int = 1,
    ncols: int = 1,
    myfont: str = "Bahnschrift",
):
    fig = plt.figure(figsize=(200, 100))
    fig.patch.set_facecolor(BG_WHITE)

    if isinstance(cols, str):
        cols = [cols]
    elif isinstance(cols, list):
        pass

    for i in enumerate(cols):
        agg_df = get_count_percent_tab(df, i[1])
        values_abs = agg_df.number_of_entries.values
        values_p = agg_df.percent.values
        labels = agg_df[f"{i[1]}"].values

        ax = plt.subplot(nrows, ncols, i[0] + 1)
        ax.set_facecolor(BG_WHITE)
        ax.barh(labels, values_abs, zorder=2, color=BLUE)
        ax.set_title(
            f"Value counts ({i[1]})",
            fontsize=gtitlesize,
            fontfamily=myfont,
            fontweight="bold",
        )

        # Visual
        ax.grid(axis="x", color="#A8BAC4", linestyle=":", linewidth=10)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_lw(10)
        ax.tick_params(axis="both", which="major", labelsize=gticksize)
        ax.set_xlim(0, max(values_abs) * 1.1)

        # Tick font
        for tick in ax.get_xticklabels():
            tick.set_fontname(myfont)
        for tick in ax.get_yticklabels():
            tick.set_fontname(myfont)

        for i in range(len(labels)):
            ax.text(
                values_abs[i] * 1.02,
                labels[i],
                f"{values_abs[i]}   ({values_p[i]}%)",
                fontname=myfont,
                va="center",
                fontsize=125,
                zorder=10,  # to make sure the line is on top
            )
    if gentitle != None:
        fig.suptitle(
            gentitle,
            fontname=myfont,
            fontsize=gentitlesize,
            weight="bold",
            ha="center",
        )

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    return fig, ax


def optimize_memory(df):
    # Categorical data optimization
    numerics = ["int64", "int32", "int16", "float64", "float32", "float16"]
    for i in df.select_dtypes(exclude=numerics).columns:
        df[i] = df[i].astype("category")

    ## Numeric data optimization
    res = {}
    for i in df.select_dtypes(include=numerics).columns:
        col_type = df[i].dtype
        res.update({i: []})
        ind = numerics.index(f"{col_type}")
        col_min = min(df[i].dropna())
        col_max = max(df[i].dropna())
        if ind == (2 | 5):
            pass

        # Check int...
        elif ind in range(1):
            next_ind = ind + 1
            while next_ind < 3:
                next_min = np.iinfo(f"{numerics[next_ind]}").min
                next_max = np.iinfo(f"{numerics[next_ind]}").max
                if (next_min < col_min) | (next_max > col_max):
                    res[f"{i}"].append(numerics[next_ind])

                next_ind += 1
        # Check float...
        elif ind in range(3, 6, 1):
            next_ind = ind + 1
            while next_ind < 6:
                next_min = np.finfo(f"{numerics[next_ind]}").min
                next_max = np.finfo(f"{numerics[next_ind]}").max
                if (next_min < col_min) | (next_max > col_max):
                    res[f"{i}"].append(numerics[next_ind])

                next_ind += 1

    for key in res:
        ind = len(res[key]) - 1
        if ind < 0:
            continue
        else:
            df[key] = df[key].astype(f"{res[key][ind]}")
    return df


from scipy.interpolate import make_interp_spline


def parallel_coordinates_custom(
    df, title_text: str = None, myfont: str = "Bahnschrift"
):
    axes = df.columns
    cat_cord = {}
    num_min_max = {}
    numerics = ["int64", "int32", "int16", "float64", "float32", "float16", "float"]
    fig, ax = plt.subplots(figsize=(50, 15))
    fig.patch.set_facecolor(BG_WHITE)
    ax.set_facecolor(BG_WHITE)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="y", linestyle=":", linewidth=5)
    ax.get_yaxis().set_visible(False)
    # Add axes for each dimension
    ax.set_xticks(
        ticks=range(1, len(axes) + 1, 1), labels=axes.values, size=25, fontfamily=myfont
    )
    # Create additional axes
    for i in range(len(axes)):  # i - column order number of orig df
        ax.axvline(x=i + 1, linewidth=3, color="#121212", zorder=8)
        # Non-numeric columns
        if df[axes[i]].dtype not in numerics:
            # print(True)
            cat_list = df[axes[i]].unique().tolist()
            add = 1 / (len(cat_list) + 1)
            colcat_dict = {}
            for j in range(len(cat_list)):  # j - order number for cat within cat column
                yparam = add * (j + 1)
                colcat_dict[cat_list[j]] = yparam
                ## Plot cat variables on axes
                ax.scatter(i + 1, yparam, s=150, zorder=10, c="#121212")
                ## add categories to axes
                ax.text((i + 1) * 1.02, yparam, cat_list[j], size=20, fontfamily=myfont)
                cat_cord[df.columns[i]] = colcat_dict
                ax.scatter(i + 1, yparam, s=150, zorder=10, c="#121212")
                ## add categories to axes
                ax.text((i + 1) * 1.02, yparam, cat_list[j], size=20, fontfamily=myfont)
                yparam += yparam

        else:
            # Ticks for numerical values
            col_min, col_max = df[axes[i]].min(), df[axes[i]].max()
            num_tick_dict = set_num_ax_ticks(df[axes[i]].values, col_min, col_max)

            ## Plot ticks from dict
            for key in num_tick_dict:
                ax.scatter(i + 1, num_tick_dict[key], s=150, zorder=10, c="#121212")
                ax.text((i + 1) * 1.02, num_tick_dict[key], round(key), size=20)
    # Adding values
    transp_df = df.transpose().reset_index()
    transp_df = transp_df.drop(columns=transp_df.columns[0])

    for col in transp_df.columns:
        line = transp_df[col].tolist()
        xvals = []
        yvals = []
        # print(line)

        for ent in enumerate(line):
            xvals.append(ent[0] + 1)
            # If categorical data
            if df[df.columns[ent[0]]].dtype.name in ["category", "object"]:
                yvals.append(cat_cord[df.columns[ent[0]]][ent[1]])

            else:
                pass
                sc_y = norm_val(
                    ent[1],
                    min(df.iloc[:, ent[0]].values),
                    max(df.iloc[:, ent[0]].values),
                )

                yvals.append(sc_y)

        X_Y_Spline = make_interp_spline(np.array(xvals), np.array(yvals))
        X_ = np.linspace(np.array(xvals).min(), np.array(xvals).max(), 500)
        Y_ = X_Y_Spline(X_)
        ax.plot(X_, Y_, alpha=0.01, linewidth=10, c=BLUE)

        if title_text != None:
            fig.suptitle(title_text, fontfamily=myfont, size=50, fontweight="bold")
        else:
            pass

    return fig, ax


def mult_coord_plots(
    datasource,
    divcol: str,
    rowsno: int,
    colno: int,
    gentitle: str = None,
    linecolor: str = BLUE,
    myfont: str = "Bahnschrift",
):
    cl_list = datasource[divcol].unique()
    fig = plt.figure(figsize=(60, 25))
    fig.patch.set_facecolor(BG_WHITE)
    if gentitle != None:
        fig.suptitle(
            gentitle,
            fontname=myfont,
            fontsize=50,
            weight="bold",
            ha="center",
        )

    for cl in enumerate(cl_list):
        subdf = datasource.loc[datasource[divcol] == cl[1]]
        axes = subdf.columns
        df = subdf
        ax = plt.subplot(rowsno, colno, cl[0] + 1)
        ax.grid(axis="y", linestyle=":", linewidth=5)
        cat_cord = {}
        numerics = ["int64", "int32", "int16", "float64", "float32", "float16", "float"]

        # Create additional axes
        for i in range(len(axes)):  # i - column order number of orig df
            ax.axvline(x=i + 1, linewidth=3, color="#121212", zorder=8)
            ax.set_facecolor(BG_WHITE)
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["left"].set_visible(False)

            ax.get_yaxis().set_visible(False)
            # Add axes for each dimension
            ax.set_xticks(
                ticks=range(1, len(axes) + 1, 1),
                labels=axes.values,
                size=25,
                fontfamily=myfont,
            )
            # Non-numeric columns
            if df[axes[i]].dtype not in numerics:
                # print(True)
                cat_list = df[axes[i]].unique().tolist()
                add = 1 / (len(cat_list) + 1)
                colcat_dict = {}
                for j in range(
                    len(cat_list)
                ):  # j - order number for cat within cat column
                    yparam = add * (j + 1)
                    colcat_dict[cat_list[j]] = yparam
                    # Plot cat variables on axes
                    ax.scatter(i + 1, yparam, s=150, zorder=10, c="#121212")
                    # add categories to axes
                    ax.text(
                        (i + 1) * 1.02, yparam, cat_list[j], size=20, fontfamily=myfont
                    )
                    cat_cord[df.columns[i]] = colcat_dict
                    ax.scatter(i + 1, yparam, s=150, zorder=10, c="#121212")
                    # add categories to axes
                    ax.text(
                        (i + 1) * 1.02, yparam, cat_list[j], size=20, fontfamily=myfont
                    )
                    yparam += yparam

            else:
                # Ticks for numerical values
                col_min, col_max = df[axes[i]].min(), df[axes[i]].max()
                num_tick_dict = set_num_ax_ticks(df[axes[i]].values, col_min, col_max)
                for key in num_tick_dict:
                    ax.scatter(i + 1, num_tick_dict[key], s=150, zorder=10, c="#121212")
                    ax.text((i + 1) * 1.02, num_tick_dict[key], key, size=20)
        # Adding values
        transp_df = df.transpose().reset_index()
        transp_df = transp_df.drop(columns=transp_df.columns[0])

        for col in transp_df.columns:
            line = transp_df[col].tolist()
            xvals = []
            yvals = []
            # print(line)

            for ent in enumerate(line):
                xvals.append(ent[0] + 1)
                # If categorical data
                if df.columns[ent[0]] in cat_cord:
                    yvals.append(cat_cord[df.columns[ent[0]]][ent[1]])

                else:
                    pass
                    sc_y = norm_val(
                        ent[1],
                        min(df.iloc[:, ent[0]].values),
                        max(df.iloc[:, ent[0]].values),
                    )

                    yvals.append(sc_y)

            X_Y_Spline = make_interp_spline(np.array(xvals), np.array(yvals))

            X_ = np.linspace(np.array(xvals).min(), np.array(xvals).max(), 500)
            Y_ = X_Y_Spline(X_)
            ax.plot(X_, Y_, alpha=0.01, linewidth=10, c=linecolor)
            ax.set_title(
                f"Parallel Coordinates Plot for {divcol} =  {cl[1]}",
                fontsize=35,
            )

            fig.tight_layout()

    return fig, ax


def set_num_ax_ticks(colvals, mincolval, maxcolval) -> dict:
    num_dict_key = {}
    r = maxcolval - mincolval
    step = r / 4
    startpoint = mincolval
    for i in range(4):
        val = round(startpoint)
        num_dict_key[val] = norm_val(val=val, minv=mincolval, maxv=maxcolval)
        startpoint += step

    num_dict_key[mincolval] = norm_val(mincolval, mincolval, maxcolval)
    num_dict_key[maxcolval] = norm_val(maxcolval, mincolval, maxcolval)

    return num_dict_key


def norm_val(val, minv, maxv) -> int:
    """
    Normalises numeric value to scale from 0 to 1
    """
    res = (val - minv) / (maxv - minv)
    return res



