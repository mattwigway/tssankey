import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches


# just using fill_between and offsetting curves vertically makes steep curves appear to get
# thinner in the middle, because you perceive the width of the line perpendicular to its direction
def _cwcurve(x_left, x_right, base_left, base_right, height, ax, steps=50, **kwargs):
    # create the center of the curve
    # just define one, which we multiply as needed to get the right height
    # This is just the ys
    # This uses the smootherstep algorithm: https://en.wikipedia.org/wiki/Smoothstep#Variations
    center_x = np.linspace(-0.01, 1.01, steps)
    smoothstepxs = np.clip(center_x, 0, 1)
    # note that this is 0 when x = 0 and 1 when x = 1
    smoothstep = 6 * smoothstepxs ** 5 - 15 * smoothstepxs ** 4 + 10 * smoothstepxs ** 3
    center_x -= np.min(center_x)
    center_x /= np.max(center_x)
    center_x *= x_right - x_left
    center_x += x_left
    center_y = smoothstep * (base_right - base_left) + base_left + height / 2

    # compute width: https://stackoverflow.com/questions/19394505
    fig = ax.get_figure()
    # width is float in [0, 1] to say how much of bbox is used by axis. 72 is inches to points
    axis_length_pt = fig.bbox_inches.height * ax.get_position().height * 72
    axis_length_data = np.abs(ax.get_ylim()[1] - ax.get_ylim()[0])
    pts_per_data_unit = axis_length_pt / axis_length_data
    width = pts_per_data_unit * height

    xy = np.array([center_x, center_y]).T
    return matplotlib.patches.Polygon(
        xy, closed=False, facecolor="none", lw=width, **kwargs
    )


def tssankey(
    df,
    bar_width=0.4,
    figsize=(12, 8),
    total_gap=100,
    ax=None,
    weights=None,
    colors=None,
    curve_color=None,
    curve_alpha=0.25,
    percent_labels=True,
):
    """
    Create a Sankey plot. The only required parameter is the data frame which has categorical columns. These are grouped
    by and summed to create the bars.

    Parameters
    ----------

    df: pd.DataFrame 
        A dataframe with one categorical column per time period and one row per observation to show how individuals transition between the categories. Categories do not need to be the same in all columns.
    bar_width: float
        The width of the bars, with 1.0 indicating they touch and have no space for the Sankey lines between them. Default 0.4.
    figsize: tuple
        Size of the figure to plot, as a tuple (x, y). Default (12, 8).
    total_gap: float
        The total vertical gap between all categories in a period. Scale is number of observations. For instance, a value of 100 with five categories will mean that there will be a space equivalent to 25 observations between each category and the next. Default 100.
    ax: axes
        axes to plot on. Figsize ignored if specified. Default is to create new axes.
    weights: pd.Series
        weights for each observation, parallel to df. Default no weights.
    colors: dict
        Map from category names to colors to use for that category. Default is to use colors from the matplotlib style.
    curve_color: function
        Function that receives first category, left category, and right category for a curve, and returns a color. Default to use the colors of the first category, as specified by colors or in the style.
    percent_labels: bool
        If True, label each category with the percent of the total represented by that category.
    """

    if ax is None:
        f, ax = plt.subplots(figsize=figsize)

    bases = {}

    cols = df.columns
    for i, col in enumerate(cols):
        base = 0
        gap = total_gap / (len(df[col].cat.categories) - 1)
        bases[col] = dict()
        for val in df[col].cat.categories:
            if weights is not None:
                hgt = weights[df[col] == val].sum()
            else:
                hgt = (df[col] == val).sum()

            if colors is None:
                color = "C4" if i > 0 else None
            else:
                if val in colors:
                    color = colors[val]
                else:
                    color = "C4"

            rect = ax.bar(
                [i], [hgt], bottom=base, width=bar_width, color=color, zorder=10
            )[0]

            # label it
            if weights is not None:
                total = np.sum(weights)
            else:
                total = len(df)
            if percent_labels:
                label = f"{val}\n({int(round(hgt / total * 100))}%)"
            else:
                label = val
            if hgt / total < 0.05:
                label = val
            ax.annotate(
                label,
                xy=(
                    rect.get_x() + rect.get_width() / 2,
                    rect.get_y() + rect.get_height() / 2,
                ),
                xytext=(0, 0),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="center",
                color="white",
                weight="bold",
                zorder=20,
            )

            bases[col][val] = base
            base += hgt + gap

    # make the snakes
    i = 0
    for lcol, rcol in zip(cols[:-1], cols[1:]):
        # protective copy
        lbases = {k: v for k, v in bases[lcol].items()}
        rbases = {k: v for k, v in bases[rcol].items()}
        for orig_idx, orig_val in enumerate(df[cols[0]].cat.categories):
            for lval in df[lcol].cat.categories:
                for rval in df[rcol].cat.categories:
                    if curve_color is not None:
                        color = curve_color(orig_val, lval, rval)
                    elif colors is None or orig_val not in colors:
                        color = f"C{orig_idx}"
                    else:
                        color = colors[orig_val]

                    if weights is not None:
                        count = np.sum(
                            weights[
                                (df[cols[0]] == orig_val)
                                & (df[lcol] == lval)
                                & (df[rcol] == rval)
                            ]
                        )
                    else:
                        count = np.sum(
                            (df[cols[0]] == orig_val)
                            & (df[lcol] == lval)
                            & (df[rcol] == rval)
                        )
                    if count == 0:
                        continue

                    ax.add_patch(
                        _cwcurve(
                            i + bar_width / 2,
                            i + 1 - bar_width / 2,
                            lbases[lval],
                            rbases[rval],
                            count,
                            ax=ax,
                            edgecolor=color,
                            alpha=curve_alpha,
                        )
                    )

                    lbases[lval] += count
                    rbases[rval] += count
        i += 1

    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(cols)
    ax.set_yticks([])
