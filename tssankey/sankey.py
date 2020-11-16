import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

# just using fill_between and offsetting curves vertically makes steep curves appear to get
# thinner in the middle, because you perceive the width of the line perpendicular to its direction
def cwcurve_poly(
    x_left, x_right, base_left, base_right, height, xy_ratio=1, steps=50, **kwargs
):
    # create the center of the curve
    # just define one, which we multiply as needed to get the right height
    # This is just the ys
    # This uses the smootherstep algorithm: https://en.wikipedia.org/wiki/Smoothstep#Variations
    center_x = np.linspace(-0.1, 1.1, steps)
    smoothstepxs = np.clip(center_x, 0, 1)
    # note that this is 0 when x = 0 and 1 when x = 1
    smoothstep = 6 * smoothstepxs ** 5 - 15 * smoothstepxs ** 4 + 10 * smoothstepxs ** 3
    center_x -= np.min(center_x)
    center_x /= np.max(center_x)
    center_x *= x_right - x_left
    center_x += x_left
    center_y = smoothstep * (base_right - base_left) + base_left + height / 2

    out_x = np.zeros(steps * 2)
    out_y = np.zeros(steps * 2)
    xdiff = (center_x[1:] - center_x[:-1]) / xy_ratio  # put in y units
    ydiff = center_y[1:] - center_y[:-1]
    ang = (
        np.arctan2(ydiff, xdiff) + np.pi / 2
    )  # rotate 90 degrees from direction of line
    # print(np.degrees(ang))
    for i, offset in enumerate([height / 2, -height / 2]):
        # print(i, offset)
        xoff = (
            np.array([*(np.cos(ang) * offset), 0]) * xy_ratio
        )  # handle last offset due to fencepost problem
        yoff = np.array([*(np.sin(ang) * offset), offset])
        edge_x = center_x + xoff
        edge_y = center_y + yoff
        if i == 1:
            # reverse this edge so polygon is wound
            edge_x = edge_x[::-1]
            edge_y = edge_y[::-1]

        out_x[i * steps : (i + 1) * steps] = edge_x
        out_y[i * steps : (i + 1) * steps] = edge_y

    xy = np.array([out_x, out_y]).T
    # return xy, center_x, center_y
    return matplotlib.patches.Polygon(xy, **kwargs)


# just using fill_between and offsetting curves vertically makes steep curves appear to get
# thinner in the middle, because you perceive the width of the line perpendicular to its direction
def cwcurve(x_left, x_right, base_left, base_right, height, ax, steps=50, **kwargs):
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
    percent_labels=False
):
    # make the sankey plot
    # this one is a little trickier because the bars are not the same in all of them
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
                        cwcurve(
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
