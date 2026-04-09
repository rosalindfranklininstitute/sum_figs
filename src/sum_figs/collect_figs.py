# SPDX-FileCopyrightText: 2026 RFI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, NamedTuple, Literal, cast
from dataclasses import dataclass
import logging
import builtins
import re

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import scipy
import numpy as np

from datargs import (
    arg_field,
    ArgType,
    ConfigFileArgs,
    NoInteractiveArgs,
)

from .interactive_total import show_interactively

from icecream import ic

logger = logging.getLogger(__name__)


@dataclass
class ProcessArgs(ConfigFileArgs, NoInteractiveArgs):
    in_path: Path = arg_field(
        "-d",
        "--directory",
        required=True,
        arg_type=ArgType.EXPLICIT_ONLY,
        doc="The input directory.",
        default=None,
    )
    out_path: Path = arg_field(
        "-o",
        "--output",
        required=True,
        arg_type=ArgType.EXPLICIT_ONLY,
        doc="The output directory.",
        default=None,
    )

    final_title: str = arg_field(
        "--title",
        doc="The title to use on the final image. This is also used to derive its filename.",
        default="Total Image",
    )

    colormap: str = arg_field(
        "--color-map",
        doc="The color map to use for the plotting. If not specified tries to find one in the fig file.",
        choices=[
            "viridis",
            "plasma",
            "inferno",
            "magma",
            "cividis",
            "grey",
            "gray",
            "auto",
        ],
        default="auto",
    )

    min_mass: float = arg_field(
        "--min",
        doc="The minimum value to scale to total image color too. If not included uses the default. If --independent-dolor-scale is not specified, then this applies to all figures.",
        default=None,
    )

    max_mass: float = arg_field(
        "--max",
        doc="The maximum value to scale to total image color too. If not included uses the default. If --independent-scale is not specified, then this applies to all figures.",
        default=None,
    )

    independent_scales: bool = arg_field(
        "--no-ind-scales",
        "--no-independent-scales",
        arg_type=ArgType.EXPLICIT_ONLY,
        doc="When specified the plots for the fig files will use the scale specified by --min-mass and --max-mass. If exlcuded the values from the fig file will be used.",
        action="store_false",
    )

    independent_colors: bool = arg_field(
        "--ind-colors",
        doc="When specified the plots for the fig files will use the colormap in their file. If excluded the --colormap will be used.",
        action="store_true",
    )

    origin: str = arg_field(
        doc="Specifies the origin of the imshow.",
        choices=["upper", "lower"],
        default="upper",
    )


class Child(NamedTuple):
    types: list[str]
    keys: list[str | int]
    data: np.ndarray


class Plot(NamedTuple):
    child: Child
    path: Path
    colormap: ListedColormap
    clim: np.ndarray
    xlim: np.ndarray
    ylim: np.ndarray
    title: str
    xlabel: str
    ylabel: str

    @staticmethod
    def from_child(mat, child: Child, path: Path) -> "Plot":
        keys = child.keys[:-2]
        colormap = ListedColormap(get(mat, *keys, "properties", "Colormap"))
        clim = get(mat, *keys, "properties", "CLim")
        xlim = get(mat, *keys, "properties", "XLim")
        ylim = get(mat, *keys, "properties", "YLim")
        xlabel = get(mat, *keys, "children", 1, "properties", "String")
        ylabel = get(mat, *keys, "children", 2, "properties", "String")
        title = get(mat, *keys, "children", 3, "properties", "String")
        return Plot(
            child=child,
            path=path,
            colormap=colormap,
            clim=clim,
            xlim=xlim,
            ylim=ylim,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
        )


def recurse_children(d, depth=0, max_depth=-1, types=[], keys=[]) -> list[Child]:
    if max_depth >= 0 and depth >= max_depth:
        return []
    results: list[Child] = []
    if isinstance(d, dict):
        if "type" in d:
            tps = [*types, d["type"]]
            if "properties" in d and "CData" in d["properties"]:
                results.append(Child(tps, keys, d["properties"]["CData"]))
        else:
            tps = types[:]

        for k, v in d.items():
            results.extend(
                recurse_children(v, depth=depth + 1, types=tps, keys=[*keys, k])
            )
    elif isinstance(d, list):
        for ii, v in enumerate(d):
            results.extend(
                recurse_children(v, depth=depth + 1, types=types, keys=[*keys, ii])
            )

    return results


def recurse(key, dat, depth=0):
    if isinstance(dat, dict):
        print(f"{'-' * depth} {key}::")
        for k, v in dat.items():
            recurse(k, v, depth + 1)
    elif isinstance(dat, np.ndarray):
        print(f"{'-' * depth} {key}: array{dat.shape}")
    elif isinstance(dat, list):
        print(f"{'-' * depth} {key}: list({len(dat)})")
        for ii, item in enumerate(dat):
            recurse(f"[{ii}]", item, depth + 1)
    else:
        print(f"{'-' * depth} {key}: {type(dat)}")


def search(
    dat, content, keys: list = [], types: list[str] = []
) -> list[tuple[list, list]]:
    result = []
    if isinstance(dat, dict):
        if "type" in dat:
            tps = [*types, dat["type"]]
        else:
            tps = types[:]
        for k, v in dat.items():
            result.extend(search(v, content, keys=[*keys, k], types=tps))
        return result
    elif isinstance(dat, np.ndarray):
        return []
    elif isinstance(dat, list):
        for ii, item in enumerate(dat):
            result.extend(search(item, content, keys=[*keys, ii], types=types))
        return result
    else:
        if dat == content:
            return [(keys, types)]
        elif hasattr(dat, "__contains__"):
            try:
                if content in dat:
                    return [(keys, types)]
                return []
            except Exception:
                return []
        else:
            return []


def get(dat, *keys):
    if len(keys) == 0:
        return dat
    else:
        return get(dat[keys[0]], *keys[1:])


def process(args: ProcessArgs, config: dict[str, Any] = {}):

    if not args.in_path.is_dir():
        raise ValueError(
            f"Input ({args.in_path}) should be a directory with .fig files."
        )
    if not args.out_path.is_dir():
        raise ValueError(
            f"Output ({args.out_path}) should be a directory with .fig files."
        )

    assert args.origin == "upper" or args.origin == "lower"
    args.origin = cast(Literal["upper", "lower"], args.origin)

    images: list[Plot] = []
    for file in args.in_path.iterdir():
        if file.suffix == ".fig":
            mat = scipy.io.loadmat(
                file,
                simplify_cells=True,
            )

            data_children = recurse_children(mat, depth=0, max_depth=3)

            found_data = False
            for c in data_children:
                if "axes" in c.types and "image" in c.types:
                    if len(c.data.shape) == 2:
                        images.append(Plot.from_child(mat, c, file))
                        found_data = True

            logger.info(file.name)
            if not found_data:
                logger.warning(f"-> Did not find any 2d data in {file.name}")

    colormap = args.colormap.strip("\"'")
    shape = None
    total_colormap = None
    can_use_colormap = False
    total_extent = None
    xlabel = None
    ylabel = None
    for ii, plot_data in enumerate(images):
        image = plot_data.child.data
        if shape is None and total_colormap is None and total_extent is None:
            shape = image.shape
            total_colormap = plot_data.colormap
            can_use_colormap = True
            total_extent = (*plot_data.xlim, *plot_data.ylim)
            xlabel = plot_data.xlabel
            ylabel = plot_data.ylabel
        else:
            assert shape == image.shape
            can_use_colormap = total_colormap == plot_data.colormap
            assert total_extent == (*plot_data.xlim, *plot_data.ylim)
            if xlabel != plot_data.xlabel:
                xlabel = None
            if ylabel != plot_data.ylabel:
                ylabel = None

    assert can_use_colormap or (colormap != "auto")
    assert shape is not None and total_extent is not None

    total_image = np.sum([img.child.data for img in images], axis=0)

    if args.interactive:
        percentiles = np.percentile(total_image, [2, 98])

        min, max, should_plot = show_interactively(
            total_image,
            colormap if colormap != "auto" else total_colormap,
            args.min_mass if args.min_mass is not None else percentiles[0],
            args.max_mass if args.max_mass is not None else percentiles[1],
            total_extent,
            origin=args.origin,
        )
    else:
        should_plot = True
        percentiles = np.percentile(total_image, [0, 100])
        min = args.min_mass if args.min_mass is not None else percentiles[0]
        max = args.max_mass if args.max_mass is not None else percentiles[1]

    if not should_plot:
        logger.info("Canceled")
        return

    figsize = (8, int((8 / shape[1]) * shape[0]))
    aspect = builtins.min((20 / shape[1]) * shape[0], 20)

    for ii, plot_data in enumerate(images):
        image = plot_data.child.data
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_title(plot_data.title)
        ax.set_xlabel(plot_data.xlabel)
        ax.set_ylabel(plot_data.ylabel)
        vmin = plot_data.clim[0] if args.independent_scales else args.min_mass
        vmax = plot_data.clim[1] if args.independent_scales else args.max_mass
        im = ax.imshow(
            image,
            cmap=plot_data.colormap
            if args.independent_colors or colormap == "auto"
            else colormap,
            vmin=vmin,
            vmax=vmax,
            extent=(*plot_data.xlim, *plot_data.ylim),
            origin=args.origin,
        )

        fig.colorbar(
            im,
            ax=ax,
            location="right",
            shrink=0.8,
            format="%.2e",
            ticks=np.linspace(vmin, vmax, 6),
            aspect=aspect,
        )

        path = args.out_path / f"{plot_data.path.stem}.fig.png"

        fig.savefig(path)
        plt.close(fig)

        np.savetxt(args.out_path / f"{plot_data.path.stem}.csv", image, delimiter=",")

    file_title = args.final_title.replace(":", "-").replace("?", "")
    file_title = re.sub(r"[^A-Za-z0-9+ ._-]", "_", file_title)
    logger.info(f"Writing: summed image to {file_title}")
    np.savetxt(args.out_path / f"{file_title}.total.csv", total_image, delimiter=",")
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(args.final_title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    im = ax.imshow(
        total_image,
        cmap=colormap if colormap != "auto" else total_colormap,
        vmin=min,
        vmax=max,
        extent=total_extent,
        origin=args.origin,
    )
    fig.colorbar(
        im,
        ax=ax,
        location="right",
        shrink=0.8,
        format="%.2e",
        ticks=np.linspace(min, max, 6),
        aspect=aspect,
    )
    fig.savefig(args.out_path / f"{file_title}.total.png")
