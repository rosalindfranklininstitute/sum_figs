# SPDX-FileCopyrightText: 2026 RFI
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, NamedTuple, Literal, cast
from dataclasses import dataclass

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

    @staticmethod
    def from_child(mat, child: Child, path: Path) -> "Plot":
        keys = child.keys[:-2]
        colormap = ListedColormap(get(mat, *keys, "properties", "Colormap"))
        clim = get(mat, *keys, "properties", "CLim")
        xlim = get(mat, *keys, "properties", "XLim")
        ylim = get(mat, *keys, "properties", "YLim")
        return Plot(
            child=child, path=path, colormap=colormap, clim=clim, xlim=xlim, ylim=ylim
        )


def recurse_children(d, depth=0, max_depth=-1, types=[], keys=[]) -> list[Child]:
    if max_depth >= 0 and depth >= max_depth:
        return []
    results = []
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

    if len(results) > 0:
        return results
    else:
        return []


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

            print(file.name)
            if not found_data:
                print(f"-> Did not find any 2d data in {file.name}")

    colormap = args.colormap.strip("\"'")
    shape = None
    total_colormap = None
    can_use_colormap = False
    total_extent = None
    for ii, plot_data in enumerate(images):
        image = plot_data.child.data
        if shape is None and total_colormap is None and total_extent is None:
            shape = image.shape
            total_colormap = plot_data.colormap
            can_use_colormap = True
            total_extent = (*plot_data.xlim, *plot_data.ylim)
        else:
            assert shape == image.shape
            can_use_colormap = total_colormap == plot_data.colormap
            assert total_extent == (*plot_data.xlim, *plot_data.ylim)

    assert can_use_colormap or (colormap != "auto")
    assert shape is not None and total_extent is not None

    total_image = np.sum([img[0].data for img in images], axis=0)

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
        print("Canceled")
        return

    for ii, plot_data in enumerate(images):
        image = plot_data.child.data
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_title(plot_data.path.name)
        im = ax.imshow(
            image,
            cmap=plot_data.colormap
            if args.independent_colors or colormap == "auto"
            else colormap,
            vmin=plot_data.clim[0] if args.independent_scales else args.min_mass,
            vmax=plot_data.clim[1] if args.independent_scales else args.max_mass,
            extent=(*plot_data.xlim, *plot_data.ylim),
            origin=args.origin,
        )
        fig.colorbar(im, ax=ax, location="right")
        fig.savefig(args.out_path / f"{plot_data.path.stem}.png")
        plt.close(fig)

        np.savetxt(args.out_path / f"{plot_data.path.stem}.csv", image, delimiter=",")

    np.savetxt(args.out_path / "Total Image.csv", total_image, delimiter=",")
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_title(f"Total Image ({min:.2g} - {max:.2g})")
    im = ax.imshow(
        total_image,
        cmap=colormap if colormap != "auto" else total_colormap,
        vmin=min,
        vmax=max,
        extent=total_extent,
        origin=args.origin,
    )
    fig.colorbar(im, ax=ax, location="right")
    fig.savefig(args.out_path / "Total Image.png")
