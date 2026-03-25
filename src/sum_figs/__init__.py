# SPDX-FileCopyrightText: 2026 RFI
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import matplotlib

from ms_nexus_tools.api import args as nxargs

from . import collect_figs

matplotlib.use("QtAgg")

from icecream import ic
from dataclasses import fields


def main() -> None:
    partial_args = collect_figs.ProcessArgs.parse_config("figs")
    process_args = collect_figs.ProcessArgs.parse_interactive(
        "figs", partial_args.remaining_args
    )

    collect_figs.process(process_args, partial_args.config)
