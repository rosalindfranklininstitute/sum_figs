# SPDX-FileCopyrightText: 2026 RFI
#
# SPDX-License-Identifier: Apache-2.0

import argparse
from ms_nexus_tools.api import args as nxargs

from . import collect_figs


def main() -> None:
    parser = argparse.ArgumentParser(prog="figs")

    nxargs.add_arguments(parser, collect_figs.ProcessArgs)

    args, config_dict = collect_figs.ProcessArgs.parse_args(parser)

    process_args = collect_figs.ProcessArgs(**vars(args))

    collect_figs.process(process_args, config_dict)
