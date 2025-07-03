# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python
"""
eval.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import tyro

from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import CONSOLE

import yaml

@dataclass
class ComputePSNR:
    """Load a checkpoint, compute some PSNR metrics, and save it to a JSON file."""

    # Path to config YAML file.
    load_config: Path
    # Name of the output file.
    output_path: Path = Path("output.json")
    # Optional path to save rendered outputs to.
    render_output_path: Optional[Path] = None
        
    # HanwenNEW: for a different evaluation path
    input_path: Optional[Path] = None  
    # HanwenNEW: write psnr on render image
    annotate_psnr_on_image: bool = False


    eval_mode: Literal["fraction", "filename", "interval", "all", "filename_fraction_test", "filename_fraction_val"] = "filename_fraction_test"


    def main(self) -> None:
        """Main function."""

        # HanwenNew
        if self.input_path is not None:
            raw_config = yaml.load(self.load_config.read_text(), Loader=yaml.Loader)
            # Set evaldata in the dataparser
            raw_config.pipeline.datamanager.dataparser.evaldata = self.input_path
            raw_config.pipeline.datamanager.dataparser.eval_mode = self.eval_mode
            # Write to a temp config so eval_setup can use it
            temp_config_path = self.load_config.parent / "patched_config.yml"
            temp_config_path.write_text(yaml.dump(raw_config))
            config_path_to_use = temp_config_path
        else:
            config_path_to_use = self.load_config

        config, pipeline, checkpoint_path, _ = eval_setup(config_path_to_use)
        
        assert self.output_path.suffix == ".json"
        if self.render_output_path is not None:
            self.render_output_path.mkdir(parents=True, exist_ok=True)
        metrics_dict = pipeline.get_average_eval_image_metrics(output_path=self.render_output_path, get_std=True, annotate_psnr_on_image = self.annotate_psnr_on_image)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        # Get the output and define the names to save to
        benchmark_info = {
            "experiment_name": config.experiment_name,
            "method_name": config.method_name,
            "checkpoint": str(checkpoint_path),
            "results": metrics_dict,
        }
        # Save output to output file
        self.output_path.write_text(json.dumps(benchmark_info, indent=2), "utf8")
        CONSOLE.print(f"Saved results to: {self.output_path}")


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(ComputePSNR).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(ComputePSNR)  # noqa
