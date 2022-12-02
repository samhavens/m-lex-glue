# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""The main runner. Spawns MCloud runs for each subtask

Unlike our GLUE finetune script, this can be run from anywhere,
if you are authenticated into MCloud. It is a script which starts MCloud runs"""

import datetime
import os
import sys
import glob
from typing import Dict, List

from mcli.api.model.run import Run
from mcli.models.run_config import RunConfig
from mcli.api.runs import create_run
from mcli.utils.utils_yaml import load_yaml


def group_tag(configs: List[Dict]):
    """If all runs are configured to report to wandb, add a wandb group to group all runs together
    If `group` is already used, fall back to `job_type`
    If both are used, do not group the runs
    @NOTE to MCLI team: I would have modified the wandb.config if that were possible here
    """
    all_wandb = all(any(i["integration_type"] == "wandb" for i in c["integrations"]) for c in configs)
    timestamp = "{:%Y-%m-%d-%Hh-%Mm-%Ss}".format(datetime.datetime.now())
    if all_wandb:
        any_grouped = any(any(i["integration_type"] == "wandb" and "group" in i for i in c["integrations"]) for c in configs)
        any_job_type = any(any(i["integration_type"] == "wandb" and "job_type" in i for i in c["integrations"]) for c in configs)
        if any_grouped and any_job_type:
            print("WARNING: cannot group these runs in wandb as both job_type and group are set")
        for c in configs:
            for i in c["integrations"]:
                if i["integration_type"] == "wandb":
                    if any_grouped:
                        # we cannot use the group attribute since it is in use, so use job_type
                        i["job_type"] = [timestamp]
                    else:
                        i["group"] = timestamp
    return configs


if __name__ == "__main__":
    yamls_to_run = sys.argv[1:]
    if len(yamls_to_run) == 1 and os.path.isdir(yamls_to_run[0]):
        yamls_to_run = glob.glob(f'{yamls_to_run[0]}/*.yaml')
    if len(yamls_to_run) == 0:
        print("Pass a list of yaml files or directory containing them to main.py to run each fine-tuning lex-GLUE job")
        exit()
    # load the yamls
    configs = [load_yaml(f) for f in yamls_to_run]
    # if they are all being tracked in wandb, group them with a tag
    configs = group_tag(configs)  # type: ignore
    mc_run_configs: List[RunConfig] = [RunConfig.from_dict(f) for f in configs]
    print("Launching runs:\n")
    for run_conf in mc_run_configs:
        run: Run = create_run(run_conf)
        print(f'- run: {run.name}')  # type: ignore
        print(f"\t- id {run.run_uid}")
        print(f"\t- status {run.status}")
        print(f"\t- created_at {run.created_at}")
        # do we want to just exit here? or show the status of all the runs?
