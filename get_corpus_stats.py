import sys

import numpy as np
from omegaconf import DictConfig, OmegaConf as om
from transformers import AutoTokenizer

from m_lex_glue.data.billsum import create_summarization_dataset


def print_percentiles(l):
    ps = [10, 50, 80, 90, 95, 97, 98, 99]
    shortest = min(l)
    longest = max(l)
    print(f"shortest: {shortest}")
    for p in ps:
        v = round(np.percentile(l, p), 1)
        print(f"{p}th %ile: {v}")
    print(f"longest: {longest}")



def main(cfg: DictConfig):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.tokenizer_name)
    ds = create_summarization_dataset(cfg.task, tokenizer, split="test",
                                      truncate=False,max_seq_length=1_000_000)
    input_lengths, target_lengths = [], []
    for batch in ds:
        input_lengths.append(batch['input_ids'].size(dim=0))
        target_lengths.append(batch['labels'].size(dim=0))
    print("Input length distributions")
    print_percentiles(input_lengths)
    print("Target length distributions")
    print_percentiles(target_lengths)


if __name__ == '__main__':
    yaml_path, args_list = sys.argv[1], sys.argv[2:]
    with open(yaml_path) as f:
        yaml_cfg = om.load(f)
    cli_cfg = om.from_cli(args_list)
    cfg: DictConfig = om.merge(yaml_cfg, cli_cfg)  # type: ignore
    main(cfg)

"""with POL tokenizer:

Input length distributions
shortest: 1182
10th %ile: 1877.8
50th %ile: 3286.0
80th %ile: 5170.8
90th %ile: 6131.0
95th %ile: 6793.2
97th %ile: 7337.4
98th %ile: 7614.3
99th %ile: 8038.4
longest: 15590

Target length distributions
shortest: 14
10th %ile: 69.8
50th %ile: 195.0
80th %ile: 323.0
90th %ile: 408.2
95th %ile: 498.0
97th %ile: 584.9
98th %ile: 637.6
99th %ile: 722.3
longest: 940

"""