# Mosaic Lex GLUE

This repo uses the [Mosaic CLI](https://internal.mcli.docs.mosaicml.com/index.html) to run [lex-GLUE](https://huggingface.co/datasets/lex_glue) fine-tuning and evaluation on the Mosaic Cloud.

Currently there is a complete YAML for the LEDGAR task.

## Notes on tasks

For length information details, see [the lex-GLUE paper, page 6](https://arxiv.org/pdf/2110.00976.pdf#page=6)

* 'case_hold'
  * type: multiple_choice_qa
  * notes: almost all samples between 200-300 tokens
* 'ecthr_a'
  * type: multi_label
  * notes: very long, most samples under 2000 tokens, but very nontrivial long tail up to ~7000
* 'ecthr_b'
  * type: multi_label
  * notes: very long, most samples under 2000 tokens, but very nontrivial long tail up to ~7000
* 'eurlex'
  * type: multi_label
  * notes: most docs < 1000 tokens, but nontrivial long tail up to 3000
* 'ledgar'
  * type: single_label
  * notes: almost all samples < 500 tokens; 100 classes
* 'scotus'
  * type: single_label
  * notes: VERY long, sample just as likely to have 6000 tokens as 500
* 'unfair_tos'
  * type: multi_label
  * notes: very short, almost all samples < 100tokens
