# Mosaic Lex GLUE

This repo uses the [Mosaic CLI](https://internal.mcli.docs.mosaicml.com/index.html) to run [lex-GLUE](https://huggingface.co/datasets/lex_glue) fine-tuning and evaluation on the Mosaic Cloud.

Currently there are working YAMLs for the tasks:
* ledgar
* unfair_tos

All tasks other than case_hold _should_ work, (and maybe even case_hold, but I haven't tested my gpt_for_multiple_choice code yet), but most we should not be able to replicate the reported numbers since the authors said they used e.g. BERT when they in fact used a custom transformer model based on BERT. Fun!

## Notes on tasks

Multiple-choice QA is like single label classification, except it needs to make that classification based on a tuple of `(input: str, choices: List[str])` (choices is always length 5). There is a task-specific HF head for this for some models; I wrote one for `gpt` style models but it needs testing.

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
  * type: multi_class
  * notes: almost all samples < 500 tokens; 100 classes
* 'scotus'
  * type: multi_class
  * notes: VERY long, sample just as likely to have 6000 tokens as 500
* 'unfair_tos'
  * type: multi_label
  * notes: very short, almost all samples < 100tokens
