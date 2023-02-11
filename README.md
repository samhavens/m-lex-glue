# Mosaic Lex GLUE

This repo uses the [Mosaic CLI](https://internal.mcli.docs.mosaicml.com/index.html) to run [lex-GLUE](https://huggingface.co/datasets/lex_glue) fine-tuning and evaluation on the Mosaic Cloud.

This repo is not well-named, as it also includes the [BillSum](https://huggingface.co/datasets/billsum) legal summarization task. It also should probably include the code for converting Pile of Law to MDS format and creating the custom tokenizer.

## Tasks and Model compatibility

Most of the tasks use task-specific classification heads. Some models, including MosaicGPT, don't have that implemented yet. We either need to write prompt templates, so all tasks can be done as language modeling tasks, or write the MosaicGPT task specific heads.

## Known Issues

* Causal LM implementation (used for summarization training) is broken. I was bringing in the code from `examples/llm` and it has errors.
* There are bugs in the summarization dataloader as well... Since we are using 16k seq len now, we should just be able to concat together document and summary and toss anything too long (should be only 1-3 documents)

## Debugging

For ROUGE to work, you have to install nltk (in requirements, not hard) and download punkt. From shell:

```python
python -c "import nltk;nltk.download('punkt')"
```

## Notes on tasks

Multiple-choice QA is like single label classification, except it needs to make that classification based on a tuple of `(input: str, choices: List[str])` (choices is always length 5). There is a task-specific HF head for this for some models; I wrote one for `gpt` style models, but it needs testing. Be suspicious.

The Ecthr, Eurlex, and Scotus tasks all involve long sequences. The lex-GLUE paper results for these tasks are misleading/inaccurate. When they report that RoBERTa-large got 75.5 / 66.3 on SCOTUS this means that "A hierarchical transformer built on top of RoBERTA representations" got that score. Actual RoBERTa scores, due to max sequence length, are substantially lower. I try to convince the author to report both scores [here](https://github.com/coastalcph/lex-glue/discussions/36).

## Notes on Data

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
