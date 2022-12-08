multi_class = (None, "text", "label")  # none, str, int
multi_label = (None, "text", "labels")  # none, str, List[int]
multiple_choice_qa = ("context", "endings", "label")  # str, str, int
summarization = ("title", "text", "summary")


task_example_types = {
    'billsum': summarization,
    'case_hold': multiple_choice_qa,
    'ecthr_a': multi_label,
    'ecthr_b': multi_label,
    'eurlex': multi_label,
    'ledgar': multi_class,
    'scotus': multi_class,
    'unfair_tos': multi_label,
}

summarization_tasks = ['billsum']

lex_glue_tasks = list(task_example_types.keys())
for sum_task in summarization_tasks:
    lex_glue_tasks.remove(sum_task)