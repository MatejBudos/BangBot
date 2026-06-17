# Generation eval results (LLM-as-judge)

| Kategória   | N  | Gold@sel | Faith. | Correct. | Cites | Slovak | Avg searches |
|-------------|----|---------:|-------:|---------:|------:|-------:|-------------:|
| interaction | 5  |      80% | 1.0/2  | 1.0/2     |   60% |   100% | 4.2          |
| lookup      | 3  |      67% | 2.0/2  | 2.0/2     |  100% |   100% | 1.7          |
| paraphrase  | 2  |      50% | 2.0/2  | 2.0/2     |  100% |   100% | 4.5          |
| refusal     | 2  | —        | —      | —        | —     |   100% | refused: 100%  |
| **TOTAL**   | 12 |      70% | 1.5/2  | 1.5/2     |   80% |   100% | 3.5          |

_Eval set: 12 prípadov. Judge: gpt-4o-mini (same-model bias — interpretuj konzervatívne)._
