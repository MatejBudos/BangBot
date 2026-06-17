# Generation eval results (LLM-as-judge)

| Kategória   | N  | Gold@sel | Faith. | Correct. | Cites | Slovak | Avg searches |
|-------------|----|---------:|-------:|---------:|------:|-------:|-------------:|
| interaction | 5  |     100% | 1.0/2  | 0.8/2     |   40% |   100% | 5.2          |
| lookup      | 3  |      67% | 2.0/2  | 2.0/2     |  100% |   100% | 1.7          |
| paraphrase  | 2  |      50% | 2.0/2  | 2.0/2     |  100% |   100% | 3.5          |
| refusal     | 2  | —        | —      | —        | —     |   100% | refused: 100%  |
| **TOTAL**   | 12 |      80% | 1.5/2  | 1.4/2     |   70% |   100% | 3.8          |

_Eval set: 12 prípadov. Judge: gpt-4o-mini (same-model bias — interpretuj konzervatívne)._
