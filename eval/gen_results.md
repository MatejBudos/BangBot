# Generation eval results (LLM-as-judge)

| Kategória   | N  | Gold@sel | Faith. | Correct. | Cites | Slovak | Avg searches |
|-------------|----|---------:|-------:|---------:|------:|-------:|-------------:|
| interaction | 5  |     100% | 1.2/2  | 0.8/2     |   20% |   100% | 5.8          |
| lookup      | 3  |      67% | 2.0/2  | 2.0/2     |  100% |   100% | 1.7          |
| paraphrase  | 4  |      75% | 1.5/2  | 1.5/2     |   75% |   100% | 6.0          |
| refusal     | 2  | —        | —      | —        | —     |   100% | refused: 100%  |
| **TOTAL**   | 14 |      83% | 1.5/2  | 1.3/2     |   58% |   100% | 4.7          |

_Eval set: 14 prípadov. Judge: gpt-4o-mini (same-model bias — interpretuj konzervatívne)._
