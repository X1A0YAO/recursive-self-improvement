# Phase 4 数据划分（IFEval 官方集，锁定）

- 生成：SEED=20261107，分层键=instruction_id_list 首 id
- Train 181 / Validation 180 / Test 180

| instruction type | train | val | test |
|---|---|---|---|
| change_case:capital_word_frequency | 6 | 6 | 5 |
| change_case:english_capital | 7 | 7 | 7 |
| change_case:english_lowercase | 9 | 9 | 9 |
| combination:repeat_prompt | 12 | 12 | 11 |
| combination:two_responses | 8 | 8 | 7 |
| detectable_content:number_placeholders | 5 | 5 | 6 |
| detectable_content:postscript | 6 | 6 | 5 |
| detectable_format:constrained_response | 3 | 3 | 4 |
| detectable_format:json_format | 6 | 6 | 5 |
| detectable_format:multiple_sections | 3 | 3 | 3 |
| detectable_format:number_bullet_lists | 6 | 6 | 7 |
| detectable_format:number_highlighted_sections | 13 | 13 | 13 |
| detectable_format:title | 7 | 7 | 6 |
| keywords:existence | 5 | 5 | 5 |
| keywords:forbidden_words | 9 | 9 | 9 |
| keywords:frequency | 8 | 8 | 7 |
| keywords:letter_frequency | 8 | 8 | 7 |
| language:response_language | 7 | 7 | 7 |
| length_constraints:nth_paragraph_first_word | 3 | 3 | 4 |
| length_constraints:number_paragraphs | 6 | 6 | 7 |
| length_constraints:number_sentences | 8 | 8 | 8 |
| length_constraints:number_words | 8 | 8 | 8 |
| punctuation:no_comma | 11 | 11 | 12 |
| startend:end_checker | 6 | 6 | 7 |
| startend:quotation | 11 | 10 | 11 |
