# 11711-final-project - Program Selection Using Code Execution Agents
## Abstract
Large Language Models (LLMs) have shown remarkable results on code generation tasks such as natural language to program synthesis, code infilling, and test-case synthesis. Despite this recent success, selecting a single correct program from multiple candidates generated by an LLM remains a hard problem. Many state-of-the-art models perform significantly worse when they are given a single opportunity to produce a program (pass@1) compared to when they are given multiple chances (pass@k) \cite{li2022competition}. In this paper, we survey the current state-of-the-art methods aimed at resolving this issue, re-implement the most popular of them (CodeT) \cite{chen2022codet}, and report the baselines we achieve. We also point out key flaws in CodeT and it's reliance on randomly generated test cases, which can lead to suboptimal program selection,. This is why we propose a new method for more accurate program selection, which we plan on implementing for the final project.
