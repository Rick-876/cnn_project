---
tags:
- ASAG
- Grading
- Short Answer Grading
- SAG
- Automated Short Answer Grading
- ASAG2024
- Automated Short Answer Grading Benchmark
- Automated Short Answer Grading Dataset
pretty_name: ASAG2024
size_categories:
- 10K<n<100K
---

# Dataset Card for ASAG2024 (Automated Short Answer Grading Benchmark)

<!-- Provide a quick summary of the dataset. -->

This is the combined ASAG2024 dataset which consists of various automated grading datasets containing questions, reference answers, provided (student) answers and human grades. 

## Examples

An example on how to use this dataset can be found on GitHub under [https://github.com/GeroVanMi/ASAG2024/blob/main/examples/example_weights.ipynb](https://github.com/GeroVanMi/ASAG2024/blob/main/examples/example_weights.ipynb)


### Dataset Description

<!-- Provide a longer summary of what this dataset is. -->

- **Curated by:** Gérôme Meyer
- **Language(s) (NLP):** English
- **License:** Data Source Licenses apply (see below)

### Citation 
If you use this in your works, please cite: 

Meyer, Gérôme, Philip Breuer, and Jonathan Fürst. "ASAG2024: A Combined Benchmark for Short Answer Grading." arXiv preprint arXiv:2409.18596 (2024).

```bibtex
@inproceedings{meyer2024asag2024,
  title={ASAG2024: A Combined Benchmark for Short Answer Grading},
  author={Meyer, G{\'e}r{\^o}me and Breuer, Philip and F{\"u}rst, Jonathan},
  booktitle={Proceedings of the 2024 on ACM Virtual Global Computing Education Conference V. 2},
  pages={322--323},
  year={2024}
}
```

Also please consider citing the original dataset sources listed below: 

### Dataset Sources

This dataset was collected from the sources listed below. If you use this in your work, please make sure to cite the original authors. 

#### Stita
Repository: https://github.com/edgresearch/dataset-automaticgrading-2022/tree/master

Citation: 
```
del Gobbo, E., Guarino, A., Cafarelli, B. et al. GradeAid: a framework for automatic short answers grading in educational contexts—design, implementation and evaluation. _Knowl Inf Syst_ 65, 4295–4334 (2023). https://doi.org/10.1007/s10115-023-01892-9
```

BibTex: 
```
@Article{delGobbo2023,
  author={del Gobbo, Emiliano
  and Guarino, Alfonso
  and Cafarelli, Barbara
  and Grilli, Luca},
  title={GradeAid: a framework for automatic short answers grading in educational contexts---design, implementation and evaluation},
  journal={Knowledge and Information Systems},
  year={2023},
  month={Oct},
  day={01},
  volume={65},
  number={10},
  pages={4295-4334},
  issn={0219-3116},
  doi={10.1007/s10115-023-01892-9},
  url={https://doi.org/10.1007/s10115-023-01892-9}
}
```

#### Short-Answer Feedback (SAF)

Citation:
```
A. Filighera, S. Parihar, T. Steuer, T. Meuser, and S. Ochs, ‘Your Answer is Incorrect… Would you like to know why? Introducing a Bilingual Short Answer Feedback Dataset’, in Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), S. Muresan, P. Nakov, and A. Villavicencio, Eds., Dublin, Ireland: Association for Computational Linguistics, May 2022, pp. 8577–8591. doi: 10.18653/v1/2022.acl-long.587.
```

BibTex:
```
@inproceedings{filighera-etal-2022-answer,
    title = "Your Answer is Incorrect... Would you like to know why? Introducing a Bilingual Short Answer Feedback Dataset",
    author = "Filighera, Anna  and
      Parihar, Siddharth  and
      Steuer, Tim  and
      Meuser, Tobias  and
      Ochs, Sebastian",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.587",
    pages = "8577--8591",
   }
```

#### Mohler et al.

BibTex:
```
@inproceedings{dataset_mohler,
    title = "Learning to Grade Short Answer Questions using Semantic Similarity Measures and Dependency Graph Alignments",
    author = "Mohler, Michael  and
      Bunescu, Razvan  and
      Mihalcea, Rada",
    editor = "Lin, Dekang  and
      Matsumoto, Yuji  and
      Mihalcea, Rada",
    booktitle = "Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2011",
    address = "Portland, Oregon, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/P11-1076",
    pages = "752--762",
}
``` 

#### Beetle II 
```
@article{dataset_beetleII,
  title={BEETLE II: Deep natural language understanding and automatic feedback generation for intelligent tutoring in basic electricity and electronics},
  author={Dzikovska, Myroslava and Steinhauser, Natalie and Farrow, Elaine and Moore, Johanna and Campbell, Gwendolyn},
  journal={International Journal of Artificial Intelligence in Education},
  volume={24},
  pages={284--332},
  year={2014},
  publisher={Springer}
}
```

#### CU-NLP
```
@ARTICLE{dataset_cunlp,
  author={Tulu, Cagatay Neftali and Ozkaya, Ozge and Orhan, Umut},
  journal={IEEE Access}, 
  title={Automatic Short Answer Grading With SemSpace Sense Vectors and MaLSTM}, 
  year={2021},
  volume={9},
  number={},
  pages={19270-19280},
  keywords={Semantics;Natural language processing;Benchmark testing;Long short term memory;Deep learning;Task analysis;Learning systems;Automatic short answer grading;MaLSTM;semspace sense vectors;synset based sense embedding;sentence similarity},
  doi={10.1109/ACCESS.2021.3054346}}

  @inproceedings{dataset_scientsbank,
  title={Annotating Students’ Understanding of Science Concepts},
  author={Rodney D. Nielsen and Wayne H. Ward and James H. Martin and Martha Palmer},
  booktitle={International Conference on Language Resources and Evaluation},
  year={2008},
  url={https://api.semanticscholar.org/CorpusID:12938607}
}
```

<!-- ## Dataset Structure -->

<!-- This section provides a description of the dataset fields, and additional information about the dataset structure such as criteria used to create the splits, relationships between data points, etc. -->

<!-- [More Information Needed]

## Dataset Creation

### Curation Rationale -->

<!-- Motivation for the creation of this dataset. -->

[More Information Needed]

<!-- ### Source Data -->

<!-- This section describes the source data (e.g. news text and headlines, social media posts, translated sentences, ...). -->

<!-- #### Data Collection and Processing -->

<!-- This section describes the data collection and processing process such as data selection criteria, filtering and normalization methods, tools and libraries used, etc. -->

<!-- [More Information Needed] -->

<!-- #### Who are the source data producers? -->

<!-- This section describes the people or systems who originally created the data. It should also include self-reported demographic or identity information for the source data creators if this information is available. -->

<!-- [More Information Needed]

<!-- ## Bias, Risks, and Limitations -->

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

<!-- [More Information Needed] -->

<!-- ### Recommendations -->

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

<!-- Users should be made aware of the risks, biases and limitations of the dataset. More information needed for further recommendations. -->



<!-- **BibTeX:**

[More Information Needed] -->

<!-- **APA:**

[More Information Needed] -->

<!-- ## Glossary [optional] -->

<!-- If relevant, include terms and calculations in this section that can help readers understand the dataset or dataset card. -->


## Dataset Card Authors

- Gérôme Meyer
- Philip Breuer

## Dataset Card Contact

- E-Mail: gerome.meyer@pm.me