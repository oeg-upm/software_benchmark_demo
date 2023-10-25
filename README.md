# software_benchmark_demo
This demo extract software mentions from a text. It uses fine-tuned models from SCIBERT using different corpora.
* SoMESCi [1]: We have used the corpus uploaded to [Github](https://github.com/dave-s477/SoMeSci/tree/9f17a43f342be026f97f03749457d4abb1b01dbf/PLoS_sentences), more specifically, the corpus created with sentences.
* Softcite [2]: This project has published another corpus for software mentions, which is also available on [Github](https://github.com/howisonlab/softcite-dataset/tree/master/data/corpus). We have to note that we only use the annotations from bio domain.
* [Benchmark Bio](https://huggingface.co/oeg/software_benchmark_bio): Corpus created by reconciling the somesci and softcite corpora. Only BIO domain
* [Benchmark Multidomain](https://huggingface.co/oeg/software_benchmark_multidomain): Corpus created by reconciling the somesci and softcite corpora (including economics domain) and papers with code [corpus](https://doi.org/10.5281/zenodo.10033751).

**Demo:** Avaliable [here](https://software-benchmark.linkeddata.es/).

**Authors:** Esteban Gonzalez and Daniel Garijo

## Requirements:
This demo has been tested in Unix and Windowsoperating systems.

- Python 3.10
- PIP
- Streamlit 1.24

You need to create a folder models in the software_benchmark_demo folder and download the models from the huggingface page.
* [Benchmark Bio](https://huggingface.co/oeg/software_benchmark_bio).
* [Benchmark Multidomain](https://huggingface.co/oeg/software_benchmark_multidomain).


## Install from GitHub

To run software_benchmark_demo, please follow the next steps:

Clone this GitHub repository

```
git clone https://github.com/oeg-upm/software_benchmark_demo.git
```

Install the python libraries required.

```
cd software_benchmark_demo
pip install -e .
```
To run the service, execute

```
streamlit run app.py
```

The service will be available in the port 8501.

## References
1. Schindler, D., Bensmann, F., Dietze, S., & Krüger, F. (2021, October). Somesci-A 5 star open data gold standard knowledge graph of software mentions in scientific articles. In Proceedings of the 30th ACM International Conference on Information & Knowledge Management (pp. 4574-4583).
2. Du, C., Cohoon, J., Lopez, P., & Howison, J. (2021). Softcite dataset: A dataset of software mentions in biomedical and economic research publications. Journal of the Association for Information Science and Technology, 72(7), 870-884.
