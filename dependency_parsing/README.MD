# Dependency Parsing

This repository is part of the experiment of replicating the Indonesian NLP modules.

## Disclaimer

This is a modified version of the original [repository](https://github.com/indolem/indolem/tree/main/dependency_parsing) for easier end-to-end use. Please follow the required steps [here](https://github.com/indolem/indolem/tree/main/dependency_parsing) before trying this modified version.


## Models

You can used pretrained model from [here](https://drive.google.com/file/d/1NlRKhEvbASgU_1PPKm9jsSUHPYejTNWJ/view?usp=sharing) and put the extracted files in `experiments/scripts/models/`.


## How to Use

Input should be pair of word and its POS Tag.

```python
from dep_parser import get_dependency

get_dependency([("Nama", "NN"), ("saya", "PRP"), ("Bambang", "NNP")])
```

## References
    Ika Alfina, Arawinda Dinakaramani, Mohamad Ivan Fanany, and Heru Suhartanto. "A Gold Standard Dependency Treebank for Indonesian". In the Proceeding of 33rd Pacific Asia Conference on Language, Information and Computation (PACLIC) 2019 in Hakodate, Japan, 13-15 September 2019.
    Ika Alfina, Daniel Zeman, Arawinda Dinakaramani, Indra Budi, and Heru Suhartanto. "Selecting the UD v2 Morphological Features for Indonesian Dependency Treebank". In the Proceeding of the 2020 International Conference of Asian Language Processing (IALP) in Kuala Lumpur, Malaysia, 4-6 Desember 2020.
    Koto, F., Rahimi, A., Lau, J. H., and Baldwin, T. (2020). IndoLEM and IndoBERT: A benchmark dataset and pre-trained language model for Indonesian NLP. In Proceedings of the 28th International Conference on Computational Linguistics, pages 757–770, Barcelona, Spain (Online). International Committee on Computational Linguistics.
    McDonald, R., Nivre, J., Quirmbach-Brundage, Y., Goldberg, Y., Das, D., Ganchev, K., Hall, K., Petrov, S., Zhang, H., Tackstr ¨ om, O., Bedini, C., Bertomeu Castell ¨ o, ´ N., and Lee, J. (2013). Universal Dependency annotation for multilingual parsing. In Proceedings of the 51st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pages 92–97, Sofia, Bulgaria. Association for Computational Linguistics
    