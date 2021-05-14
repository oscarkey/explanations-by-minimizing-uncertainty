# Interpretable Counterfactual Explanations By Minimizing Uncertainty

This repository contains the code for _Generating Interpretable Counterfactual Explanations By Implicit Minimisation of Epistemic and Aleatoric Uncertainties_;
Lisa Schut*, Oscar Key*, Rory McGrath, Luca Costabello, Bogdan Sacaleanu, Medb Corcoran, Yarin Gal.

The paper has been [accepted at AISTATS 2021](https://aistats.org/aistats2021/accepted.html).

It is also available on arXiv: [arXiv:2103.08951](https://arxiv.org/abs/2103.08951)

## Environment set up
Install Miniconda or Anaconda, then use the command:
`conda env update -f environment.yml`

To run the tests: `pytest tests`

To format the code: `black -l 99 **/*.py`


## Reproducing results
1. Train an ensemble with and without adversarial training. This will generate checkpoints in `~/ces_results/ensembles/[id]`:

    `python experiments/train_ensemble.py --results_dir=~/ces_results --data_dir=~/pytorch-datasets --epochs=100 --n_ensembles=50 --n_hidden=200 --dataset=mnist --adv_training`

    `python experiments/train_ensemble.py --results_dir=~/ces_results --data_dir=~/pytorch-datasets --epochs=100 --n_ensembles=50 --n_hidden=200 --dataset=mnist`

2. Generate and evaluate the CEs

    a. For quantitative evaluation (generating CEs to examine by eye)

    `python experiments/exp_qualitative_evaluation.py --results_dir=~/ces_results --data_dir=~/pytorch-datasets --adv_training_id=[id] --no_adv_training_id=[id]`

    b. For qualitative evaluation (computing evaluation metrics over several repeats of the experiment)

    `python experiments/exp_quantitative_evaluation.py --results_dir=~/ces_results --data_dir=~/pytorch-datasets --adv_training_id=[id] --no_adv_training_id=[id]`

   Note that running the evaluation will generate a set of CEs, and train several evaluation autoencoders.
   These are cached in `results_dir` to allow the evaluation to be re-run quickly for debugging purposes.
   If you alter the code you may need to delete this cache to see the results of your change.


## Pointers to useful files
- `uces/generators.py` contains the algorithm for generating CEs
- `uces/im_evaluation.py` contains the implementation of the IM1 and IM2 evaluation metrics


## Citation
If the paper or code has been useful in your work, please reference it as follows:
```
@InProceedings{pmlr-v130-schut21a,
  title     = { Generating Interpretable Counterfactual Explanations By Implicit Minimisation of Epistemic and Aleatoric Uncertainties },
  author    = {Schut, Lisa and Key, Oscar and Mc Grath, Rory and Costabello, Luca and Sacaleanu, Bogdan and Corcoran, Medb and Gal, Yarin},
  booktitle = {Proceedings of The 24th International Conference on Artificial Intelligence and Statistics},
  pages     = {1756--1764},
  year      = {2021},
  editor    = {Banerjee, Arindam and Fukumizu, Kenji},
  volume    = {130},
  series    = {Proceedings of Machine Learning Research},
  month     = {13--15 Apr},
  publisher = {PMLR},
  pdf       = {http://proceedings.mlr.press/v130/schut21a/schut21a.pdf},
  url       = {http://proceedings.mlr.press/v130/schut21a.html},
}
```

## License
We release this code under the MIT license, see `LICENSE.txt`.
