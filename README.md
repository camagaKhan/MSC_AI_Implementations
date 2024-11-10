# Exploring Hate Speech Classification Methods in Online Maltese News Portal Comments

## How to install the environment? 

1) Download miniconda: [Miniconda Page](https://docs.anaconda.com/miniconda/miniconda-install/)
2) After cloning the repository, locate the ```environment.yml``` file and install all the project dependencies by typing ```conda env create -f environment.yaml``` in the terminal. Be sure to update this line ```prefix: C:\Users\liamm\anaconda3\envs\msc_proper`` inside the *.yml file.


## Project Structure

1) The Metrics_results contain the all the metric results for all the models we used for our experiments. All these will be saved as pickle files, while training, validation, and evaluation of models are performed.
2) The PreProcess folder contains all the valida notebooks for analysing the chosen dataset.
3) The augmentation folder contains all the logic for augmenting the Jigsaw Civil Comments dataset using FastText and Back Translation. For translation purposes, we used Google Translate. One should look at all the Jupyter Notebooks to understand where the data is manipulated and stored
4) We have an evaluation folder which evaluates the test set using all the datasets we used for our project.
5) The models contains all the models we used for our experiments. Follow this path to access all the logic we used. Following this path: models/Super_BERT. All the *.py files are all the binary label classification models we used for our experiments. We also contain a replication folder, where we replicate some of the models of this study: [A benchmark for toxic comment classification on Civil Comments dataset](https://paperswithcode.com/paper/a-benchmark-for-toxic-comment-classification#code)
6) The multi_label folder, contains another folder named ```multi_label_implementations```. Inside, you have all the implementations of our multi label classifiers. This is the full path (models/Super_BERT/multi_label/multi_label_implementations)
7) Our Max Pooling architecture is defined inside this file: ```models/Super_BERT/multi_label/multi_label_implementations/model_skeleton_multilabel_v3.py```
