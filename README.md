# Pwml
`Pwml` stands for `P`ython `W`rappers for `M`achine `L`earning

[![Quality Gate Status](https://sonarcloud.io/api/project_badges/measure?project=braibaud_pwml&metric=alert_status)](https://sonarcloud.io/dashboard?id=braibaud_pwml) [![PyPI version](https://badge.fury.io/py/Pwml.svg)](https://badge.fury.io/py/Pwml)



## Installation

* The `Pwml` package is published at https://pypi.org/project/Pwml/
* Install package: `pip install Pwml`



## Samples

#### Hierarchical Model Training

The hierarchical model is basically defined like follow.

```python
from Pwml.classifiers import hierarchical as hc
from Pwml.classifiers import features as fe

# (...)

model = hc.HierarchicalClassifierModel(
    model_name=model_name,
    experiment_name=experiment_name,
    input_features=[
        fe.InputFeature(feature_name='Style', feature_type='text'),
        fe.InputFeature(feature_name='Gender', feature_type='text'),
        fe.InputFeature(feature_name='Brand', feature_type='text'),
        fe.InputFeature(feature_name='Category', feature_type='text') ],
    output_feature_hierarchy=fe.OutputFeature(
        feature_name='Division',
        child_feature=fe.OutputFeature(feature_name='Class')))
```

It will fit `n+1` models, where `n` is the number of distinct `Division` values:

* 1 model is fitted to handle the `Division` value. 
* For each `Division` value a specific `model` is fitted to handle the possible `Class` values available under that `Division`.

Each `model` type is selected and fine-tuned separately. 

#### Inference

This repository contains a [sample solution](https://github.com/braibaud/pwml/blob/master/samples/modelhosting.py) implemented as a `web-service` using [Flask](https://github.com/pallets/flask), [Flask-Restful](https://github.com/flask-restful/flask-restful), [Flask-Cors](https://github.com/corydolphin/flask-cors) and [Flask-Wtf](https://github.com/lepture/flask-wtf).

The web-service allows loading one or more pre-trained models and making a classification prediction based on a given sample along with its input features.



