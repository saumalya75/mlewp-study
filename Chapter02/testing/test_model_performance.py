import pytest
import numpy as np
import pandas as pd
from typing import Union
import sklearn
from sklearn.model_selection import train_test_split
from huggingface_hub import hf_hub_download
from sklearn.metrics import classification_report

import joblib

@pytest.fixture
def test_dataset() -> Union[np.array, np.array]:
    REPO_ID = "julien-c/wine-quality"
    # Load the dataset
    data_file = hf_hub_download(REPO_ID, "winequality-red.csv")
    # create an array of True for 2 and False otherwise
    winedf = pd.read_csv(data_file, sep=";")
    X = winedf.drop(["quality"], axis=1)
    Y = winedf["quality"]
    _, X_test, _, y_test = train_test_split(X, Y, random_state=42)
    return X_test, y_test

@pytest.fixture
def model() -> sklearn.ensemble._forest.RandomForestClassifier:
    REPO_ID = "julien-c/wine-quality"
    FILENAME = "sklearn_model.joblib"
    model = joblib.load(hf_hub_download(REPO_ID, FILENAME))
    return model


def test_model_inference_types(model, test_dataset):
    assert isinstance(model.predict(test_dataset[0]), np.ndarray)
    assert isinstance(test_dataset[0], pd.core.frame.DataFrame)
    assert isinstance(test_dataset[1], pd.core.series.Series)

def test_model_performance(model, test_dataset):
    metrics = classification_report(y_true=test_dataset[1], y_pred=model.predict(test_dataset[0]), output_dict=True)
    assert metrics['accuracy'] > 0.60
    assert metrics['weighted avg']['f1-score'] > 0.6
    assert metrics['weighted avg']['precision'] > 0.6
    assert metrics['weighted avg']['f1-score'] > 0.6
    assert metrics['weighted avg']['recall'] > 0.6