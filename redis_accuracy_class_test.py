import yfinance
from ta_indicators import get_ta
import pandas as pd
import namegenerator

from hmmlearn.hmm import GaussianHMM, GMMHMM
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from mlxtend.feature_extraction import PrincipalComponentAnalysis
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVR
from itertools import product
import numpy as np
from sklearn.svm import SVC
from itertools import combinations
from strategy import AccuracyStrat
from strategy import setup_strategy
import os
from rq import Queue
from redis import Redis

def model_generator_2(name):
    print(name)
