"""AI-Driven Phishing Email Detection Using NLP.

Modules
-------
config               paths, constants, hyperparameter grids
data_loader          raw CSV loading + stratified split
preprocessing        cleaning, tokenising, lemmatising
feature_engineering  TF-IDF / counts / Word2Vec / SVD / metadata
train_models         tuning, training, persistence
evaluation           metrics, plots, comparison table
explain              risk scoring + human-readable explanations
"""
