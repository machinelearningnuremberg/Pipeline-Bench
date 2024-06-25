import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from ConfigSpace.conditions import InCondition

import importlib
from sklearn.pipeline import Pipeline

import numpy as np

def get_search_space():
    cs = CS.ConfigurationSpace("TPOTClassificationPipeline")

    # Define preprocessor choices
    preprocessor = CSH.CategoricalHyperparameter(
        "preprocessor", choices=[
            "None",
            "sklearn.preprocessing.Binarizer",
            "sklearn.decomposition.FastICA",
            "sklearn.cluster.FeatureAgglomeration",
            "sklearn.preprocessing.MaxAbsScaler",
            "sklearn.preprocessing.MinMaxScaler",
            "sklearn.preprocessing.Normalizer",
            "sklearn.kernel_approximation.Nystroem",
            "sklearn.decomposition.PCA",
            "sklearn.preprocessing.PolynomialFeatures",
            "sklearn.kernel_approximation.RBFSampler",
            "sklearn.preprocessing.RobustScaler",
            "sklearn.preprocessing.StandardScaler",
            "tpot.builtins.ZeroCount",
            "tpot.builtins.OneHotEncoder"
        ]
    )
    cs.add_hyperparameters([preprocessor])
    # Define HPs for each preprocessor
    # sklearn.preprocessing.Binarizer
    threshold = CSH.UniformFloatHyperparameter(
        "sklearn.preprocessing.Binarizer:threshold", lower=0.0, upper=1.0, default_value=0.5
    )
    cs.add_hyperparameters([threshold])
    cs.add_condition(InCondition(child=threshold, parent=preprocessor, values=["sklearn.preprocessing.Binarizer"]))
    # sklearn.decomposition.FastICA
    tol = CSH.UniformFloatHyperparameter(
        "sklearn.decomposition.FastICA:tol", lower=0.0, upper=1.0, default_value=0.0
    )
    cs.add_hyperparameters([tol])
    cs.add_condition(InCondition(child=tol, parent=preprocessor, values=["sklearn.decomposition.FastICA"]))
    # sklearn.cluster.FeatureAgglomeration
    linkage = CSH.CategoricalHyperparameter(
        "sklearn.cluster.FeatureAgglomeration:linkage", choices=["ward", "complete", "average"]
    )
    metric = CSH.CategoricalHyperparameter(
        "sklearn.cluster.FeatureAgglomeration:metric", choices=["euclidean", "l1", "l2", "manhattan", "cosine"]
    )
    cs.add_hyperparameters([linkage, metric])
    cs.add_condition(InCondition(child=linkage, parent=preprocessor, values=["sklearn.cluster.FeatureAgglomeration"]))
    cs.add_condition(InCondition(child=metric, parent=preprocessor, values=["sklearn.cluster.FeatureAgglomeration"]))
    # sklearn.preprocessing.Normalizer
    norm = CSH.CategoricalHyperparameter(
        "sklearn.preprocessing.Normalizer:norm", choices=["l1", "l2", "max"]
    )
    cs.add_hyperparameters([norm])
    cs.add_condition(InCondition(child=norm, parent=preprocessor, values=["sklearn.preprocessing.Normalizer"]))
    # sklearn.kernel_approximation.Nystroem
    kernel = CSH.CategoricalHyperparameter(
        "sklearn.kernel_approximation.Nystroem:kernel", choices=["rbf", "cosine", "chi2", "laplacian", "polynomial", "poly", "linear", "additive_chi2", "sigmoid"]
    )
    gamma = CSH.UniformFloatHyperparameter(
        "sklearn.kernel_approximation.Nystroem:gamma", lower=0.0, upper=1.0, default_value=0.5
    )
    n_components = CSH.UniformIntegerHyperparameter(
        "sklearn.kernel_approximation.Nystroem:n_components", lower=1, upper=10, default_value=5
    )
    cs.add_hyperparameters([kernel, gamma, n_components])
    cs.add_condition(InCondition(child=kernel, parent=preprocessor, values=["sklearn.kernel_approximation.Nystroem"]))
    cs.add_condition(InCondition(child=gamma, parent=preprocessor, values=["sklearn.kernel_approximation.Nystroem"]))
    cs.add_condition(InCondition(child=n_components, parent=preprocessor, values=["sklearn.kernel_approximation.Nystroem"]))
    # sklearn.decomposition.PCA
    svd_solver = CSH.CategoricalHyperparameter(
        "sklearn.decomposition.PCA:svd_solver", choices=["randomized"]
    )
    iterated_power = CSH.UniformIntegerHyperparameter(
        "sklearn.decomposition.PCA:iterated_power", lower=1, upper=10, default_value=5
    )
    cs.add_hyperparameters([svd_solver, iterated_power])
    cs.add_condition(InCondition(child=svd_solver, parent=preprocessor, values=["sklearn.decomposition.PCA"]))
    cs.add_condition(InCondition(child=iterated_power, parent=preprocessor, values=["sklearn.decomposition.PCA"]))
    # sklearn.preprocessing.PolynomialFeatures
    degree = CSH.Constant(
        "sklearn.preprocessing.PolynomialFeatures:degree", value=2
    )
    include_bias = CSH.CategoricalHyperparameter(
        "sklearn.preprocessing.PolynomialFeatures:include_bias", choices=[False]
    )
    interaction_only = CSH.CategoricalHyperparameter(
        "sklearn.preprocessing.PolynomialFeatures:interaction_only", choices=[False]
    )
    cs.add_hyperparameters([degree, include_bias, interaction_only])
    cs.add_condition(InCondition(child=degree, parent=preprocessor, values=["sklearn.preprocessing.PolynomialFeatures"]))
    cs.add_condition(InCondition(child=include_bias, parent=preprocessor, values=["sklearn.preprocessing.PolynomialFeatures"]))
    cs.add_condition(InCondition(child=interaction_only, parent=preprocessor, values=["sklearn.preprocessing.PolynomialFeatures"]))
    # sklearn.kernel_approximation.RBFSampler
    gamma = CSH.UniformFloatHyperparameter(
        "sklearn.kernel_approximation.RBFSampler:gamma", lower=0.0, upper=1.0, default_value=0.5
    )
    cs.add_hyperparameters([gamma])
    cs.add_condition(InCondition(child=gamma, parent=preprocessor, values=["sklearn.kernel_approximation.RBFSampler"]))
    # tpot.builtins.OneHotEncoder
    minimum_fraction = CSH.UniformFloatHyperparameter(
        "tpot.builtins.OneHotEncoder:minimum_fraction", lower=0.05, upper=0.25, default_value=0.15
    )
    sparse = CSH.CategoricalHyperparameter(
        "tpot.builtins.OneHotEncoder:sparse", choices=[False]
    )
    threshold = CSH.Constant(
        "tpot.builtins.OneHotEncoder:threshold", value=10
    )
    cs.add_hyperparameters([minimum_fraction, sparse, threshold])
    cs.add_condition(InCondition(child=minimum_fraction, parent=preprocessor, values=["tpot.builtins.OneHotEncoder"]))
    cs.add_condition(InCondition(child=sparse, parent=preprocessor, values=["tpot.builtins.OneHotEncoder"]))
    cs.add_condition(InCondition(child=threshold, parent=preprocessor, values=["tpot.builtins.OneHotEncoder"]))
                     
    # Define selector choices
    selector = CSH.CategoricalHyperparameter(
        "selector", choices=[
            "None",
            "sklearn.feature_selection.SelectFwe",
            "sklearn.feature_selection.SelectPercentile",
            "sklearn.feature_selection.VarianceThreshold",
            "sklearn.feature_selection.RFE",
            "sklearn.feature_selection.SelectFromModel"
        ]
    )
    cs.add_hyperparameters([selector])
    # Define HPs for each selector
    # sklearn.feature_selection.SelectFwe
    alpha = CSH.UniformFloatHyperparameter(
        "sklearn.feature_selection.SelectFwe:alpha", lower=0.0, upper=0.05, default_value=0.025
    )
    # score_func = CSH.Constant(
    #     "sklearn.feature_selection.SelectFwe:score_func", value="None"
    # )
    # cs.add_hyperparameters([alpha, score_func])
    cs.add_hyperparameters([alpha])
    cs.add_condition(InCondition(child=alpha, parent=selector, values=["sklearn.feature_selection.SelectFwe"]))
    # cs.add_condition(InCondition(child=score_func, parent=selector, values=["sklearn.feature_selection.SelectFwe"]))
    # sklearn.feature_selection.SelectPercentile
    percentile = CSH.UniformIntegerHyperparameter(
        "sklearn.feature_selection.SelectPercentile:percentile", lower=1, upper=100, default_value=50
    )
    # score_func = CSH.Constant(
    #     "sklearn.feature_selection.SelectPercentile:score_func", value="None"
    # )
    # cs.add_hyperparameters([percentile, score_func])
    cs.add_hyperparameters([percentile])
    cs.add_condition(InCondition(child=percentile, parent=selector, values=["sklearn.feature_selection.SelectPercentile"]))
    # cs.add_condition(InCondition(child=score_func, parent=selector, values=["sklearn.feature_selection.SelectPercentile"]))
    # sklearn.feature_selection.VarianceThreshold
    threshold = CSH.UniformFloatHyperparameter(
        "sklearn.feature_selection.VarianceThreshold:threshold", lower=0.0001, upper=0.2, default_value=0.1
    )
    cs.add_hyperparameters([threshold])
    cs.add_condition(InCondition(child=threshold, parent=selector, values=["sklearn.feature_selection.VarianceThreshold"]))
    # sklearn.feature_selection.RFE
    step = CSH.UniformFloatHyperparameter(
        "sklearn.feature_selection.RFE:step", lower=0.05, upper=1.0, default_value=0.525
    )
    estimator = CSH.CategoricalHyperparameter(
        "sklearn.feature_selection.RFE:estimator", choices=["sklearn.ensemble.ExtraTreesClassifier"]
    )
    cs.add_hyperparameters([step, estimator])
    cs.add_condition(InCondition(child=step, parent=selector, values=["sklearn.feature_selection.RFE"]))
    cs.add_condition(InCondition(child=estimator, parent=selector, values=["sklearn.feature_selection.RFE"]))
    n_estimators = CSH.Constant(
        "sklearn.feature_selection.RFE:estimator:n_estimators", value=100
    )
    criterion = CSH.CategoricalHyperparameter(
        "sklearn.feature_selection.RFE:estimator:criterion", choices=["gini", "entropy"]
    )
    max_features = CSH.UniformFloatHyperparameter(
        "sklearn.feature_selection.RFE:estimator:max_features", lower=0.05, upper=1.0, default_value=0.525
    )
    cs.add_hyperparameters([n_estimators, criterion, max_features])
    cs.add_condition(InCondition(child=n_estimators, parent=estimator, values=["sklearn.ensemble.ExtraTreesClassifier"]))
    cs.add_condition(InCondition(child=criterion, parent=estimator, values=["sklearn.ensemble.ExtraTreesClassifier"]))
    cs.add_condition(InCondition(child=max_features, parent=estimator, values=["sklearn.ensemble.ExtraTreesClassifier"]))
    # sklearn.feature_selection.SelectFromModel
    threshold = CSH.UniformFloatHyperparameter(
        "sklearn.feature_selection.SelectFromModel:threshold", lower=0.0, upper=1.0, default_value=0.5
    )
    estimator = CSH.CategoricalHyperparameter(
        "sklearn.feature_selection.SelectFromModel:estimator", choices=["sklearn.ensemble.ExtraTreesClassifier"]
    )
    cs.add_hyperparameters([threshold, estimator])
    cs.add_condition(InCondition(child=threshold, parent=selector, values=["sklearn.feature_selection.SelectFromModel"]))
    cs.add_condition(InCondition(child=estimator, parent=selector, values=["sklearn.feature_selection.SelectFromModel"]))
    n_estimators = CSH.Constant(
        "sklearn.feature_selection.SelectFromModel:estimator:n_estimators", value=100
    )
    criterion = CSH.CategoricalHyperparameter(
        "sklearn.feature_selection.SelectFromModel:estimator:criterion", choices=["gini", "entropy"]
    )
    max_features = CSH.UniformFloatHyperparameter(
        "sklearn.feature_selection.SelectFromModel:estimator:max_features", lower=0.05, upper=1.0, default_value=0.525
    )
    cs.add_hyperparameters([n_estimators, criterion, max_features])
    cs.add_condition(InCondition(child=n_estimators, parent=estimator, values=["sklearn.ensemble.ExtraTreesClassifier"]))
    cs.add_condition(InCondition(child=criterion, parent=estimator, values=["sklearn.ensemble.ExtraTreesClassifier"]))
    cs.add_condition(InCondition(child=max_features, parent=estimator, values=["sklearn.ensemble.ExtraTreesClassifier"]))

    # Define classifier choices
    classifier = CSH.CategoricalHyperparameter(
        "classifier", choices=[
            "sklearn.naive_bayes.GaussianNB",
            "sklearn.naive_bayes.BernoulliNB",
            "sklearn.naive_bayes.MultinomialNB",
            "sklearn.tree.DecisionTreeClassifier",
            "sklearn.ensemble.ExtraTreesClassifier",
            "sklearn.ensemble.RandomForestClassifier",
            "sklearn.ensemble.GradientBoostingClassifier",
            "sklearn.neighbors.KNeighborsClassifier",
            # "sklearn.svm.LinearSVC",
            "sklearn.linear_model.LogisticRegression",
            "xgboost.XGBClassifier",
            "sklearn.linear_model.SGDClassifier",
            "sklearn.neural_network.MLPClassifier"
        ]
    )
    cs.add_hyperparameters([classifier])
    # Define HPs for each classifier
    # sklearn.naive_bayes.BernoulliNB
    alpha = CSH.UniformFloatHyperparameter(
        "sklearn.naive_bayes.BernoulliNB:alpha", lower=1e-3, upper=100.0, default_value=50.0
    )
    fit_prior = CSH.CategoricalHyperparameter(
        "sklearn.naive_bayes.BernoulliNB:fit_prior", choices=[True, False]
    )
    cs.add_hyperparameters([alpha, fit_prior])
    cs.add_condition(InCondition(child=alpha, parent=classifier, values=["sklearn.naive_bayes.BernoulliNB"]))
    cs.add_condition(InCondition(child=fit_prior, parent=classifier, values=["sklearn.naive_bayes.BernoulliNB"]))
    # sklearn.naive_bayes.MultinomialNB
    alpha = CSH.UniformFloatHyperparameter(
        "sklearn.naive_bayes.MultinomialNB:alpha", lower=1e-3, upper=100.0, default_value=50.0
    )
    fit_prior = CSH.CategoricalHyperparameter(
        "sklearn.naive_bayes.MultinomialNB:fit_prior", choices=[True, False]
    )
    cs.add_hyperparameters([alpha, fit_prior])
    cs.add_condition(InCondition(child=alpha, parent=classifier, values=["sklearn.naive_bayes.MultinomialNB"]))
    cs.add_condition(InCondition(child=fit_prior, parent=classifier, values=["sklearn.naive_bayes.MultinomialNB"]))
    # sklearn.tree.DecisionTreeClassifier
    criterion = CSH.CategoricalHyperparameter(
        "sklearn.tree.DecisionTreeClassifier:criterion", choices=["gini", "entropy"]
    )
    max_depth = CSH.UniformIntegerHyperparameter(
        "sklearn.tree.DecisionTreeClassifier:max_depth", lower=1, upper=10, default_value=5
    )
    min_samples_split = CSH.UniformIntegerHyperparameter(
        "sklearn.tree.DecisionTreeClassifier:min_samples_split", lower=2, upper=20, default_value=11
    )
    min_samples_leaf = CSH.UniformIntegerHyperparameter(
        "sklearn.tree.DecisionTreeClassifier:min_samples_leaf", lower=1, upper=20, default_value=11
    )
    cs.add_hyperparameters([criterion, max_depth, min_samples_split, min_samples_leaf])
    cs.add_condition(InCondition(child=criterion, parent=classifier, values=["sklearn.tree.DecisionTreeClassifier"]))
    cs.add_condition(InCondition(child=max_depth, parent=classifier, values=["sklearn.tree.DecisionTreeClassifier"]))
    cs.add_condition(InCondition(child=min_samples_split, parent=classifier, values=["sklearn.tree.DecisionTreeClassifier"]))
    cs.add_condition(InCondition(child=min_samples_leaf, parent=classifier, values=["sklearn.tree.DecisionTreeClassifier"]))
    # sklearn.ensemble.ExtraTreesClassifier
    n_estimators = CSH.Constant(
        "sklearn.ensemble.ExtraTreesClassifier:n_estimators", value=100
    )
    criterion = CSH.CategoricalHyperparameter(
        "sklearn.ensemble.ExtraTreesClassifier:criterion", choices=["gini", "entropy"]
    )
    max_features = CSH.UniformFloatHyperparameter(
        "sklearn.ensemble.ExtraTreesClassifier:max_features", lower=0.05, upper=1.0, default_value=0.525
    )
    min_samples_split = CSH.UniformIntegerHyperparameter(
        "sklearn.ensemble.ExtraTreesClassifier:min_samples_split", lower=2, upper=20, default_value=11
    )
    min_samples_leaf = CSH.UniformIntegerHyperparameter(
        "sklearn.ensemble.ExtraTreesClassifier:min_samples_leaf", lower=1, upper=20, default_value=11
    )
    bootstrap = CSH.CategoricalHyperparameter(
        "sklearn.ensemble.ExtraTreesClassifier:bootstrap", choices=[True, False]
    )
    cs.add_hyperparameters([n_estimators, criterion, max_features, min_samples_split, min_samples_leaf, bootstrap])
    cs.add_condition(InCondition(child=n_estimators, parent=classifier, values=["sklearn.ensemble.ExtraTreesClassifier"]))
    cs.add_condition(InCondition(child=criterion, parent=classifier, values=["sklearn.ensemble.ExtraTreesClassifier"]))
    cs.add_condition(InCondition(child=max_features, parent=classifier, values=["sklearn.ensemble.ExtraTreesClassifier"]))
    cs.add_condition(InCondition(child=min_samples_split, parent=classifier, values=["sklearn.ensemble.ExtraTreesClassifier"]))
    cs.add_condition(InCondition(child=min_samples_leaf, parent=classifier, values=["sklearn.ensemble.ExtraTreesClassifier"]))
    cs.add_condition(InCondition(child=bootstrap, parent=classifier, values=["sklearn.ensemble.ExtraTreesClassifier"]))
    # sklearn.ensemble.RandomForestClassifier
    n_estimators = CSH.Constant(
        "sklearn.ensemble.RandomForestClassifier:n_estimators", value=100
    )
    criterion = CSH.CategoricalHyperparameter(
        "sklearn.ensemble.RandomForestClassifier:criterion", choices=["gini", "entropy"]
    )
    max_features = CSH.UniformFloatHyperparameter(
        "sklearn.ensemble.RandomForestClassifier:max_features", lower=0.05, upper=1.0, default_value=0.525
    )
    min_samples_split = CSH.UniformIntegerHyperparameter(
        "sklearn.ensemble.RandomForestClassifier:min_samples_split", lower=2, upper=20, default_value=11
    )
    min_samples_leaf = CSH.UniformIntegerHyperparameter(
        "sklearn.ensemble.RandomForestClassifier:min_samples_leaf", lower=1, upper=20, default_value=11
    )
    bootstrap = CSH.CategoricalHyperparameter(
        "sklearn.ensemble.RandomForestClassifier:bootstrap", choices=[True, False]
    )
    cs.add_hyperparameters([n_estimators, criterion, max_features, min_samples_split, min_samples_leaf, bootstrap])
    cs.add_condition(InCondition(child=n_estimators, parent=classifier, values=["sklearn.ensemble.RandomForestClassifier"]))
    cs.add_condition(InCondition(child=criterion, parent=classifier, values=["sklearn.ensemble.RandomForestClassifier"]))
    cs.add_condition(InCondition(child=max_features, parent=classifier, values=["sklearn.ensemble.RandomForestClassifier"]))
    cs.add_condition(InCondition(child=min_samples_split, parent=classifier, values=["sklearn.ensemble.RandomForestClassifier"]))
    cs.add_condition(InCondition(child=min_samples_leaf, parent=classifier, values=["sklearn.ensemble.RandomForestClassifier"]))
    cs.add_condition(InCondition(child=bootstrap, parent=classifier, values=["sklearn.ensemble.RandomForestClassifier"]))
    # sklearn.ensemble.GradientBoostingClassifier
    n_estimators = CSH.Constant(
        "sklearn.ensemble.GradientBoostingClassifier:n_estimators", value=100
    )
    learning_rate = CSH.UniformFloatHyperparameter(
        "sklearn.ensemble.GradientBoostingClassifier:learning_rate", lower=1e-3, upper=1.0, default_value=0.5
    )
    max_depth = CSH.UniformIntegerHyperparameter(
        "sklearn.ensemble.GradientBoostingClassifier:max_depth", lower=1, upper=10, default_value=5
    )
    min_samples_split = CSH.UniformIntegerHyperparameter(
        "sklearn.ensemble.GradientBoostingClassifier:min_samples_split", lower=2, upper=20, default_value=11
    )
    min_samples_leaf = CSH.UniformIntegerHyperparameter(
        "sklearn.ensemble.GradientBoostingClassifier:min_samples_leaf", lower=1, upper=20, default_value=11
    )
    subsample = CSH.UniformFloatHyperparameter(
        "sklearn.ensemble.GradientBoostingClassifier:subsample", lower=0.05, upper=1.0, default_value=0.525
    )
    max_features = CSH.UniformFloatHyperparameter(
        "sklearn.ensemble.GradientBoostingClassifier:max_features", lower=0.05, upper=1.0, default_value=0.525
    )
    cs.add_hyperparameters([n_estimators, learning_rate, max_depth, min_samples_split, min_samples_leaf, subsample, max_features])
    cs.add_condition(InCondition(child=n_estimators, parent=classifier, values=["sklearn.ensemble.GradientBoostingClassifier"]))
    cs.add_condition(InCondition(child=learning_rate, parent=classifier, values=["sklearn.ensemble.GradientBoostingClassifier"]))
    cs.add_condition(InCondition(child=max_depth, parent=classifier, values=["sklearn.ensemble.GradientBoostingClassifier"]))
    cs.add_condition(InCondition(child=min_samples_split, parent=classifier, values=["sklearn.ensemble.GradientBoostingClassifier"]))
    cs.add_condition(InCondition(child=min_samples_leaf, parent=classifier, values=["sklearn.ensemble.GradientBoostingClassifier"]))
    cs.add_condition(InCondition(child=subsample, parent=classifier, values=["sklearn.ensemble.GradientBoostingClassifier"]))
    cs.add_condition(InCondition(child=max_features, parent=classifier, values=["sklearn.ensemble.GradientBoostingClassifier"]))
    # sklearn.neighbors.KNeighborsClassifier
    n_neighbors = CSH.UniformIntegerHyperparameter(
        "sklearn.neighbors.KNeighborsClassifier:n_neighbors", lower=1, upper=100, default_value=50
    )
    weights = CSH.CategoricalHyperparameter(
        "sklearn.neighbors.KNeighborsClassifier:weights", choices=["uniform", "distance"]
    )
    p = CSH.CategoricalHyperparameter(
        "sklearn.neighbors.KNeighborsClassifier:p", choices=[1, 2]
    )
    cs.add_hyperparameters([n_neighbors, weights, p])
    cs.add_condition(InCondition(child=n_neighbors, parent=classifier, values=["sklearn.neighbors.KNeighborsClassifier"]))
    cs.add_condition(InCondition(child=weights, parent=classifier, values=["sklearn.neighbors.KNeighborsClassifier"]))
    cs.add_condition(InCondition(child=p, parent=classifier, values=["sklearn.neighbors.KNeighborsClassifier"]))
    # # sklearn.svm.LinearSVC
    # penalty = CSH.CategoricalHyperparameter(
    #     "sklearn.svm.LinearSVC:penalty", choices=["l1", "l2"]
    # )
    # loss = CSH.CategoricalHyperparameter(
    #     "sklearn.svm.LinearSVC:loss", choices=["hinge", "squared_hinge"]
    # )
    # dual = CSH.CategoricalHyperparameter(
    #     "sklearn.svm.LinearSVC:dual", choices=[True, False]
    # )
    # tol = CSH.UniformFloatHyperparameter(
    #     "sklearn.svm.LinearSVC:tol", lower=1e-5, upper=0.1, default_value=0.05005
    # )
    # C = CSH.UniformFloatHyperparameter(
    #     "sklearn.svm.LinearSVC:C", lower=1e-4, upper=25.0, default_value=12.525
    # )
    # # solver = CSH.Constant(
    # #     "sklearn.svm.LinearSVC:solver", value="liblinear"
    # # )
    # cs.add_hyperparameters([penalty, loss, dual, tol, C]) #, solver])
    # cs.add_condition(InCondition(child=penalty, parent=classifier, values=["sklearn.svm.LinearSVC"]))
    # cs.add_condition(InCondition(child=loss, parent=classifier, values=["sklearn.svm.LinearSVC"]))
    # cs.add_condition(InCondition(child=dual, parent=classifier, values=["sklearn.svm.LinearSVC"]))
    # cs.add_condition(InCondition(child=tol, parent=classifier, values=["sklearn.svm.LinearSVC"]))
    # cs.add_condition(InCondition(child=C, parent=classifier, values=["sklearn.svm.LinearSVC"]))
    # # cs.add_condition(InCondition(child=solver, parent=classifier, values=["sklearn.svm.LinearSVC"]))
    # sklearn.linear_model.LogisticRegression
    penalty = CSH.CategoricalHyperparameter(
        "sklearn.linear_model.LogisticRegression:penalty", choices=["l1", "l2"]
    )
    C = CSH.UniformFloatHyperparameter(
        "sklearn.linear_model.LogisticRegression:C", lower=1e-4, upper=25.0, default_value=12.525
    )
    dual = CSH.CategoricalHyperparameter(
        "sklearn.linear_model.LogisticRegression:dual", choices=[True, False]
    )
    solver = CSH.Constant(
        "sklearn.linear_model.LogisticRegression:solver", value="liblinear"
    )
    cs.add_hyperparameters([penalty, C, dual, solver])
    cs.add_condition(InCondition(child=penalty, parent=classifier, values=["sklearn.linear_model.LogisticRegression"]))
    cs.add_condition(InCondition(child=C, parent=classifier, values=["sklearn.linear_model.LogisticRegression"]))
    cs.add_condition(InCondition(child=dual, parent=classifier, values=["sklearn.linear_model.LogisticRegression"]))
    cs.add_condition(InCondition(child=solver, parent=classifier, values=["sklearn.linear_model.LogisticRegression"]))
    # xgboost.XGBClassifier
    n_estimators = CSH.Constant(
        "xgboost.XGBClassifier:n_estimators", value=100
    )
    max_depth = CSH.UniformIntegerHyperparameter(
        "xgboost.XGBClassifier:max_depth", lower=1, upper=10, default_value=5
    )
    learning_rate = CSH.UniformFloatHyperparameter(
        "xgboost.XGBClassifier:learning_rate", lower=1e-3, upper=1.0, default_value=0.5
    )
    subsample = CSH.UniformFloatHyperparameter(
        "xgboost.XGBClassifier:subsample", lower=0.05, upper=1.0, default_value=0.525
    )
    min_child_weight = CSH.UniformIntegerHyperparameter(
        "xgboost.XGBClassifier:min_child_weight", lower=1, upper=20, default_value=11
    )
    n_jobs = CSH.CategoricalHyperparameter(
        "xgboost.XGBClassifier:n_jobs", choices=[1]
    )
    verbosity = CSH.CategoricalHyperparameter(
        "xgboost.XGBClassifier:verbosity", choices=[0]
    )
    cs.add_hyperparameters([n_estimators, max_depth, learning_rate, subsample, min_child_weight, n_jobs, verbosity])
    cs.add_condition(InCondition(child=n_estimators, parent=classifier, values=["xgboost.XGBClassifier"]))
    cs.add_condition(InCondition(child=max_depth, parent=classifier, values=["xgboost.XGBClassifier"]))
    cs.add_condition(InCondition(child=learning_rate, parent=classifier, values=["xgboost.XGBClassifier"]))
    cs.add_condition(InCondition(child=subsample, parent=classifier, values=["xgboost.XGBClassifier"]))
    cs.add_condition(InCondition(child=min_child_weight, parent=classifier, values=["xgboost.XGBClassifier"]))
    cs.add_condition(InCondition(child=n_jobs, parent=classifier, values=["xgboost.XGBClassifier"]))
    cs.add_condition(InCondition(child=verbosity, parent=classifier, values=["xgboost.XGBClassifier"]))
    # sklearn.linear_model.SGDClassifier
    loss = CSH.CategoricalHyperparameter(
        "sklearn.linear_model.SGDClassifier:loss", choices=["log_loss", "modified_huber"]
    )
    penalty = CSH.CategoricalHyperparameter(
        "sklearn.linear_model.SGDClassifier:penalty", choices=["elasticnet"]
    )
    alpha = CSH.UniformFloatHyperparameter(
        "sklearn.linear_model.SGDClassifier:alpha", lower=0.0, upper=0.01, default_value=0.005
    )
    learning_rate = CSH.CategoricalHyperparameter(
        "sklearn.linear_model.SGDClassifier:learning_rate", choices=["invscaling", "constant"]
    )
    fit_intercept = CSH.CategoricalHyperparameter(
        "sklearn.linear_model.SGDClassifier:fit_intercept", choices=[True, False]
    )
    l1_ratio = CSH.UniformFloatHyperparameter(
        "sklearn.linear_model.SGDClassifier:l1_ratio", lower=0.0, upper=1.0, default_value=0.5
    )
    eta0 = CSH.UniformFloatHyperparameter(
        "sklearn.linear_model.SGDClassifier:eta0", lower=0.01, upper=1.0, default_value=0.505
    )
    power_t = CSH.UniformFloatHyperparameter(
        "sklearn.linear_model.SGDClassifier:power_t", lower=0.0, upper=100.0, default_value=50.0
    )
    cs.add_hyperparameters([loss, penalty, alpha, learning_rate, fit_intercept, l1_ratio, eta0, power_t])
    cs.add_condition(InCondition(child=loss, parent=classifier, values=["sklearn.linear_model.SGDClassifier"]))
    cs.add_condition(InCondition(child=penalty, parent=classifier, values=["sklearn.linear_model.SGDClassifier"]))
    cs.add_condition(InCondition(child=alpha, parent=classifier, values=["sklearn.linear_model.SGDClassifier"]))
    cs.add_condition(InCondition(child=learning_rate, parent=classifier, values=["sklearn.linear_model.SGDClassifier"]))
    cs.add_condition(InCondition(child=fit_intercept, parent=classifier, values=["sklearn.linear_model.SGDClassifier"]))
    cs.add_condition(InCondition(child=l1_ratio, parent=classifier, values=["sklearn.linear_model.SGDClassifier"]))
    cs.add_condition(InCondition(child=eta0, parent=classifier, values=["sklearn.linear_model.SGDClassifier"]))
    cs.add_condition(InCondition(child=power_t, parent=classifier, values=["sklearn.linear_model.SGDClassifier"]))
    # sklearn.neural_network.MLPClassifier
    alpha = CSH.UniformFloatHyperparameter(
        "sklearn.neural_network.MLPClassifier:alpha", lower=1e-4, upper=0.1, default_value=0.05
    )
    learning_rate_init = CSH.UniformFloatHyperparameter(
        "sklearn.neural_network.MLPClassifier:learning_rate_init", lower=0.0, upper=1.0, default_value=0.5
    )
    cs.add_hyperparameters([alpha, learning_rate_init])
    cs.add_condition(InCondition(child=alpha, parent=classifier, values=["sklearn.neural_network.MLPClassifier"]))
    cs.add_condition(InCondition(child=learning_rate_init, parent=classifier, values=["sklearn.neural_network.MLPClassifier"]))

    return cs

def create_component(class_path, **params):
    module_path, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    
    # Preprocess parameters to handle nested object creation
    processed_params = {}
    nested_params = {}
    for key, value in params.items():
        if ':' in key:
            nested_param_key, sub_key = key.split(':', 1)
            if nested_param_key not in nested_params:
                nested_params[nested_param_key] = {}
            nested_params[nested_param_key][sub_key] = value
        else:
            processed_params[key] = value

    # Create nested objects if necessary
    for nested_key, sub_params in nested_params.items():
        # Recursively create the nested object
        processed_params[nested_key] = create_component(processed_params['estimator'], **sub_params)

    return cls(**processed_params)


def extract_params(config, prefix):
    return {key[len(prefix):]: value for key, value in config.items() if key.startswith(prefix)}

def configure_pipeline(config):
    steps = []

    config = dict(config)

    cast_none = lambda dictionary: {k: None if v == 'None' else v for k, v in dictionary.items()}
    # Configure the preprocessor
    if 'preprocessor' in config and config['preprocessor'] != 'None':
        preprocessor_class = config['preprocessor']
        preprocessor_params = cast_none(extract_params(config, f"{preprocessor_class}:"))
        preprocessor = create_component(preprocessor_class, **preprocessor_params)
        steps.append(('preprocessor', preprocessor))
    
    # Configure the selector
    if 'selector' in config and config['selector'] != 'None':
        selector_class = config['selector']
        selector_params = cast_none(extract_params(config, f"{selector_class}:"))
        selector = create_component(selector_class, **selector_params)
        steps.append(('selector', selector))
    
    # Configure the classifier
    classifier_class = config['classifier']
    classifier_params = cast_none(extract_params(config, f"{classifier_class}:"))
    classifier = create_component(classifier_class, **classifier_params)
    steps.append(('classifier', classifier))
    
    # Create and return the pipeline
    return Pipeline(steps)


def config_sampler(
    global_seed_gen, config_id: int = None
) -> tuple[CS.Configuration, Pipeline, int]:

    if config_id is None:
        random_state = next(global_seed_gen)
    else:
        for _ in range(config_id):
            random_state = next(global_seed_gen)
    
    search_space = get_search_space()
    search_space.seed(random_state)

    config = search_space.sample_configuration()
    pipeline = configure_pipeline(dict(config))

    return config, pipeline, random_state
