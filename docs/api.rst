API Reference
####################################

==============================================================
selecting_OOD_detector.pipeline.ood_pipeline API documentation
==============================================================

.. contents::
   :depth: 3
..

.. container::

   .. container:: section
      :name: section-intro

      Expand source code
      ::

         from collections import defaultdict
         from typing import Optional

         import pandas as pd
         from selecting_OOD_detector.pipeline.base import BasePipeline
         from selecting_OOD_detector.utils.scores_metrics import (score_dataset,
                                                                  get_ood_aucs_score_for_all_models,
                                                                  average_values_in_nested_dict,
                                                                  get_mean_stderr_annots_in_nested_dict
                                                                  )
         from selecting_OOD_detector.utils.general import check_and_convert_dfs_to_numpy
         from selecting_OOD_detector.utils.plotting import plot_heatmap, plot_scores_boxplot, plot_scores_distr
         from selecting_OOD_detector.models.novelty_estimators_info import SCORING_FUNCTIONS


         class OODPipeline(BasePipeline):
             """
             Pipeline to fit novelty estimators on in-distribution data and evaluate novelty of Out-of-Distribution (OOD)
             groups.


             Example of usage:
                 # Initialize the pipeline
                 oodpipeline = OODPipeline()

                 # Fit the pipeline on in-distribution training data and compute novelty scores for in-distribution test data
                 oodpipeline.fit(X_train= X_train, X_test=X_test)

                 # Define OOD groups and evaluate by the pipeline
                 ood_groups = {"Flu patients": X_flu, "Ventilated patients": X_vent}
                 oodpipeline.evaluate(ood_groups)

                 # Inspect AUC-ROC scores of detecting OOD groups
                 oodpipeline.get_ood_aucs_scores()
             """

             def __init__(self,
                          **kwargs):
                 """

                 Parameters
                 ----------
                 kwargs
                     model_selection: set
                         Define which models to train, e.g. {"PPCA", "LOF", "VAE"}. If selection is not provided, all available
                         models are used.
                 """
                 super().__init__(**kwargs)
                 self.in_domain_scores = defaultdict(dict)
                 self.out_domain_scores = defaultdict(dict)
                 self.feature_names = None

             def fit(self,
                     X_train,
                     X_test,
                     **kwargs):
                 """
                 Fits models on training data with n_trials different runs.  Novelty estimators from each run are stored
                 in a nested dictionary in self.novelty_estimators.
                 (E.g.: {0: {"AE": NoveltyEstimator, "PPCA": NoveltyEstimator},
                        1: {"AE": NoveltyEstimator, "PPCA": NoveltyEstimator}} )
                 Parameters
                 ----------
                 X_train: pd.DataFrame
                     Training in-distribution data. Used to fit novelty estimators.
                 X_test: pd.DataFrame
                     Test in-distribution data. Used to calculate self.in_domain_scores which are taken as base novelty scores
                     for the dataset and used for comparison against OOD groups later.
                 kwargs:
                     y_train: pd.DataFrame
                         Labels corresponding to training data.
                     n_trials: int
                         Number of trials to run.
                 """
                 y_train = kwargs.get("y_train", None)
                 n_trials = kwargs.get("n_trials", 5)

                 assert list(X_train.columns) == list(X_test.columns), "Train and test data have different features!"
                 self.feature_names = list(X_train.columns)

                 print("--- OOD Pipeline ---")
                 print("1/2 Fitting novelty estimators...")
                 self._fit(X_train=X_train, y_train=y_train, n_trials=n_trials)

                 print("2/2 Scoring in-domain data...")
                 self.in_domain_scores = self._score_in_domain(X_test)

             def evaluate_ood_groups(self,
                                     ood_groups: dict,
                                     return_averaged: bool = False):
                 """
                 Gives novelty scores to OOD groups.
                 Returns and stores dictionary of novelty scores given by each model for each sample in every OOD group.
                 If the function is called repeadetly, updates internally stored novelty scores for the OOD groups.

                 Parameters
                 ----------
                 ood_groups: dict
                     Dictionary of OOD groups. Dictionary has to contain a name of each OOD group and features in a pd.DataFrame.
                     Example: {"Flu patients": X_flu, "Ventilated patients": X_vent}

                 return_averaged: bool
                     If true, returns averaged novelty score for each sample. The shape of novelty scores given by each model
                     then corresponds to (1, n_samples).
                     Else, the shape of novelty scores given by each model is (n_trials, n_samples) where n_trials is
                     the number of trials used in the fit function.

                 Returns
                 -------
                 out_domain_scores: dict
                     Returns a dictionary of novelty scores given by each model for each sample in every OOD group.

                 """
                 assert self.novelty_estimators, "Novelty estimator dictionary is empty." \
                                                 "Please fit novelty estimators to in-distribution data before calling" \
                                                 "evaluate_ood_groups."

                 assert all([list(X_ood.columns) == self.feature_names for _, X_ood in ood_groups.items()]), \
                     "All OOD groups must have identical features to in-distribution data!"

                 out_domain_scores = self._score_out_domain(ood_groups)
                 self.out_domain_scores.update(out_domain_scores)

                 if return_averaged:
                     return average_values_in_nested_dict(out_domain_scores)

                 return out_domain_scores

             def _score_in_domain(self,
                                  X_test: pd.DataFrame):
                 """
                 Returns novelty scores for each sample in the dataset.
                 """
                 scores = score_dataset(X=X_test, models_trials_dict=self.novelty_estimators)
                 return scores

             def _score_out_domain(self,
                                   ood_groups: dict):
                 """
                 Returns novelty scores for each sample in all OOD groups.
                 """
                 out_domain_scores = dict()

                 print("Scoring OOD data:")
                 for ood_group_name, X_ood in ood_groups.items():
                     print(f"\t{ood_group_name}...", end=" ")
                     X_ood = check_and_convert_dfs_to_numpy([X_ood])[0]
                     scores = score_dataset(X=X_ood, models_trials_dict=self.novelty_estimators)
                     out_domain_scores[ood_group_name] = scores
                     print("done.")

                 return out_domain_scores

             def get_auc_scores(self,
                                ood_groups_selections: Optional[list] = None,
                                return_averaged: bool = True):
                 """
                 Computes AUC-ROC scores of OOD detection for each OOD group as compared to the in-distribution test data.
                 By default, returns scores for every group evaluated by the pipeline (evaluate_ood_groups).

                 Parameters
                 ----------
                 ood_groups_selections: Optional(list)
                     Optionally provide a selection of OOD groups for which AUC-ROC score should be returned. If no selection
                     is provided, all groups ever evaluate by the pipeline will be included.
                 return_averaged: bool
                     Indicates whether to return averaged AUC-ROC scores over n_trials run or a list of scores for every trial.

                 Returns
                 -------
                 aucs_dict_groups: dict
                     A nested dictionary that contains a name of OOD group, name of novelty estimator and either a float (if
                     averaged) or a list of AUC-ROC scores.

                 """
                 selected_ood_group = self._filter_ood_groups(ood_groups_selections)
                 aucs_dict_groups = defaultdict(lambda: defaultdict(list))

                 for ood_group_name in selected_ood_group:
                     aucs_dict_groups[ood_group_name] = get_ood_aucs_score_for_all_models(
                         ood_scores_trials_dict=self.out_domain_scores[ood_group_name],
                         test_scores_trials_dict=self.in_domain_scores,
                     )

                 if return_averaged:
                     return average_values_in_nested_dict(aucs_dict_groups)

                 return aucs_dict_groups

             def plot_auc_scores(self,
                                 ood_groups_selections: Optional[list] = None,
                                 return_averaged: bool = True,
                                 save_dir: str = None,
                                 **plot_kwargs):

                 auc_scores = self.get_auc_scores(ood_groups_selections=ood_groups_selections,
                                                  return_averaged=return_averaged)
                 if not return_averaged:
                     annots = get_mean_stderr_annots_in_nested_dict(auc_scores)
                     annots = pd.DataFrame(annots).values.T
                     auc_scores = average_values_in_nested_dict(auc_scores)
                     plot_fmt = "s"
                 else:
                     annots = True
                     plot_fmt = ".2g"

                 plot_df = pd.DataFrame(auc_scores)

                 plot_heatmap(plot_df,
                              title="AUC",
                              save_dir=save_dir,
                              annot=annots,
                              fmt=plot_fmt,
                              annot_kws={"fontsize": 9},
                              figsize=(12, 0.75 * len(plot_df.columns)),
                              **plot_kwargs,
                              )

             def plot_score_distr(self,
                                  ood_groups_selections: Optional[list] = None,
                                  save_dir=None
                                  ):
                 out_domain_scores_mean = average_values_in_nested_dict(self.out_domain_scores, axis=0, dict_level=2)
                 in_domain_scores_mean = average_values_in_nested_dict(self.in_domain_scores, axis=0, dict_level=1)

                 selected_ood_group = self._filter_ood_groups(ood_groups_selections)

                 for ood_name in selected_ood_group:
                     ood_scores = out_domain_scores_mean[ood_name]
                     save_group_name = ood_name.lower().replace(" ", "_")

                     if save_dir is not None:
                         save_dir_ = f"{save_dir}_{save_group_name}.png"
                     else:
                         save_dir_ = None

                     plot_scores_distr(scores_test=in_domain_scores_mean,
                                       scores_new=ood_scores,
                                       title=ood_name,
                                       clip_q=0.05,
                                       kind="hist",
                                       bins=30,
                                       save_dir=save_dir_,
                                       labels=SCORING_FUNCTIONS)

             def plot_box_plot(self,
                               ood_groups_selections: Optional[list] = None,
                               save_dir=None
                               ):

                 out_domain_scores_mean = average_values_in_nested_dict(self.out_domain_scores, axis=0, dict_level=2)
                 in_domain_scores_mean = average_values_in_nested_dict(self.in_domain_scores, axis=0, dict_level=1)

                 selected_ood_group = self._filter_ood_groups(ood_groups_selections)

                 for ood_name in selected_ood_group:
                     ood_scores = out_domain_scores_mean[ood_name]

                     if save_dir is not None:
                         save_dir_ = f"{save_dir}_{ood_name}.png"
                     else:
                         save_dir_ = None

                     plot_scores_boxplot(scores_test=in_domain_scores_mean,
                                         scores_new=ood_scores,
                                         show_outliers=True,
                                         title=ood_name,
                                         return_results=False,
                                         save_dir=save_dir_)

             def _filter_ood_groups(self, ood_groups_selections: Optional[list]):
                 if ood_groups_selections is not None:
                     assert all([ood_group_name in self.out_domain_scores for ood_group_name in ood_groups_selections]), \
                         "Please these OOD groups first before calculating AUC-ROC scores"
                     selected_ood_group = self.out_domain_scores.keys() & ood_groups_selections
                 else:
                     selected_ood_group = self.out_domain_scores.keys()

                 return selected_ood_group

   .. container:: section

   .. container:: section

   .. container:: section

   .. container:: section

      .. rubric:: Classes
         :name: header-classes
         :class: section-title

      ``class OODPipeline (**kwargs)``
         .. container:: desc

            Pipeline to fit novelty estimators on in-distribution data
            and evaluate novelty of Out-of-Distribution (OOD) groups.

            Example of usage: # Initialize the pipeline oodpipeline =
            OODPipeline()

            ::

               # Fit the pipeline on in-distribution training data and compute novelty scores for in-distribution test data
               oodpipeline.fit(X_train= X_train, X_test=X_test)

               # Define OOD groups and evaluate by the pipeline
               ood_groups = {"Flu patients": X_flu, "Ventilated patients": X_vent}
               oodpipeline.evaluate(ood_groups)

               # Inspect AUC-ROC scores of detecting OOD groups
               oodpipeline.get_ood_aucs_scores()

            .. rubric:: Parameters
               :name: parameters

            **``kwargs``**
               model_selection: set Define which models to train, e.g.
               {"PPCA", "LOF", "VAE"}. If selection is not provided, all
               available models are used.

         Expand source code

         ::

            class OODPipeline(BasePipeline):
                """
                Pipeline to fit novelty estimators on in-distribution data and evaluate novelty of Out-of-Distribution (OOD)
                groups.


                Example of usage:
                    # Initialize the pipeline
                    oodpipeline = OODPipeline()

                    # Fit the pipeline on in-distribution training data and compute novelty scores for in-distribution test data
                    oodpipeline.fit(X_train= X_train, X_test=X_test)

                    # Define OOD groups and evaluate by the pipeline
                    ood_groups = {"Flu patients": X_flu, "Ventilated patients": X_vent}
                    oodpipeline.evaluate(ood_groups)

                    # Inspect AUC-ROC scores of detecting OOD groups
                    oodpipeline.get_ood_aucs_scores()
                """

                def __init__(self,
                             **kwargs):
                    """

                    Parameters
                    ----------
                    kwargs
                        model_selection: set
                            Define which models to train, e.g. {"PPCA", "LOF", "VAE"}. If selection is not provided, all available
                            models are used.
                    """
                    super().__init__(**kwargs)
                    self.in_domain_scores = defaultdict(dict)
                    self.out_domain_scores = defaultdict(dict)
                    self.feature_names = None

                def fit(self,
                        X_train,
                        X_test,
                        **kwargs):
                    """
                    Fits models on training data with n_trials different runs.  Novelty estimators from each run are stored
                    in a nested dictionary in self.novelty_estimators.
                    (E.g.: {0: {"AE": NoveltyEstimator, "PPCA": NoveltyEstimator},
                           1: {"AE": NoveltyEstimator, "PPCA": NoveltyEstimator}} )
                    Parameters
                    ----------
                    X_train: pd.DataFrame
                        Training in-distribution data. Used to fit novelty estimators.
                    X_test: pd.DataFrame
                        Test in-distribution data. Used to calculate self.in_domain_scores which are taken as base novelty scores
                        for the dataset and used for comparison against OOD groups later.
                    kwargs:
                        y_train: pd.DataFrame
                            Labels corresponding to training data.
                        n_trials: int
                            Number of trials to run.
                    """
                    y_train = kwargs.get("y_train", None)
                    n_trials = kwargs.get("n_trials", 5)

                    assert list(X_train.columns) == list(X_test.columns), "Train and test data have different features!"
                    self.feature_names = list(X_train.columns)

                    print("--- OOD Pipeline ---")
                    print("1/2 Fitting novelty estimators...")
                    self._fit(X_train=X_train, y_train=y_train, n_trials=n_trials)

                    print("2/2 Scoring in-domain data...")
                    self.in_domain_scores = self._score_in_domain(X_test)

                def evaluate_ood_groups(self,
                                        ood_groups: dict,
                                        return_averaged: bool = False):
                    """
                    Gives novelty scores to OOD groups.
                    Returns and stores dictionary of novelty scores given by each model for each sample in every OOD group.
                    If the function is called repeadetly, updates internally stored novelty scores for the OOD groups.

                    Parameters
                    ----------
                    ood_groups: dict
                        Dictionary of OOD groups. Dictionary has to contain a name of each OOD group and features in a pd.DataFrame.
                        Example: {"Flu patients": X_flu, "Ventilated patients": X_vent}

                    return_averaged: bool
                        If true, returns averaged novelty score for each sample. The shape of novelty scores given by each model
                        then corresponds to (1, n_samples).
                        Else, the shape of novelty scores given by each model is (n_trials, n_samples) where n_trials is
                        the number of trials used in the fit function.

                    Returns
                    -------
                    out_domain_scores: dict
                        Returns a dictionary of novelty scores given by each model for each sample in every OOD group.

                    """
                    assert self.novelty_estimators, "Novelty estimator dictionary is empty." \
                                                    "Please fit novelty estimators to in-distribution data before calling" \
                                                    "evaluate_ood_groups."

                    assert all([list(X_ood.columns) == self.feature_names for _, X_ood in ood_groups.items()]), \
                        "All OOD groups must have identical features to in-distribution data!"

                    out_domain_scores = self._score_out_domain(ood_groups)
                    self.out_domain_scores.update(out_domain_scores)

                    if return_averaged:
                        return average_values_in_nested_dict(out_domain_scores)

                    return out_domain_scores

                def _score_in_domain(self,
                                     X_test: pd.DataFrame):
                    """
                    Returns novelty scores for each sample in the dataset.
                    """
                    scores = score_dataset(X=X_test, models_trials_dict=self.novelty_estimators)
                    return scores

                def _score_out_domain(self,
                                      ood_groups: dict):
                    """
                    Returns novelty scores for each sample in all OOD groups.
                    """
                    out_domain_scores = dict()

                    print("Scoring OOD data:")
                    for ood_group_name, X_ood in ood_groups.items():
                        print(f"\t{ood_group_name}...", end=" ")
                        X_ood = check_and_convert_dfs_to_numpy([X_ood])[0]
                        scores = score_dataset(X=X_ood, models_trials_dict=self.novelty_estimators)
                        out_domain_scores[ood_group_name] = scores
                        print("done.")

                    return out_domain_scores

                def get_auc_scores(self,
                                   ood_groups_selections: Optional[list] = None,
                                   return_averaged: bool = True):
                    """
                    Computes AUC-ROC scores of OOD detection for each OOD group as compared to the in-distribution test data.
                    By default, returns scores for every group evaluated by the pipeline (evaluate_ood_groups).

                    Parameters
                    ----------
                    ood_groups_selections: Optional(list)
                        Optionally provide a selection of OOD groups for which AUC-ROC score should be returned. If no selection
                        is provided, all groups ever evaluate by the pipeline will be included.
                    return_averaged: bool
                        Indicates whether to return averaged AUC-ROC scores over n_trials run or a list of scores for every trial.

                    Returns
                    -------
                    aucs_dict_groups: dict
                        A nested dictionary that contains a name of OOD group, name of novelty estimator and either a float (if
                        averaged) or a list of AUC-ROC scores.

                    """
                    selected_ood_group = self._filter_ood_groups(ood_groups_selections)
                    aucs_dict_groups = defaultdict(lambda: defaultdict(list))

                    for ood_group_name in selected_ood_group:
                        aucs_dict_groups[ood_group_name] = get_ood_aucs_score_for_all_models(
                            ood_scores_trials_dict=self.out_domain_scores[ood_group_name],
                            test_scores_trials_dict=self.in_domain_scores,
                        )

                    if return_averaged:
                        return average_values_in_nested_dict(aucs_dict_groups)

                    return aucs_dict_groups

                def plot_auc_scores(self,
                                    ood_groups_selections: Optional[list] = None,
                                    return_averaged: bool = True,
                                    save_dir: str = None,
                                    **plot_kwargs):

                    auc_scores = self.get_auc_scores(ood_groups_selections=ood_groups_selections,
                                                     return_averaged=return_averaged)
                    if not return_averaged:
                        annots = get_mean_stderr_annots_in_nested_dict(auc_scores)
                        annots = pd.DataFrame(annots).values.T
                        auc_scores = average_values_in_nested_dict(auc_scores)
                        plot_fmt = "s"
                    else:
                        annots = True
                        plot_fmt = ".2g"

                    plot_df = pd.DataFrame(auc_scores)

                    plot_heatmap(plot_df,
                                 title="AUC",
                                 save_dir=save_dir,
                                 annot=annots,
                                 fmt=plot_fmt,
                                 annot_kws={"fontsize": 9},
                                 figsize=(12, 0.75 * len(plot_df.columns)),
                                 **plot_kwargs,
                                 )

                def plot_score_distr(self,
                                     ood_groups_selections: Optional[list] = None,
                                     save_dir=None
                                     ):
                    out_domain_scores_mean = average_values_in_nested_dict(self.out_domain_scores, axis=0, dict_level=2)
                    in_domain_scores_mean = average_values_in_nested_dict(self.in_domain_scores, axis=0, dict_level=1)

                    selected_ood_group = self._filter_ood_groups(ood_groups_selections)

                    for ood_name in selected_ood_group:
                        ood_scores = out_domain_scores_mean[ood_name]
                        save_group_name = ood_name.lower().replace(" ", "_")

                        if save_dir is not None:
                            save_dir_ = f"{save_dir}_{save_group_name}.png"
                        else:
                            save_dir_ = None

                        plot_scores_distr(scores_test=in_domain_scores_mean,
                                          scores_new=ood_scores,
                                          title=ood_name,
                                          clip_q=0.05,
                                          kind="hist",
                                          bins=30,
                                          save_dir=save_dir_,
                                          labels=SCORING_FUNCTIONS)

                def plot_box_plot(self,
                                  ood_groups_selections: Optional[list] = None,
                                  save_dir=None
                                  ):

                    out_domain_scores_mean = average_values_in_nested_dict(self.out_domain_scores, axis=0, dict_level=2)
                    in_domain_scores_mean = average_values_in_nested_dict(self.in_domain_scores, axis=0, dict_level=1)

                    selected_ood_group = self._filter_ood_groups(ood_groups_selections)

                    for ood_name in selected_ood_group:
                        ood_scores = out_domain_scores_mean[ood_name]

                        if save_dir is not None:
                            save_dir_ = f"{save_dir}_{ood_name}.png"
                        else:
                            save_dir_ = None

                        plot_scores_boxplot(scores_test=in_domain_scores_mean,
                                            scores_new=ood_scores,
                                            show_outliers=True,
                                            title=ood_name,
                                            return_results=False,
                                            save_dir=save_dir_)

                def _filter_ood_groups(self, ood_groups_selections: Optional[list]):
                    if ood_groups_selections is not None:
                        assert all([ood_group_name in self.out_domain_scores for ood_group_name in ood_groups_selections]), \
                            "Please these OOD groups first before calculating AUC-ROC scores"
                        selected_ood_group = self.out_domain_scores.keys() & ood_groups_selections
                    else:
                        selected_ood_group = self.out_domain_scores.keys()

                    return selected_ood_group

         .. rubric:: Ancestors
            :name: ancestors

         -  `BasePipeline <base.html#selecting_OOD_detector.pipeline.base.BasePipeline>`__
         -  abc.ABC

         .. rubric:: Methods
            :name: methods

         ``def evaluate_ood_groups(self, ood_groups: dict, return_averaged: bool = False)``
            .. container:: desc

               Gives novelty scores to OOD groups. Returns and stores
               dictionary of novelty scores given by each model for each
               sample in every OOD group. If the function is called
               repeadetly, updates internally stored novelty scores for
               the OOD groups.

               .. rubric:: Parameters
                  :name: parameters

               **``ood_groups``** : ``dict``
                  Dictionary of OOD groups. Dictionary has to contain a
                  name of each OOD group and features in a pd.DataFrame.
                  Example: {"Flu patients": X_flu, "Ventilated
                  patients": X_vent}
               **``return_averaged``** : ``bool``
                  If true, returns averaged novelty score for each
                  sample. The shape of novelty scores given by each
                  model then corresponds to (1, n_samples). Else, the
                  shape of novelty scores given by each model is
                  (n_trials, n_samples) where n_trials is the number of
                  trials used in the fit function.

               .. rubric:: Returns
                  :name: returns

               **``out_domain_scores``** : ``dict``
                  Returns a dictionary of novelty scores given by each
                  model for each sample in every OOD group.

            Expand source code

            ::

               def evaluate_ood_groups(self,
                                       ood_groups: dict,
                                       return_averaged: bool = False):
                   """
                   Gives novelty scores to OOD groups.
                   Returns and stores dictionary of novelty scores given by each model for each sample in every OOD group.
                   If the function is called repeadetly, updates internally stored novelty scores for the OOD groups.

                   Parameters
                   ----------
                   ood_groups: dict
                       Dictionary of OOD groups. Dictionary has to contain a name of each OOD group and features in a pd.DataFrame.
                       Example: {"Flu patients": X_flu, "Ventilated patients": X_vent}

                   return_averaged: bool
                       If true, returns averaged novelty score for each sample. The shape of novelty scores given by each model
                       then corresponds to (1, n_samples).
                       Else, the shape of novelty scores given by each model is (n_trials, n_samples) where n_trials is
                       the number of trials used in the fit function.

                   Returns
                   -------
                   out_domain_scores: dict
                       Returns a dictionary of novelty scores given by each model for each sample in every OOD group.

                   """
                   assert self.novelty_estimators, "Novelty estimator dictionary is empty." \
                                                   "Please fit novelty estimators to in-distribution data before calling" \
                                                   "evaluate_ood_groups."

                   assert all([list(X_ood.columns) == self.feature_names for _, X_ood in ood_groups.items()]), \
                       "All OOD groups must have identical features to in-distribution data!"

                   out_domain_scores = self._score_out_domain(ood_groups)
                   self.out_domain_scores.update(out_domain_scores)

                   if return_averaged:
                       return average_values_in_nested_dict(out_domain_scores)

                   return out_domain_scores

         ``def fit(self, X_train, X_test, **kwargs)``
            .. container:: desc

               Fits models on training data with n_trials different
               runs. Novelty estimators from each run are stored in a
               nested dictionary in self.novelty_estimators. (E.g.: {0:
               {"AE": NoveltyEstimator, "PPCA": NoveltyEstimator}, 1:
               {"AE": NoveltyEstimator, "PPCA": NoveltyEstimator}} )
               Parameters

               --------------

               **``X_train``** : ``pd.DataFrame``
                  Training in-distribution data. Used to fit novelty
                  estimators.
               **``X_test``** : ``pd.DataFrame``
                  Test in-distribution data. Used to calculate
                  self.in_domain_scores which are taken as base novelty
                  scores for the dataset and used for comparison against
                  OOD groups later.

               kwargs: y_train: pd.DataFrame Labels corresponding to
               training data. n_trials: int Number of trials to run.

            Expand source code

            ::

               def fit(self,
                       X_train,
                       X_test,
                       **kwargs):
                   """
                   Fits models on training data with n_trials different runs.  Novelty estimators from each run are stored
                   in a nested dictionary in self.novelty_estimators.
                   (E.g.: {0: {"AE": NoveltyEstimator, "PPCA": NoveltyEstimator},
                          1: {"AE": NoveltyEstimator, "PPCA": NoveltyEstimator}} )
                   Parameters
                   ----------
                   X_train: pd.DataFrame
                       Training in-distribution data. Used to fit novelty estimators.
                   X_test: pd.DataFrame
                       Test in-distribution data. Used to calculate self.in_domain_scores which are taken as base novelty scores
                       for the dataset and used for comparison against OOD groups later.
                   kwargs:
                       y_train: pd.DataFrame
                           Labels corresponding to training data.
                       n_trials: int
                           Number of trials to run.
                   """
                   y_train = kwargs.get("y_train", None)
                   n_trials = kwargs.get("n_trials", 5)

                   assert list(X_train.columns) == list(X_test.columns), "Train and test data have different features!"
                   self.feature_names = list(X_train.columns)

                   print("--- OOD Pipeline ---")
                   print("1/2 Fitting novelty estimators...")
                   self._fit(X_train=X_train, y_train=y_train, n_trials=n_trials)

                   print("2/2 Scoring in-domain data...")
                   self.in_domain_scores = self._score_in_domain(X_test)

         ``def get_auc_scores(self, ood_groups_selections: Optional[list] = None, return_averaged: bool = True)``
            .. container:: desc

               Computes AUC-ROC scores of OOD detection for each OOD
               group as compared to the in-distribution test data. By
               default, returns scores for every group evaluated by the
               pipeline (evaluate_ood_groups).

               .. rubric:: Parameters
                  :name: parameters

               **``ood_groups_selections``** : ``Optional(list)``
                  Optionally provide a selection of OOD groups for which
                  AUC-ROC score should be returned. If no selection is
                  provided, all groups ever evaluate by the pipeline
                  will be included.
               **``return_averaged``** : ``bool``
                  Indicates whether to return averaged AUC-ROC scores
                  over n_trials run or a list of scores for every trial.

               .. rubric:: Returns
                  :name: returns

               **``aucs_dict_groups``** : ``dict``
                  A nested dictionary that contains a name of OOD group,
                  name of novelty estimator and either a float (if
                  averaged) or a list of AUC-ROC scores.

            Expand source code

            ::

               def get_auc_scores(self,
                                  ood_groups_selections: Optional[list] = None,
                                  return_averaged: bool = True):
                   """
                   Computes AUC-ROC scores of OOD detection for each OOD group as compared to the in-distribution test data.
                   By default, returns scores for every group evaluated by the pipeline (evaluate_ood_groups).

                   Parameters
                   ----------
                   ood_groups_selections: Optional(list)
                       Optionally provide a selection of OOD groups for which AUC-ROC score should be returned. If no selection
                       is provided, all groups ever evaluate by the pipeline will be included.
                   return_averaged: bool
                       Indicates whether to return averaged AUC-ROC scores over n_trials run or a list of scores for every trial.

                   Returns
                   -------
                   aucs_dict_groups: dict
                       A nested dictionary that contains a name of OOD group, name of novelty estimator and either a float (if
                       averaged) or a list of AUC-ROC scores.

                   """
                   selected_ood_group = self._filter_ood_groups(ood_groups_selections)
                   aucs_dict_groups = defaultdict(lambda: defaultdict(list))

                   for ood_group_name in selected_ood_group:
                       aucs_dict_groups[ood_group_name] = get_ood_aucs_score_for_all_models(
                           ood_scores_trials_dict=self.out_domain_scores[ood_group_name],
                           test_scores_trials_dict=self.in_domain_scores,
                       )

                   if return_averaged:
                       return average_values_in_nested_dict(aucs_dict_groups)

                   return aucs_dict_groups

         ``def plot_auc_scores(self, ood_groups_selections: Optional[list] = None, return_averaged: bool = True, save_dir: str = None, **plot_kwargs)``
            .. container:: desc

            Expand source code

            ::

               def plot_auc_scores(self,
                                   ood_groups_selections: Optional[list] = None,
                                   return_averaged: bool = True,
                                   save_dir: str = None,
                                   **plot_kwargs):

                   auc_scores = self.get_auc_scores(ood_groups_selections=ood_groups_selections,
                                                    return_averaged=return_averaged)
                   if not return_averaged:
                       annots = get_mean_stderr_annots_in_nested_dict(auc_scores)
                       annots = pd.DataFrame(annots).values.T
                       auc_scores = average_values_in_nested_dict(auc_scores)
                       plot_fmt = "s"
                   else:
                       annots = True
                       plot_fmt = ".2g"

                   plot_df = pd.DataFrame(auc_scores)

                   plot_heatmap(plot_df,
                                title="AUC",
                                save_dir=save_dir,
                                annot=annots,
                                fmt=plot_fmt,
                                annot_kws={"fontsize": 9},
                                figsize=(12, 0.75 * len(plot_df.columns)),
                                **plot_kwargs,
                                )

         ``def plot_box_plot(self, ood_groups_selections: Optional[list] = None, save_dir=None)``
            .. container:: desc

            Expand source code

            ::

               def plot_box_plot(self,
                                 ood_groups_selections: Optional[list] = None,
                                 save_dir=None
                                 ):

                   out_domain_scores_mean = average_values_in_nested_dict(self.out_domain_scores, axis=0, dict_level=2)
                   in_domain_scores_mean = average_values_in_nested_dict(self.in_domain_scores, axis=0, dict_level=1)

                   selected_ood_group = self._filter_ood_groups(ood_groups_selections)

                   for ood_name in selected_ood_group:
                       ood_scores = out_domain_scores_mean[ood_name]

                       if save_dir is not None:
                           save_dir_ = f"{save_dir}_{ood_name}.png"
                       else:
                           save_dir_ = None

                       plot_scores_boxplot(scores_test=in_domain_scores_mean,
                                           scores_new=ood_scores,
                                           show_outliers=True,
                                           title=ood_name,
                                           return_results=False,
                                           save_dir=save_dir_)

         ``def plot_score_distr(self, ood_groups_selections: Optional[list] = None, save_dir=None)``
            .. container:: desc

            Expand source code

            ::

               def plot_score_distr(self,
                                    ood_groups_selections: Optional[list] = None,
                                    save_dir=None
                                    ):
                   out_domain_scores_mean = average_values_in_nested_dict(self.out_domain_scores, axis=0, dict_level=2)
                   in_domain_scores_mean = average_values_in_nested_dict(self.in_domain_scores, axis=0, dict_level=1)

                   selected_ood_group = self._filter_ood_groups(ood_groups_selections)

                   for ood_name in selected_ood_group:
                       ood_scores = out_domain_scores_mean[ood_name]
                       save_group_name = ood_name.lower().replace(" ", "_")

                       if save_dir is not None:
                           save_dir_ = f"{save_dir}_{save_group_name}.png"
                       else:
                           save_dir_ = None

                       plot_scores_distr(scores_test=in_domain_scores_mean,
                                         scores_new=ood_scores,
                                         title=ood_name,
                                         clip_q=0.05,
                                         kind="hist",
                                         bins=30,
                                         save_dir=save_dir_,
                                         labels=SCORING_FUNCTIONS)
