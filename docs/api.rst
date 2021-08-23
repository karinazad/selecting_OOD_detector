API Reference
####################################


OODPipeline
*****************


``selecting_OOD_detector.pipeline.ood_pipeline API documentation``



.. contents::
   :depth: 3
..

.. container::

   .. container:: section
      :name: section-intro

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


         ``def plot_auc_scores(self, ood_groups_selections: Optional[list] = None, return_averaged: bool = True, save_dir: str = None, **plot_kwargs)``
            .. container:: desc


         ``def plot_box_plot(self, ood_groups_selections: Optional[list] = None, save_dir=None)``
            .. container:: desc

         ``def plot_score_distr(self, ood_groups_selections: Optional[list] = None, save_dir=None)``
            .. container:: desc

           

Generated by `pdoc 0.10.0 <https://pdoc3.github.io/pdoc>`__.


