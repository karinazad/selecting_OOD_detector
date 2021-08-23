API Reference
####################################

         
.. container::

   -  .. rubric:: `Class <#header-class>`__
         :name: classes

      -  .. rubric:: ``OODPipeline``
            :name: oodpipeline

         -  ``fit``
         -  ``evaluate_ood_groups``
         -  ``get_auc_scores``
         -  ``plot_auc_scores``
         -  ``plot_box_plot``
         -  ``plot_score_distr``
         

OODPipeline
*****************

   .. container:: section
      :name: section-intro

   .. container:: section
   
         .. rubric:: Module
            :name: header-classes
            :class: section-title

            ``selecting_OOD_detector.pipeline.ood_pipeline.OODPipeline``

   .. container:: section

   .. container:: section

   .. container:: section
   
   .. rubric:: Class
            :name: header-class
            :class: section-title

      ``class OODPipeline (**kwargs)``
         .. container:: desc

            Pipeline to fit novelty estimators on in-distribution data
            and evaluate novelty of Out-of-Distribution (OOD) groups.

            Example of usage:

            ::

               # Initialize the pipeline
               oodpipeline = OODPipeline()

               # Fit the pipeline on in-distribution training data and compute novelty scores for in-distribution test data
               oodpipeline.fit(X_train= X_train, X_test=X_test)

               # Define OOD groups and evaluate by the pipeline
               ood_groups = {"Flu patients": X_flu, "Ventilated patients": X_vent}
               oodpipeline.evaluate(ood_groups)

               # Inspect AUC-ROC scores of detecting OOD groups
               oodpipeline.get_auc_scores()

            .. rubric:: Parameters
               :name: parameters

            ``kwargs``
               model_selection: set Define which models to train, e.g.
               ``{"PPCA", "LOF", "VAE"}``. If selection is not provided, all
               available models are used.


         .. rubric:: Ancestors
            :name: ancestors

         -  `BasePipeline <base.html#selecting_OOD_detector.pipeline.base.BasePipeline>`__

         .. rubric:: Methods
            :name: methods

       
         ``def fit(self, X_train, X_test, **kwargs)``
            .. container:: desc

               Fits models on training data with n_trials different
               runs. Novelty estimators from each run are stored in a
               nested dictionary in self.novelty_estimators. (E.g.: ``{0:
               {"AE": NoveltyEstimator, "PPCA": NoveltyEstimator}, 1:
               {"AE": NoveltyEstimator, "PPCA": NoveltyEstimator}} )``
               Parameters

               --------------

               ``X_train`` : ``pd.DataFrame``
                  Training in-distribution data. Used to fit novelty
                  estimators.
               ``X_test`` : ``pd.DataFrame``
                  Test in-distribution data. Used to calculate
                  self.in_domain_scores which are taken as base novelty
                  scores for the dataset and used for comparison against
                  OOD groups later.

               kwargs: 
               
               ``y_train``: ``pd.DataFrame``
                  Labels corresponding to training data.
               ``n_trials``: ``int`` 
                  Number of trials to run.
                  
         ``def evaluate_ood_groups(self, ood_groups, return_averaged=False)``
            .. container:: desc

               Gives novelty scores to OOD groups. Returns and stores
               dictionary of novelty scores given by each model for each
               sample in every OOD group. If the function is called
               repeadetly, updates internally stored novelty scores for
               the OOD groups.

               .. rubric:: Parameters
                  :name: parameters

               ``ood_groups`` : ``dict``
                  Dictionary of OOD groups. Dictionary has to contain a
                  name of each OOD group and features in a pd.DataFrame.
                  Example: ``{"Flu patients": X_flu, "Ventilated
                  patients": X_vent}``
               ``return_averaged`` : ``bool``
                  If true, returns averaged novelty score for each
                  sample. The shape of novelty scores given by each
                  model then corresponds to (1, n_samples). Else, the
                  shape of novelty scores given by each model is
                  (n_trials, n_samples) where n_trials is the number of
                  trials used in the fit function.

               .. rubric:: Returns
                  :name: returns

             ``out_domain_scores`` : ``dict``
                  Returns a dictionary of novelty scores given by each
                  model for each sample in every OOD group.
                  

         ``def get_auc_scores(self, ood_groups_selections=None, return_averaged=True)``
            .. container:: desc

               Computes AUC-ROC scores of OOD detection for each OOD
               group as compared to the in-distribution test data. By
               default, returns scores for every group evaluated by the
               pipeline (evaluate_ood_groups).

               .. rubric:: Parameters
                  :name: parameters

               ``ood_groups_selections``: ``Optional(list)``
                  Optionally provide a selection of OOD groups for which
                  AUC-ROC score should be returned. If no selection is
                  provided, all groups ever evaluate by the pipeline
                  will be included.
               ``return_averaged``: ``bool``
                  Indicates whether to return averaged AUC-ROC scores
                  over n_trials run or a list of scores for every trial.

               .. rubric:: Returns
                  :name: returns

               ``aucs_dict_groups``: ``dict``
                  A nested dictionary that contains a name of OOD group,
                  name of novelty estimator and either a float (if
                  averaged) or a list of AUC-ROC scores.

            

         ``def plot_auc_scores(self, ood_groups_selections=None, show_stderr=True, save_dir=None, **plot_kwargs)``
            .. container:: desc

               Plots a heatmap of AUC-ROC scores of OOD detection for
               each OOD group as compared to the in-distribution test
               data.

               .. rubric:: Parameters
                  :name: parameters

               ``ood_groups_selections`` : ``Optional(list)``
                  Optionally provide a selection of OOD groups for which
                  AUC-ROC score should be returned. If no selection is
                  provided, all groups ever evaluate by the pipeline
                  will be included.
               ``show_stderr``: ``Optional(bool)``
                  If True (default), annotates the heatmpa with means
                  and standard error (calculated using jacknife
                  resampling). Else, plots the mean values only.
               ``save_dir``: ``Optional(str)``
                  If a path to a directory is provided, saves plots for
                  each OOD group separately.
               ``plot_kwargs``
                  Other arguments to be passed to sns.heatmap function.

      

         ``def plot_box_plot(self, ood_groups_selections=None, save_dir=None)``
            .. container:: desc

               .. rubric:: Parameters
                  :name: parameters

               ``ood_groups_selections`` : ``Optional(list)``
                  Optionally provide a selection of OOD groups for which
                  AUC-ROC score should be returned. If no selection is
                  provided, all groups ever evaluate by the pipeline
                  will be included.
               ``save_dir``: ``Optional(str)``
                  If a path to a directory is provided, saves plots for
                  each OOD group separately.

           

         ``def plot_score_distr(self, ood_groups_selections=None, save_dir=None)``
            .. container:: desc

               .. rubric:: Parameters
                  :name: parameters

               ``ood_groups_selections``: ``Optional(list)``
                  Optionally provide a selection of OOD groups for which
                  AUC-ROC score should be returned. If no selection is
                  provided, all groups ever evaluate by the pipeline
                  will be included.
               ``save_dir``: ``Optional(str)``
                  If a path to a directory is provided, saves plots for
                  each OOD group separately.
                  


Generated by `pdoc 0.10.0 <https://pdoc3.github.io/pdoc>`__.

