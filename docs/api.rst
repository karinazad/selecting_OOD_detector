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
      :name: section-intr

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




