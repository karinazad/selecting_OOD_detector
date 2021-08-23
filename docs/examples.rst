##########
Examples
##########

Detecting Clinically Relevant OOD Groups
*****************************************

This example shows how to test OOD detectors on two groups using dummy
variables of 

* patients under 18 years 

* COVID-19 patients

First, define the in-distribution and OOD data:

.. code:: py

    import pandas as pd
    import numpy as np

    n_features = 15
    # Define training and testing in-distribution data
    X_train = pd.DataFrame(np.random.rand(80, n_features))
    X_test = pd.DataFrame(np.random.rand(20, n_features))

    # Define OOD groups
    X_under18 = pd.DataFrame(np.random.rand(12, n_features))
    X_covid = pd.DataFrame(np.random.rand(7, n_features))

    ood_groups = {"Patients under 18 years": X_under18,
                  "COVID-19 patients": X_covid}
                  

Next, initialize and fit OOD Pipeline to in-distribution data and score
OOD groups:

.. code:: py

    from selecting_OOD_detector.pipeline.ood_pipeline import OODPipeline

    # Initialize the pipeline
    oodpipe = OODPipeline()

    # Fit OOD detection models on in-distribution training data and score in-distribution test data to calculate novelty baseline.
    oodpipe.fit(X_train, X_test=X_test)

    # Compute novelty scores of the defined OOD groups
    oodpipe.evaluate_ood_groups(ood_groups)

Finally, inspect AUC-ROC score of OOD detection:

.. code:: py

    auc_scores = oodpipe.get_ood_aucs_scores(return_averaged=True)

+---------------------+---------+---------+---------+---------+---------+---------+
|                     | AE      | DUE     | Flow    | LOF     | PPCA    | VAE     |
+=====================+=========+=========+=========+=========+=========+=========+
| Patients Under 18   | 0.513   | 0.552   | 0.493   | 0.489   | 0.514   | 0.654   |
+---------------------+---------+---------+---------+---------+---------+---------+
| COVID-19 patients   | 0.525   | 0.631   | 0.567   | 0.567   | 0.474   | 0.553   |
+---------------------+---------+---------+---------+---------+---------+---------+

AUC-ROC score of 1 would indicate perfect separation of an OOD group
from testing data while score of 0.5 suggests that models are unable to
detect which samples are in- and out-of-distribution.


To visualize distributions of novelty scores, plot histogram using `plot_score_distributions` or boxplots using `plot_box_plot` functions:

.. code:: py

    oodpipe.plot_box_plot()

.. image:: https://raw.githubusercontent.com/karinazad/selecting_OOD_detector/master/docs/img/download%20(1).png

.. image:: https://raw.githubusercontent.com/karinazad/selecting_OOD_detector/master/docs/img/download.png

