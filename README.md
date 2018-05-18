# Prediction of Finger Movements from EEG Recordings

The aim of this project is to correctly associate EEG data recordings coming from
28 electrodes (channels) attached to a patient to binary data (indicating if
the person moves its finger to the right or to the left).

Tests
-------
The main script is can be run with `python testsuite.py`, which trains the best
model we found with an "optimal" choice of hyperparameters (kernel sizes,
convolutional layer dimensions...) and then computes the number of errors wrt the test.

The hyperparameters optimization was performed by running
`python hyperoptimization.py`.

We also provide a script to run a forward pass of the best "already trained" model. This gives a test error of 18% and a train error of around 2%. To launch this model,
run `python run_forward_of_best_model.py`.
