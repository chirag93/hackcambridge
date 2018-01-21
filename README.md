# KNOW-BRAINER
# Languages/Libs Used: Python, Keras, Tensorflow, Pandas, Numpy, Scipy, Scikit-learn, Javascript, React, D3.js
# Platforms: Azure Cognitive Services, AWS
# Location: HackCambridge Jan 20-21, 2018.

# Setup

    pip install -r requirements.txt

You also need to install The Virtual Brain if you want to run the simulation software.

## Inspiration
There are ~20M people in the world with epilepsy. A possible life-changing therapy involves surgery after intensive seizure monitoring in a hospital setting. This seizure monitoring however, involves hours and hours of seizure annotation by hand because clinicians do not have a good method for automatically determining when seizure happens.

However, we are inspired by innovation with dynamical simulations developed in academia, and the machine learning research progressing rapidly through every industry. We combine state-of-the-art whole-brain simulations, and machine learning algorithms to create a webapp for seizure annotation.

## What it does
It is a web application that performs 3 tasks:

### 1. Data Interaction
We provide a web interface to plot the raw data of the patient's EEG activity, for visualization and comparison with our algorithms.

### 2. Personalized Algorithm Training For Patient
We provide a backend that only needs certain metadata about the patient to generate simulated seizure activity for that specific patient. Then the simulated data can be used to train a personalized ensemble of machine learning algorithms (i.e. CNN, CNN+LSTM, Random Forest, Logistic Regression, SVM).

Note that this is COMPLETELY unsupervised in the sense that no real data would be used.

### 3. Seizure Annotation of Patient's EEG
The clinician can then upload a dataset that can be automatically annotated with our algorithms and then compared with the visual EEGs to ensure absolute correctness. This ensures robustness, but also efficiency in this seizure annotation approach, thus saving incredible amounts of time and money for the healthcare system.

## How we built it
**Nonlinear Whole-Brain Simulations**
Using The Virtual Brain (developed in Marseille, France), we modified various models to generate seizure activity across a whole suite of simulation parameters. This generated a diverse and realistic seizure dataset given a patient.

**Data Processing and Pipelining**
We took these seizure datasets and applied various digital signal processing and augmentation techniques to create a final dataset that was labeled with seizure, or no seizure time windows. 

**Semi-Supervised Deep Learning**
We fed this final dataset into various algorithms, such as a CNN, CNN+LSTM, Random Forest, Logistic Regression and SVM to output predictions of seizure vs. no seizure on EEG time windows.

We experimented with Microsoft Azure Cognitive Services - Image Recognition to determine the recognition capabilities of their service.

We also used AWS to host the entire project and it can run the pipeline from beginning to end, but might require some hacking :p!

## Challenges we ran into
Setting up a GPU instance during a 24 hour hackathon using Google Cloud, AWS, Azure, or IBM Cloud is not trivial. You require a 48 hour approval even with student credits. This was something not anticipated though, and we made the best use of AWS and Azure we could given their hosting and cognitive services available.

## Accomplishments that we're proud of
We managed to create an end-to-end pipeline product that has potentially real-world usage in a clinical setting. 

Even though our GPU instance didn't work, we bounced back and got the entire pipeline working from end-to-end.

## What we learned
- how to work with keras/tensorflow
- how to do data processing/pipelining
- how to setup cloud instances
- how to train machine learning algorithms
- how to work with nonlinear models for computational simulations
