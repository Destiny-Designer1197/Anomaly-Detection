# Secure CAN Bus Communications: Anomaly Detection through Ensembles
This git repository contains the code files, logs, documents and trained models for the Secure CAN Bus Communications using Ensemble approach. This repository and all its contents are the private work of Banupriya Valluvan, as part of Master Thesis in the department of IAS, University of Stuttgart to complete the masters in INFOTECH course.

## Introduction
- The shift towards advanced infotainment systems and self-driving technology in the   automotive industry has transformed vehicles into mobile networks of connected computing devices, including potential Internet connections. This opens up opportunities for innovation but also makes cars vulnerable to cyber-attacks that can exploit software weaknesses and pose safety risks for drivers and those nearby.

- The CAN bus in vehicles is an efficient communication system between electronic control units, but lacks security features. To effectively detect potential threats, an intrusion detection system is necessary.However, there are few known attack signatures for vehicle networks, and high accuracy is needed to avoid false-positive errors that could affect driver safety. The proposed solution is the Ensemble Learning Approach and Deep Hybrid Learning Approach(DHL). Both previously stated approaches can detect unknown attacks using raw data and has shown high accuracy in detecting attacks in our experiments.

## Datasets used:
- OTIDS dataset
- Car hacking dataset
- ROAD dataset

## Different Ensemble Approaches:
- Independent Ensemble
- Data Centered Ensemble
- Bagging Ensemble
- Boosting Ensemble
- Stacking Ensemble
- Deep Hybrid Learning Ensemble

## Concept for different Ensemble Approaches:
- The generalized dataset consists of ROAD , OTIDS and CarHacking datasets.
- The dataset is downsampled to attain a proper class balance.
- The dataset is divided into five different subsets to evaluate the performance of base models.
- Train the base models on the divided subsets
- Shortlist the base models that are performing good on all five subsets.
- Use the shortlisted base models on the different Ensemble Approaches.

## Concept for Deep Hybrid Learning Ensemble:
- The generalized dataset consists of ROAD , OTIDS and CarHacking datasets.
- The dataset is downsampled to attain a proper class balance.
- Train the 1d CNN with the downsampled dataset.
- Extract the trained CNN features and use that as input to the ML Classifier.
- Train the ML Classifier with the considered base models.

## Requirements for Running the Code in this Repository
- Make sure to download an Integrated Development Environment (IDE) that supports both Python files (.py) and Python notebooks (.ipynb) such as Microsoft Visual Studio Code. 

- Additionally, ensure that all the required libraries specified in the requirement.txt file have been installed in the system to prevent any issues when running the code. It's worth noting that upon installing the specified libraries, any dependent libraries will be automatically installed as well.





