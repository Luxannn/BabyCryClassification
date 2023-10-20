# BabyCryClassification

This repository contains a model and data for classifying baby cries into different categories such as "belly_pain," "burping," "discomfort," "hungry," and "lonely" using .wav audio files.

## Contents

- `data/`: This directory contains the audio data used for training and testing the model.
  - `belly_pain/`: .wav files related to baby cries due to belly pain.
  - `burping/`: .wav files related to baby cries when they need to burp.
  - `discomfort/`: .wav files related to baby cries when they are uncomfortable.
  - `hungry/`: .wav files related to baby cries when they are hungry.
  - `lonely/`: .wav files related to baby cries when they are lonely.

- `BabyCry.py`: Python code for Streamlit interface.
- `BabyCryModel.pkl`: The trained machine learning model for baby cry classification.
- `baby cry classification.ipynb`: Jupyter Notebook containing code for training and evaluating the model.
- `requirements.txt`: List of Python packages and dependencies required to run the code.

## Usage

To use the model for classifying baby cries, you can go to out streamlit website:

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/Luxannn/BabyCryClassification.git
