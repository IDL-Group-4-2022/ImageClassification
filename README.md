# ImageClassification

## Running instructions for training the models

1. Extract dl2021-image-corpus-proj into resource/data/original (this is already done if you download the google drive zip)
1. Extract image-test-corpus-139ArJI into resource/data/original (this is already done if you download the google drive zip)
1. Run `python src/data_wrangling/generate_dataframe.py` to get the train.csv
1. Run `python src/data_wrangling/training.py` to train the CNN2 model

### Location of adjustable parameters

* `src/utils/image_loader.py` -- adjust (a) transforms
* `src/testing/project.py` -- adjust (b) model and (c) optimizer

## Other scripts

* `src/error_analysis.py` is used to see the images and their actual and predicted labels using our models
* `src/testing.py` is used to predict the images in the testing folder and produce a test.csv file
* `src/plotting/label_plots.py` creates some data exploration plots which we used in the report
