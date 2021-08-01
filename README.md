# DisasterResponsePipeline


## Python packages used:

This project can be executed using the Anaconda distribution of Python 3.x plus: 
    sklearn==0.23.1 
    nltk==3.5 
    Flask==1.1.2 
    plotly==5.1.0 
## Files contained in repository:
Readme.md <br>


## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Acknowledgment
I want to thank Udacity for the opportunity to work on such interesting project.
