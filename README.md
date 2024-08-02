# Water Consumption Prediction Service 
A complete end to end system for water consumption forecasting involving Data Preprocessing, Model Building using CNN-LSTM, and Model Deployment using Docker and Flask.


# Table of Contents
1. [ Data Format ](#data)
2. [ Model Training ](#Using)
3. [ Model Deployment ](#Future_scope) 
4. [ Additional Information ](#info)

## Repository Contents
The repository contains the following Files and Folders

1. Images: Various Images such as logo, docker deployment, results etc.
2. PreProcessing_Notebook.ipynb: Notebook containing preprocessing codes of the data.
3. Training_Notebook.ipynb: Notebook containing training codes for water consumption forecasting.
4. Dockerfile: Docker File for Deployment.
5. requirements.txt: Contains various packages and dependencies required for the project.
6. config.yaml: Configuration file for changing variables and paths of the project.
7. main.py: Contains script for training the model.
8. app.py: Backend for web application deployment using flask.
9. templates/index.html: Frontend for web application deployment using flask.



<a name="data"></a>
# Data Format

Due to privacy reason, the data cant be shared, but details and format of the data is shown below.

1. Data has water consumption details for 5 households: A, B, C, D, E.
2. Sample data format is shown below.

        id,starttime,stoptime,water_consumption
        B,2020-12-21 17:49:09.000,2020-12-21 17:50:05.000,2.8
        B,2020-12-22 20:08:54.000,2020-12-22 20:10:01.000,4.2
        A,2020-11-20 06:18:14.000,2020-11-20 06:18:14.000,0.5
        A,2020-11-04 23:31:43.000,2020-11-04 23:31:45.000,0.5
        A,2020-09-21 10:08:48.000,2020-09-21 10:08:53.000,2.7
        C,2020-09-02 23:06:45.000,2020-09-02 23:06:45.000,0.9
        C,2020-12-04 06:07:23.000,2020-12-04 06:07:23.000,1.1

3. In data places where there is no consumption of water, data is not provided.

4. Data is present in txt file format in our case.



<a name="using"></a>
# Model Training

1. Open the Notebook `PreProcessing_Notebook.ipynb` to follow all the preprocessing steps of the data.

2. Open the Notebook `Training_Notebook.ipynb` to follow all the training steps of the model.
     
    Alternatively to train the model.

    1.  Open command line cmd at the root of the repository.

    2.  Run the command   

        `pip install -r requirements.txt` 

    3. Run the command 

        `python main.py`

NOTE:  In order to make path, variables or any related change, please change the `config.yaml` file. 

<a name="Model Deployment"></a>
# Model Deployment

1. A `Dockerfile` is provided which can be used for deployment. From this `Dockerfile` a docker image can be created and deployed in cloud, etc.

    1. To create a docker image, first download docker for your OS from the official docker website.
    
    2. Then, open a command line cmd at the root of the repository, and run the command: `docker build -t water_consumption_image:v1 .`

    3. Once the image is created, you can push the docker image to the docker hub after signing in, from where the image can be used.

    4. To run the docker image, open a command line cmd at the root of the repository, and run the command: `docker run -p 5000:5000 water_consumption_image:v1`

    5. Open the link on your preffered browser: `http://127.0.0.1:5000/`, or check the logs provided by Docker in command line, to find the link.

2. Also a seperate `templates/index.html` and `app.py` is provided which can serve as frontend and backend for a web app based water demand prediction deployment system.

    To run the application, open a command line cmd at the root of the repository, and run the command: `flask run`

3. In the future all models can be stored on cloud for sending a `request` and getting a `response` for demand prediction.


<a name="Version"></a>

<a name="info"></a>
# Additional Information
## Python Version
The whole project is developed with python version `Python 3.7.7` and pip version `pip 19.2.3`.
## Contact
In case of error, feel free to contact me over Linkedin at [Adnan](https://www.linkedin.com/in/adnan-karol-aa1666179/).

