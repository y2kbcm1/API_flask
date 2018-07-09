# A gentle guide to API building in Machine Learning with Python and Flask module


![Flask logo](/img/flask.png)

## Why do we need API's in Machine Learning ?

As we design complex predictive models based on Machine Learning or Deep Learning, we need
to make it available to customers. This is the _deployment_ or _production_ step, we face
several options to to so :
- create a web app based on any framework compatible to the language used for Machine Learning
inference phase. This might be an easy way with cool frameworks such [RShiny](http://shiny.rstudio.com/) or [Dash](https://plot.ly/products/dash/),
however this naturally enforces the need to have the prediction model embedded in the app and in the same programming language as the app framework.
In particular, if we think of Python and R, their interpreted behavior makes difficult to hide the predictive model within the app.

- API approach, Application Programming Interfaces delivers the output as a webservice, allowing to connect to the prediction model as-a-service.
This approach make cross-language applications. Typically if a backend developper needs to call the result of the predictive model within his/her
PHP code or a frontend developper needs to egt the same result within his/her js framework, they would only need to get to the URL endpoint where the API is served. 
Meanwhile, the API can be served in the same programming language as the predictive model, typically in Python with Flask.


## Formal definition of API's

An API is an interface between two different softwares in a specified format. Typically, the user software gives an input with the specified format to the API, and the 
API provides back the output to the user software. All cloud providers and Machine Learning providers deliver API's. The important aspect of API's in the Machine
Learning framework is that it does give the ability to non-ML engineers to incorporate in their product ML capabilities.

Speaking specifically of Machine Learning, an API might provide :
- _the inference phase_ : once a Machine Learning predictive model has been trained, the API simply contains the serialized predictive model and the ability to give
it new input for prediction. This is the simplest use of API's, that will cover in this tutorial.
- _the training phase_ : most Machine Learning models needs to be retrained or tuned as new data is available. An API can also be used to receive at meantime new data (including labels), update the model (typically with online learning algorithms) and provide output for new data. This latter use is more complex and won't be treated here.


A typical professional API is the one provided by Google [API Vision](https://cloud.google.com/vision/), this API embeds in reality many different minimalist services.


## Flask

Flask is Python framework that allows to embeds microservices. It is defined as a web framework in Python. Unlike Python, Flask is not really opiniated and does not
force you to things the way Flask developers want it. There are obviously tons of other frameworks : Django, Falcon to name a few...
(N.B For R users, we will need a separate tutorial for [plumber](https://www.rplumber.io/)).

### Flask set up

Start with creating a `conda` environment with `flask` and `gunicorn`

```console
conda create --name FlaskAPI python=3.6
```

and activate it 

```console
conda activate FlaskAPI
```

and now install `flask` and `gunicorn`

```console
pip install flask gunicorn
```
This will install also `Werkzeug`, `Jinja2`, `click`, `MarkupSafe`and `itsdangerous`. But one mai advantage of `Flask`is that it is a very light environment.


### First Hello application 

Now in `src`, you will find a `hello-ekimetrics.py` Python early implementation of a simple `Flask` application. That we detail here :

```python
""" File : hello_ekimetrics.py 
"""

from flask import Flask 

app = Flask(__name__)

@app.route('/users/<string:username>')
def hello_ekimetrics(username=None):
    return("Welcome {} to Ekimetrics!".format(username))

if __name__ == "__main__":
    app.run()
```

Now execute the Python file

```console
python hello_ekimetrics.py
```

and you should see something like :

```console
(FlaskAPI) C:\git\API_FLASK\src>python hello_ekimetrics.py
 * Serving Flask app "hello_ekimetrics" (lazy loading)
 * Environment: production
   WARNING: Do not use the development server in a production environment.
   Use a production WSGI server instead.
 * Debug mode: off
 * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)
 ```

 Now go to http://127.0.0.1:5000/users/someone, and you should see 


![Flask logo](/img/welcome.PNG)


## First ML API with Flask

Still in `FlaskAPI` environment, install some Python module to train and evaluate a predictive model.

We will build a predictive model for the famous `Pima Indians Diabetes` dataset (see below) 

| `pregnancies` | `Glucose`     | `BloodPressure`    | `SkinThickness`            | `Insulin`       | `BMI`          | `DiabetesPedigreeFunction` | `Age` | `Outcome` |
|--------------:|--------------:|-------------------:|---------------------------:|----------------:|---------------:|---------------------------:|------:|----------:|
| 6             | 148           | 72                 | 35                         | 0               | 33.6           | 0.627                      | 50    | 1         |


In `data` folder, you will find `training.csv` and `test.csv`. We will use `training.csv` to find the right hyperparameters of a Gaussian kernel SVM.