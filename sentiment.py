from fastapi import FastAPI
from joblib import load
from pydantic import  BaseModel
import numpy as np

# class MySentimentModel
# load model
clf = load('model.joblib')
model_ = load('word2vec.model')


def get_prediction(param1):

    list_text = []
    # model = Word2Vec(sentences=param1,size=5, window=5, min_count=1, workers=2)

    for t in param1:
        list_text.append(model_.wv[t])

    # print(y)
    list_text=np.array([[0]*5]*(500-len(list_text))+list_text)

    # class MySentimentModel()
    y = clf.predict(np.array([list_text]))
    # print(np.array([list_text]).shape)
    # print(np.argmax(y)

    return {'prediction': int(np.argmax(y))}
    


# initiate API
app = FastAPI()


# define model for post request.
# class ModelParams(BaseModel):
#     param1: str


@app.get("/predict/{text}")
def predict(text):

    pred = get_prediction(text)

    return pred

@app.get("/")
def predict():

    pred = get_prediction('test')

    return pred
# if __name__ == "__main__":
#     app.run(debug=True, host='0.0.0.0')