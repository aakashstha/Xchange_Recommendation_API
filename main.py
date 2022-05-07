import uvicorn
from fastapi import FastAPI
from pymongo import MongoClient
from bson.objectid import ObjectId
import pandas as pd
from fastapi.responses import JSONResponse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem.porter import PorterStemmer
import warnings
warnings.filterwarnings("ignore")


# 2. Create the app object
app = FastAPI()


# Home route
@app.get('/')
def index():
    mydb = MongoClient(
        'mongodb+srv://xchange:xchange@xchange.nrbdi.mongodb.net/myFirstDatabase?retryWrites=true&w=majority')
    collection = mydb['myFirstDatabase']['products']
    a = collection.count_documents({})

    return {'message': 'Hello, World', 'count': a}


# Main route for predicting
@app.get('/predict/{productId}')
def get_name(productId: str):
    client = MongoClient(
        'mongodb+srv://xchange:xchange@xchange.nrbdi.mongodb.net/myFirstDatabase?retryWrites=true&w=majority')

    database = client.myFirstDatabase  # Connecting to the database
    collection = database.products  # Accessing the collection

    cursor = collection.find()

    # Converting cursor to the list of dictionaries
    list_cur = list(cursor)

    # Converting to the DataFrame
    products = pd.DataFrame(list_cur)
    # print(products)

    # To find the adTitle from the given _id
    productAdTitle = collection.find_one({"_id": ObjectId(productId)})
    print(productAdTitle['adTitle'])

    products = products[['_id', 'adTitle',
                         'description', 'category', 'location']]
    products['tags'] = products['adTitle'] + " " + products['description'] + \
        " " + products['category'] + " " + products['location']
    new_df = products[['_id', 'adTitle', 'tags']]
    new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())

    ps = PorterStemmer()

    def stem(text):
        y = []
        for i in text.split():
            y.append(ps.stem(i))
        return " ". join(y)

    new_df['tags'] = new_df['tags'].apply(stem)

    cv = CountVectorizer(max_features=1000, stop_words='english')
    vector = cv.fit_transform(new_df['tags']).toarray()

    similarity = cosine_similarity(vector)

    product_index = new_df[new_df['adTitle']
                           == productAdTitle['adTitle']].index[0]
    distances = similarity[product_index]
    product_list = sorted(list(enumerate(distances)),
                          reverse=True, key=lambda x: x[1])[1:5]
    localTitle = []
    localid = []

    # print(new_df.iloc[1]._id)
    for i in product_list:
        # print(i)
        localTitle.append(new_df.iloc[i[0]].adTitle)
        a = new_df.iloc[i[0]]._id
        localid.append(str(a))
        # print(new_df.iloc[i[0]].adTitle)

    return JSONResponse({"Product Title": localTitle, "id": localid})


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8001)

# uvicorn main:app --reload
# uvicorn main:app --port 8001

# first parameter is the file name and second parameter is the object name
