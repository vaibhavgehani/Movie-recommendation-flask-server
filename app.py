from flask import Flask,request,jsonify,Response
from flask_cors import CORS;
import pandas as pd
import json
app=Flask(__name__)
CORS(app)
#importing the dataset
movie_data = pd.read_csv('movies.csv')

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(stop_words='english')
movie_data.overview=movie_data.overview.fillna('')
tfidf_matrix = tfidf.fit_transform(movie_data.overview);

from sklearn.metrics.pairwise import linear_kernel


indices = pd.Series(movie_data.index,index=movie_data['title']).drop_duplicates();
cosine_sim = linear_kernel(tfidf_matrix,tfidf_matrix)
print(cosine_sim.shape)
def recommend_movie(movieName,cosine_sim=cosine_sim):
    try:
        indx=indices[movieName]
        score_tuple=list(enumerate(cosine_sim[indx]))
        sorted_tuple=sorted(score_tuple,key=lambda x: x[1],reverse=True)
        top_10_score=sorted_tuple[1:6]
        top_10_index=[i[0] for i in top_10_score]
        return movie_data[['title','spoken_languages','popularity','release_date','runtime','poster_path']].iloc[top_10_index]
    except(Exception):
        print('Erorr')
@app.route('/movie')
def main():
    name=request.args.get('name')
    print(name)
    if(name != None):
        recom_array=recommend_movie(name)
    print(recom_array)
    try:
        return recom_array.to_json(orient='records')
    except:
        return []

if __name__ == '__main__':
    app.run(debug=True)