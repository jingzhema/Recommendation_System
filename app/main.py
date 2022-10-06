import pandas as pd
from src.recommendater import Recommendater
from flask import Flask, render_template, request
app = Flask(__name__)
rec = Recommendater()

@app.route("/", methods=["GET", "POST"])
def index():
    message = ""
    user_id = '0'
    action = ""
    movie_title = ""
    if request.method == "POST":
        user_id = request.form["user_id"]
        movie_title = request.form["movie_title"].title().strip()
        action = request.form["action"]
        if (user_id.isnumeric() and int(user_id) > 0):
            message = "Welcome " + str(user_id)

    pd.set_option('colheader_justify', 'center')

    if (action == 'Content based'):
        if(movie_title != ""):
            df = genres_based(movie_title)
            if(isinstance(df, pd.DataFrame) and df.empty != True):
                return render_template("index.html",
                                       welcome_message=message,
                                       recommendation_title="Recommendations based on genres of " + movie_title,
                                       recommendation_tables=[df.to_html(classes='table_style', header="true")],
                                       titles=df.columns.values)
            else:
                return render_template("index.html",
                                       welcome_message="The movie name you entered is not in our system!")
        else:
            return render_template("index.html",
                                   welcome_message="Please enter a movie title to get recommendations of the same genres.")
    elif (action == 'Surprise SVD' and user_id.isnumeric() and int(user_id) > 0):
        df = memory_based(user_id)
        if isinstance(df, pd.DataFrame):
            return render_template("index.html",
                                   welcome_message=message,
                                   recommendation_title="Recommendations based on user " + user_id + "'s historical rating data:",
                                   recommendation_tables=[df.to_html(classes='table_style', header="true")],
                                   user_ratings_tables=[get_user_ratings(user_id).to_html(classes='table_style', header="true")],
                                   user_rating_title="User " + user_id + "'s historical rating data",
                                   titles=df.columns.values)
    elif (action == 'Content based2'):
        if(movie_title != ""):
            df = content_based(movie_title)
            if(isinstance(df, pd.DataFrame) and df.empty != True):
                return render_template("index.html",
                                       welcome_message=message,
                                       recommendation_tables=[df.to_html(classes='table_style', header="true")],
                                       titles=df.columns.values)
            else:
                return render_template("index.html",
                                       welcome_message="The movie name you entered is not in our system!")
        else:
            return render_template("index.html",
                                   welcome_message="Please enter a movie title to get recommendations of the same genres.")
    elif (action == 'Memory based2' and user_id.isnumeric() and int(user_id) > 0):
        df = memory_based2(user_id)
        if isinstance(df, pd.DataFrame):
            return render_template("index.html",
                                   welcome_message=message,
                                   recommendation_tables=[df.to_html(classes='table_style', header="true")],
                                   user_ratings_tables=[get_user_ratings(user_id).to_html(classes='table_style', header="true")],
                                   titles=df.columns.values)
    else:
        return render_template("index.html",
                               welcome_message="Please select a User ID or enter a movie title for recommendations.")

def genres_based(movie_title):
    return rec.genres_based(movie_title)

def memory_based(user_id):
    return rec.memory_based(user_id)

def memory_based2(user_id):
    return rec.memory_based2(user_id)

def get_user_ratings(user_id):
    data = rec.get_user_ratings(user_id)
    return data

def content_based(movie_title):
    return rec.content_based(movie_title)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)