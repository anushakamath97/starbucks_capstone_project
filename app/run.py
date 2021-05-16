import json
import plotly
import pandas as pd

from flask import Flask
from flask import render_template, request
from plotly.graph_objs import Bar
import joblib

app = Flask(__name__)

# load data
user_data = pd.read_csv("../data/processed/profile.csv")
offer_data = pd.read_csv("../data/processed/portfolio.csv")
completion_time_data = pd.read_csv("../data/processed/completion.csv")

del user_data["Unnamed: 0"]
del completion_time_data["Unnamed: 0"]

# classification classes
classification_class = ["Customer will likely not like the offer :(", "Customer will mostly like the offer. Send it!",
                        "No need to send offer, customer generally buys this product :)",
                        "It is useful to send this information to customer."]

# load model
model = joblib.load("../models/decision_tree_2.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    channels = ["web", "email", "mobile", "social"]

    portfolio_df = pd.read_csv("../data/portfolio.csv")
    offer_type = portfolio_df.offer_type.unique().tolist()
    channel_data = []
    for o_type in offer_type:
        usage_count = []
        for chn in channels:
            usage_count.append(portfolio_df[(portfolio_df["offer_type"] == o_type) & (portfolio_df[chn] == 1)].shape[0])
        channel_data.append(Bar(x=channels, y=usage_count, name=o_type))

    # became members across year plot data

    profile_df = pd.read_csv("../data/profile.csv")
    profile_df["became_member_on"] = pd.to_datetime(profile_df["became_member_on"], format="%Y-%m-%d")
    profile_df["year"] = profile_df["became_member_on"].apply(lambda date: date.year)
    year_counts = profile_df["year"].value_counts().reset_index()

    # number of offers received, viewed and completed data
    transcript_df = pd.read_csv("../data/transcript.csv")
    received_df = transcript_df[transcript_df.event_offer_received == 1].join(
        portfolio_df[["id", "offer_type"]].set_index("id"), on="value")
    received_df = received_df["offer_type"].value_counts().reset_index()

    viewed_df = transcript_df[transcript_df.event_offer_viewed == 1].join(
        portfolio_df[["id", "offer_type"]].set_index("id"), on="value")
    viewed_df = viewed_df["offer_type"].value_counts().reset_index()

    completed_df = transcript_df[transcript_df.event_offer_completed == 1].join(
        portfolio_df[["id", "offer_type"]].set_index("id"), on="value")
    completed_df = completed_df["offer_type"].value_counts().reset_index()

    # create visuals
    graphs = [
        {
            'data': channel_data,
            'layout': {
                'title': 'Distribution of Channel usage across Offer Types',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Channel"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=year_counts["index"].tolist(),
                    y=year_counts["year"].tolist()
                )
            ],
            'layout': {
                'title': 'Number of members joined across years',
                'yaxis': {
                    'title': "No. of members"
                },
                'xaxis': {
                    'title': "Year"
                }
            }

        },
        {
            'data': [
                Bar(
                    x=received_df['index'].tolist(),
                    y=received_df["offer_type"].tolist(),
                    name="received"
                ),
                Bar(
                    x=viewed_df['index'].tolist(),
                    y=viewed_df["offer_type"].tolist(),
                    name="viewed"
                ),
                Bar(
                    x=completed_df['index'].tolist(),
                    y=completed_df["offer_type"].tolist(),
                    name="completed"
                )
            ],
            'layout': {
                'title': 'Number of user, offer interaction data',
                'yaxis': {
                    'title': "No. of offers"
                },
                'xaxis': {
                    'title': "Offer Type"
                }
            }

        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph_{}".format(i) for i in range(len(graphs))]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in variables
    user_id = int(request.args.get("user_id", 0))
    offer_id = int(request.args.get("offer_id", 0))

    # create input for model
    compl_time = completion_time_data[(completion_time_data.user_id == user_id) & (completion_time_data.offer_id == offer_id)][
        "completion_time"].tolist()
    if len(compl_time) > 0:
        compl_time = compl_time[0]
    else:
        compl_time = 0

    query = [user_id, offer_id, compl_time] + offer_data.set_index("offer_id").iloc[
        offer_id].tolist() + user_data.set_index("user_id").iloc[user_id].tolist()

    # use model to predict classification for query
    classification_label = model.predict([query])[0]
    classification_result = {classification_label: classification_class[classification_label]}

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        classification_result=classification_result,
        user=user_id,
        offer=offer_id
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()

"""
{
            'data': [
                Bar(
                    x=category_names,
                    y=category_direct,
                    name="Direct"
                ),
                Bar(
                    x=category_names,
                    y=category_news,
                    name="News"
                ),
                Bar(
                    x=category_names,
                    y=category_social,
                    name="Social"
                )
            ],

            'layout': {
                'title': 'Distribution of Message Category',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category Name",
                    "automargin": True
                },
                'barmode': "stack"
            }
        }
"""
