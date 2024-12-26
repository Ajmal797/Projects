import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from datetime import datetime
import matplotlib.pyplot
from streamlit_option_menu import option_menu
import plotly.express as px
import pydeck as pdk
import seaborn as sns
import altair as alt

image=Image.open('imdb photo1.jpg')
st.image(image,caption='imdb photo1.jpg')

df=pd.read_csv('imdb fresh1.csv')
df
##prediction 1
st.subheader("predict the movie based on the rating")

cast_names = df["movie_title"].str.split(",", expand=True).stack().str.strip()
selected_cast = st.selectbox("Select a movie name", cast_names)


cast_data = df[df["movie_title"].str.contains(selected_cast)].sort_values(by=["movies_prediction under rating"], ascending=False)

rating = df[df["movie_title"].str.contains(selected_cast)].sort_values(by=["movies_prediction under rating"], ascending=False)["movies_prediction under rating"].iloc[0]
st.markdown(f"{selected_cast}'s movie rating is ({rating}).\n\nThe 0 and 1 will occur based on the movie rating, which is lesser than 8.")

## preditc 2
st.subheader("predict the movie based on the budget and gross")

cast_names = df["movie_title"].str.split(",", expand=True).stack().str.strip()

# Create a unique key based on the length of the cast_names array
widget_key = f"selectbox_{len(cast_names)}"

# Use the selectbox with the unique key
selected_cast = st.selectbox("Select a movie name", cast_names, key=widget_key)

# Filter data for the selected movie
cast_data = df[df["movie_title"].str.contains(selected_cast)].sort_values(by=["gross"], ascending=False)#to replace gross and give "movies next part based on budget is under 1lak"

st.write(f"Movies featuring {selected_cast}:")
for budget, gross in zip(cast_data["budget"], cast_data["gross"]):
    st.write(f"- The total budget: {budget}/ The total income: {gross} \n\n If the budget digit is higher than the gross dont take next season because it was seems risk,gross digit is higher than budget suggest to take next part")

##predict 3
st.subheader("predict the movie based on the user vote")

cast_names = df["movie_title"].str.split(",", expand=True).stack().str.strip().unique()

# Create a unique key based on the length of the cast_names array
widget_key = f"selectbox_{len(cast_names)}"

# Use the selectbox with the unique key
selected_cast = st.selectbox("Select a movie name", cast_names, key=widget_key)

# Filter data for the selected movie
cast_data = df[df["movie_title"].str.contains(selected_cast)].sort_values(by=["num_voted_users"], ascending=False)

st.write(f"Movies featuring {selected_cast}:")
for budget, users in zip(cast_data["budget"], cast_data["num_voted_users"]):
    st.write(f"- The total budget: {budget}/ The total users vote: {users}")
    if 150 * users < budget:
        #st.markdown(f"{selected_cast}'s movie rating is ({rating}).\n\nThe 0 and 1 will occur based on the movie rating, which is lesser than 8.")

        st.markdown("Warning: The movie is predicted to fail.\n\n""You have to take the next season at your own risk but not to take the next part is adviseble")
    else:
        st.write("the movie can get suceess")


###predict 4
#####jarcard
st.subheader("predict the movie based by using the jaccard techiniques")


selected_title = st.selectbox("Select a movie title", df["movie_title"])

# Get the genres for the selected title
selected_genres = set(df[df["movie_title"] == selected_title]["genres"].str.split(',').explode().str.strip())

# Calculate Jaccard similarity with other titles
similarities = {}
for title, genres in zip(df["movie_title"], df["genres"]):
    other_genres = set(genres.split(','))
    jaccard_similarity = len(selected_genres.intersection(other_genres)) / len(selected_genres.union(other_genres))
    similarities[title] = jaccard_similarity

# Count the number of movies similar and dissimilar
threshold_similarity = 0.5
num_similar = sum(similarity > threshold_similarity for _, similarity in similarities.items())
num_dissimilar = len(df) - num_similar
#st.write("Similarity Scores:")
#st.write(similarities)

# Predict the majority label
prediction = "there is no other movie similar in the genres " if num_similar > num_dissimilar else "there is lot of the simliar movies in the genres"

# Display concise output
st.write(f"Prediction using jaccard technique: \n\n  {selected_title}: {prediction}")

###predict 5
#### jacard recommendation
st.subheader("Give the recommendation based on the jaccard in buliding Film foresight Engine alogorithm ")

def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    similarity = intersection / union
    return similarity

def get_top_k_recommendations(target_movie, df, k=5):
    # Convert genres to sets
    df['genres_set'] = df['genres'].apply(lambda x: set(x.split(', ')))

    # Calculate Jaccard similarity between the target movie and all other movies
    df['similarity_score'] = df['genres_set'].apply(lambda x: jaccard_similarity(target_movie, x))

    # Sort movies by Jaccard similarity in descending order
    sorted_recommendations = df.sort_values(by='similarity_score', ascending=False)

    # Get the top K recommendations
    top_recommendations = sorted_recommendations.head(k)

    return top_recommendations[['movie_title', 'similarity_score','imdb_score']]

# Streamlit App
st.subheader("Movie Recommendation")

# Dropdown for selecting a movie
selected_movie = st.selectbox("Select a movie", df['movie_title'])

# Get movie recommendations based on Jaccard similarity
target_movie_genres = set(df.loc[df['movie_title'] == selected_movie, 'genres'].values[0].split(', '))
recommendations = get_top_k_recommendations(target_movie_genres, df, k=3)

# Display recommendations
st.subheader("Top Recommendations:")
st.dataframe(recommendations[['movie_title', 'similarity_score','imdb_score']])

# Additional section for the selected movie's details
st.subheader(f"Details for {selected_movie}:")
cast_data = df[df["movie_title"].str.contains(selected_movie)].sort_values(by=["movies_prediction under rating"], ascending=False)
rating = cast_data["movies_prediction under rating"].iloc[0]
st.markdown(f"{selected_movie}'s movie rating is ({rating}).\nThe 0 and 1 will occur based on the movie rating, which is lesser than 8.")

#####predict 6
st.subheader("prediction based on the movie budget lower than 1lakh can take next season")
selected_cast = st.selectbox("Select a movie name", df['movie_title'])

# Filter data for the selected movie
cast_data = df[df["movie_title"].str.contains(selected_cast)].sort_values(by=["movies next part based on budget is under 1lak"], ascending=False)

# Display information about the selected movie
st.write(f"Movies featuring {selected_cast}:")
for budget, movies in zip(cast_data["budget"], cast_data["movies next part based on budget is under 1lak"]):
    st.write(f" ({movies}) ")
    if movies==0:
        st.write("That might be a Risk")
    else:
        st.write("There is no risk")

####prediction 7
st.subheader("The director selecting the actor based on the actor Facebook popularity")

selected_title = st.selectbox("Select a movie title for actor 1", df["actor_1_name"])

c_vote_data = df[df["actor_1_name"].str.contains(selected_title)].sort_values(by=["movie season 2 predict based on the actor 1fb likes"], ascending=False)

# Check if any 0 value is present in the column
if 0 in c_vote_data["movie season 2 predict based on the actor 1fb likes"].values:
    st.write("That director choice might be a risk")
else:
    st.write("The director can choose the actor")

#prediction 8
selected_title = st.selectbox("Select a movie title for actor 2", df["actor_2_name"])

c_vote_data = df[df["actor_2_name"].str.contains(selected_title)].sort_values(by=["movie season 2 predict based on the actor 2 fb likes"], ascending=False)

# Check if any 0 value is present in the column
if 0 in c_vote_data["movie season 2 predict based on the actor 2 fb likes"].values:
    st.write("That director choice might be a risk")
else:
    st.write("The director can choose the actor")

##########################################################################################################

###########################################################################
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor

# Streamlit App
st.title("IMDb Rating Prediction for Next Season")

# Example: Input for the new season data
new_season_data_key = "new_season_data_input"
new_season_data = st.text_input("Enter genres for the new season (comma-separated):", "Action,Adventure,Sci-Fi", key=new_season_data_key)

# Preprocessing: Creating dummy variables for each genre
genre_dummies = df['genres'].str.get_dummies(sep='|')

# Joining the dummy variables with the imdb_score
model_data = genre_dummies.join(df['imdb_score'])

# Splitting the dataset into training and testing sets
X = model_data.drop('imdb_score', axis=1)
y = model_data['imdb_score']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training a Linear Regression model
lr_regressor = LinearRegression()
lr_regressor.fit(X_train, y_train)

# Make predictions on the test set for Linear Regression
y_pred_lr = lr_regressor.predict(X_test)

# Evaluate the Linear Regression model
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)
rmse_lr = np.sqrt(mse_lr)


# Display the evaluation metrics for Linear Regression
st.subheader("Linear Regression Model Evaluation")
st.write(f"Mean Squared Error (Linear Regression): {mse_lr}")
st.write(f"R-squared (Linear Regression): {r2_lr}")
st.write(f"Root Mean Squared Error (Linear Regression): {rmse_lr}")

st.write("Linear Regression has good accuracy so linear regression prediction is advisable")

# Training a Decision Tree Regressor
dt_regressor = DecisionTreeRegressor(random_state=42)
dt_regressor.fit(X_train, y_train)

# Make predictions on the test set for Decision Tree
y_pred_dt = dt_regressor.predict(X_test)

# Evaluate the Decision Tree model
mse_dt = mean_squared_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)
rmse_dt = np.sqrt(mse_dt)
# Display the evaluation metrics for Decision Tree
st.subheader("Decision Tree Model Evaluation")
st.write(f"Mean Squared Error (Decision Tree): {mse_dt}")
st.write(f"R-squared (Decision Tree): {r2_dt}")
st.write(f"Root Mean Squared Error (Linear Regression): {rmse_dt}")

# Training a Random Forest Regressor
rf_regressor = RandomForestRegressor(random_state=42)
rf_regressor.fit(X_train, y_train)

# Make predictions on the test set for Random Forest
y_pred_rf = rf_regressor.predict(X_test)

# Evaluate the Random Forest model
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)
rmse_rf = np.sqrt(mse_rf)

# Display the evaluation metrics for Random Forest
st.subheader("Random Forest Model Evaluation")
st.write(f"Mean Squared Error (Random Forest): {mse_rf}")
st.write(f"R-squared (Random Forest): {r2_rf}")
st.write(f"Root Mean Squared Error (Random Forest): {rmse_rf}")

########## mlp regressor
#mlp_regressor = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
#mlp_regressor.fit(X_train, y_train)
#y_pred_mlp = mlp_regressor.predict(X_test)
#mse_mlp = mean_squared_error(y_test, y_pred_mlp)
#rmse_mlp = np.sqrt(mse_mlp)
#r2_mlp = r2_score(y_test, y_pred_mlp)
#### display evolution
#st.subheader("MLP Model Evaluation")
#st.write(f"Mean Squared Error (MLP): {mse_mlp}")
#st.write(f"R-squared (MLP): {r2_mlp}")
#st.write(f"Root Mean Squared Error (MLP): {rmse_mlp}")


# Preprocess the new season data using NumPy
new_genre_dummies = pd.DataFrame(0, index=np.arange(1), columns=genre_dummies.columns)
for genre in new_season_data.split(','):
    new_genre_dummies[genre] = 1

# Make predictions for the new season using Linear Regression
predicted_rating_lr = lr_regressor.predict(new_genre_dummies)[0]

# Make predictions for the new season using Decision Tree
predicted_rating_dt = dt_regressor.predict(new_genre_dummies)[0]

# Make predictions for the new season using Random Forest
predicted_rating_rf = rf_regressor.predict(new_genre_dummies)[0]
#Make predictions for the new season using MLP
#predicted_rating_mlp = mlp_regressor.predict(new_genre_dummies)[0]
#############################jaccard
from sklearn.metrics import jaccard_score
# Calculate Jaccard similarities for Linear Regression
jaccard_lr = [jaccard_score(X_train.iloc[i], new_genre_dummies.values.flatten(), average='micro') for i in range(X_train.shape[0])]

# Calculate Jaccard similarities for Decision Tree
jaccard_dt = [jaccard_score(X_train.iloc[i], new_genre_dummies.values.flatten(), average='micro') for i in range(X_train.shape[0])]

# Combine the Jaccard similarities with the movie titles
movie_titles = df['movie_title']
jaccard_similarity_dict = dict(zip(movie_titles, jaccard_lr))  # Use either jaccard_lr or jaccard_dt based on your preference

# Sort the movies based on Jaccard similarity in descending order
sorted_movies = sorted(jaccard_similarity_dict.items(), key=lambda x: x[1], reverse=True)

# Display the top 3 recommended movies in a table
st.subheader("Top 3 Recommended Movies Based on Jaccard Similarity")
table_data = [{"Movie Title": movie[0], "Jaccard Similarity": movie[1]} for movie in sorted_movies[:3]]
st.table(pd.DataFrame(table_data))


# Display the predicted ratings for the new season
st.subheader("Predicted IMDb Ratings for the Next Season:")
st.write(f"Linear Regression based movie rating: <span style='color:green'>{predicted_rating_lr:.2f}</span>", unsafe_allow_html=True)
if predicted_rating_lr <= 7:
    st.write("Warning: Film Foresight Engine Algorithm concludes this movie genre is not advisable for the next season")
else:
    st.write("Success: Film Foresight Engine algorithm recommends taking the next season")

st.write("This prediction is only based on the genre not for the story or cast ")
st.write(f"Decision Tree based movie rating: <span style='color:green'>{predicted_rating_dt:.2f}</span>", unsafe_allow_html=True)
if predicted_rating_dt <= 7:
    st.write("Warning: Film Foresight Engine Algorithm concludes this movie genre is not advisable for the next season")
else:
    st.write("Success: Film Foresight Engine algorithm recommends taking the next season")

st.write(f"Random Forest based movie rating: <span style='color:green'>{predicted_rating_rf:.2f}</span>", unsafe_allow_html=True)
if predicted_rating_rf <= 7:
    st.write("Warning: Film Foresight Engine Algorithm concludes this movie genre is not advisable for the next season")
else:
    st.write("Success: Film Foresight Engine algorithm recommends taking the next season")

#st.write(f"MLP based movie rating: <span style='color:green'>{predicted_rating_mlp:.2f}</span>", unsafe_allow_html=True)
#if predicted_rating_mlp <= 7:
#    st.write("Warning: Film Foresight Engine Algorithm concludes this movie genre is not advisable for the next season")
#else:
#    st.write("Success: Film Foresight Engine algorithm recommends taking the next season")


prediction_text = "NOTE: This prediction is only based on the genre not for the story or cast"

# Display the text in red color
st.markdown(f'<p style="color: red;">{prediction_text}</p>', unsafe_allow_html=True)


######################################################################
st.subheader("based on genre forecasting ")
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Assuming df is your DataFrame with the necessary data
# You might need to adjust the column names according to your DataFrame
df['title_year'] = pd.to_datetime(df['title_year'], format='%Y')

# Example: Input for the genre
selected_genre = st.text_input("Enter the genre:", "Action|Adventure|Fantasy|Sci-Fi")

# Example: Input for the time range
start_year = st.slider("Select the start year:", min_value=int(df['title_year'].dt.year.min()), max_value=int(df['title_year'].dt.year.max()), value=1930)
end_year = st.slider("Select the end year:", min_value=start_year, max_value=int(df['title_year'].dt.year.max()), value=2020)

# Filter data based on the selected genre and time range
filtered_data = df[(df['genres'].str.contains(selected_genre)) & (df['title_year'].dt.year.between(start_year, end_year))]

# Group data by year and calculate the average IMDb score for each year
grouped_data = filtered_data.groupby(filtered_data['title_year'].dt.year)['imdb_score'].mean().reset_index()

# Prepare the data for forecasting
data = grouped_data.rename(columns={'title_year': 'ds', 'imdb_score': 'y'})

# Create and fit the Exponential Smoothing model
model = ExponentialSmoothing(data['y'], trend='add', seasonal='add', seasonal_periods=12)  # Assuming yearly seasonality
fit_model = model.fit()

# Make predictions for the future
forecast_start_year = 2025
forecast_end_year = 2040
forecast_period = forecast_end_year - forecast_start_year + 1
future_years = pd.date_range(start=str(forecast_start_year), periods=forecast_period, freq='Y')  # Forecast from 2025 to 2040
forecast = fit_model.forecast(steps=len(future_years))

# Plot only the forecast with a line chart
fig, ax = plt.subplots()
ax.plot(future_years, forecast, label='Forecasted IMDb Score', color='red')
ax.set_xlabel('Year')
ax.set_ylabel('IMDb Score')
ax.set_title(f'IMDb Score Forecast for {selected_genre} Genre from {start_year} to {end_year} and Forecast from {forecast_start_year} to {forecast_end_year}')
ax.legend()

# Display the chart using Streamlit
st.pyplot(fig)
############################################
