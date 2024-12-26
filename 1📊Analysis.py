import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from datetime import datetime
from streamlit_option_menu import option_menu
import plotly.express as px
import pydeck as pdk
import seaborn as sns
import altair as alt
from wordcloud import WordCloud, STOPWORDS
import pandas as pd
import matplotlib.pyplot as plt


image=Image.open('imdb photo 3.jpg')
st.image(image,caption='imdb photo 3.jpg')

df=pd.read_csv('imdb fresh1.csv')
#----------1graph
st.subheader("GROSS and BUDGET ")
fig= px.bar (df.head(15),x="budget",y= "gross" ,orientation='v' )
st.plotly_chart(fig)


#---------2graph
st.subheader("MOVIE and IMDB RATING")
top10 = df.sort_values(by=["imdb_score"], ascending=False).head(10)

chart = alt.Chart(top10).mark_bar().encode(
    x=alt.X("movie_title", sort="-y"),
    y=alt.Y("imdb_score", scale=alt.Scale(domain=(0, 10))),
    color=alt.Color("imdb_score", scale=alt.Scale(scheme="reds")),
    tooltip=["movie_title", "imdb_score"]
)

st.altair_chart(chart, use_container_width=True)
# Get top 10 titles and gross earnings
#-----------3
st.subheader("GROSS OF THE MOVIE")
chart = alt.Chart(top10).mark_bar().encode(
    x=alt.X("movie_title", sort="-y"),
    y=alt.Y("gross", axis=alt.Axis(format="$", title="Gross Earnings")),
    color=alt.Color("gross", scale=alt.Scale(scheme="greens")),
    tooltip=["movie_title", alt.Tooltip("gross", format="$,.2f")]
)

st.altair_chart(chart, use_container_width=True)

# Create select box
#actor 1
st.subheader("THE ACTOR BASED ON THE MOVIE AND IMDB RATING")
cast_names = df["actor_1_name"].str.split(",", expand=True).stack().str.strip().unique()
selected_cast = st.selectbox("Select a cast member", cast_names)

# Filter data for selected cast member
cast_data = df[df["actor_1_name"].str.contains(selected_cast)].sort_values(by=["imdb_score"], ascending=False)

# Create bar chart


chart = alt.Chart(cast_data).mark_rect().encode(
    x=alt.X("movie_title", sort="-y"),
    y=alt.Y("imdb_score", axis=alt.Axis(title="rating")),
    tooltip=["movie_title", "imdb_score"]
)

# Display rating for selected cast member
rating = df[df["actor_1_name"].str.contains(selected_cast)].sort_values(by=["imdb_score"], ascending=False)["imdb_score"].iloc[0]
st.write(f"{selected_cast}'s highest rated movie is {rating}")

# Display chart
st.altair_chart(chart, use_container_width=True)


# display movie and rating
st.write(f"Movies featuring {selected_cast}:")
for title, rating in zip(cast_data["movie_title"], cast_data["imdb_score"]):
    st.write(f"- {title}: {rating}")

### actor 2 
cast_names = df["actor_2_name"].str.split(",", expand=True).stack().str.strip().unique()
selected_cast = st.selectbox("Select a cast member", cast_names)

# Filter data for selected cast member
cast_data = df[df["actor_2_name"].str.contains(selected_cast)].sort_values(by=["imdb_score"], ascending=False)

# Create bar chart
chart = alt.Chart(cast_data).mark_area(color="purple").encode(
    x=alt.X("movie_title", sort="-y"),
    y=alt.Y("imdb_score", axis=alt.Axis(title="rating")),
    tooltip=["movie_title", "imdb_score"]
)
# Display rating for selected cast member
rating = df[df["actor_2_name"].str.contains(selected_cast)].sort_values(by=["imdb_score"], ascending=False)["imdb_score"].iloc[0]
st.write(f"{selected_cast}'s highest rated movie is {rating}")

# Display chart
st.altair_chart(chart, use_container_width=True)

st.write(f"Movies featuring {selected_cast}:")
for title, rating in zip(cast_data["movie_title"], cast_data["imdb_score"]):
    st.write(f"- {title}: {rating}")


### actor 3 
cast_names = df["actor_3_name"].str.split(",", expand=True).stack().str.strip().unique()
selected_cast = st.selectbox("Select a cast member", cast_names)

# Filter data for selected cast member
cast_data = df[df["actor_3_name"].str.contains(selected_cast)].sort_values(by=["imdb_score"], ascending=False)

# Create bar chart
chart = alt.Chart(cast_data).mark_bar(color='green').encode(
    x=alt.X("movie_title", sort="-y"),
    y=alt.Y("imdb_score", axis=alt.Axis(title="rating")),
    tooltip=["movie_title", "imdb_score"]
)
# Display rating for selected cast member
rating = df[df["actor_3_name"].str.contains(selected_cast)].sort_values(by=["imdb_score"], ascending=False)["imdb_score"].iloc[0]
st.write(f"{selected_cast}'s highest rated movie is {rating}")

# Display chart
st.altair_chart(chart, use_container_width=True)

st.write(f"Movies featuring {selected_cast}:")
for title, rating in zip(cast_data["movie_title"], cast_data["imdb_score"]):
    st.write(f"- {title}: {rating}")


### director
st.subheader("DIRECTOR AND HIS MOVIES BASED ON THE IMDB RATING ")
direct_names = df["director_name"].str.split(",", expand=True).stack().str.strip().unique()
selected_cast = st.selectbox("Select a director name", direct_names)

# Filter data for selected cast member
cast_data = df[df["director_name"].str.contains(selected_cast)].sort_values(by=["imdb_score"], ascending=False)

# Create bar chart
chart = alt.Chart(cast_data).mark_bar(color='orange').encode(
    x=alt.X("movie_title", sort="-y"),
    y=alt.Y("imdb_score", axis=alt.Axis(title="rating")),
    tooltip=["movie_title", "imdb_score"]
)
# Display rating for selected cast member
rating = df[df["director_name"].str.contains(selected_cast)].sort_values(by=["imdb_score"], ascending=False)["imdb_score"].iloc[0]
st.write(f"{selected_cast}'s highest rated movie is {rating}")

# Display chart
st.altair_chart(chart, use_container_width=True)

st.write(f"Movies featuring {selected_cast}:")
for title, rating in zip(cast_data["movie_title"], cast_data["imdb_score"]):
    st.write(f"- {title}: {rating}")

### budget and gross
st.subheader("BUDGET AND GROSS OF THE MOVIE ")
movie_names = df["movie_title"].str.split(",", expand=True).stack().str.strip()
selected_cast = st.selectbox("Select a movie name", movie_names)

# Filter data for selected cast member
cast_data = df[df["movie_title"].str.contains(selected_cast)].sort_values(by=["gross"], ascending=False)

st.write(f"Movies featuring {selected_cast}:")
for budget, gross in zip(cast_data["budget"], cast_data["gross"]):
    st.write(f"- The total budget of the movie {budget}/ The total income that the movie can give {gross}")

################################################################################################################################################
st.subheader("FIND THE PEOPLE FAVORITE AND CRITIC FAVORITE FOR AWARD")
cast_names = df["actor_1_name"].str.split(",", expand=True).stack().str.strip()
selected_cast = st.selectbox("Select a cast member", cast_names)


cast_data = df[df["actor_1_name"].str.contains(selected_cast)].sort_values(by=["imdb_score"], ascending=False)


chart = alt.Chart(cast_data).mark_area(color='cyan').encode(
    x=alt.X("movie_title", sort="-y"),
    y=alt.Y("num_critic_for_reviews", axis=alt.Axis(title="votes")),
    tooltip=["movie_title", "num_critic_for_reviews"]

)
st.altair_chart(chart, use_container_width=True)
chart = alt.Chart(cast_data).mark_area(color='cyan').encode(
    x=alt.X("movie_title", sort="-y"),
    y=alt.Y("num_voted_users", axis=alt.Axis(title="votes")),
    tooltip=["movie_title", "num_voted_users"]

)
st.altair_chart(chart, use_container_width=True)

c_vote = df[df["actor_1_name"].str.contains(selected_cast)].sort_values(by=["num_critic_for_reviews"], ascending=False)["num_critic_for_reviews"].iloc[0]
st.write(f"Movies featuring {selected_cast}:")

for critic, uservote in zip(cast_data["num_critic_for_reviews"], cast_data["num_voted_users"]):
    st.write(f"- The total critic vote  {critic}/ The total fan vote {uservote}")
##########################################################################
cast_names = df["actor_2_name"].str.split(",", expand=True).stack().str.strip()
selected_cast = st.selectbox("Select a cast member", cast_names)


cast_data = df[df["actor_2_name"].str.contains(selected_cast)].sort_values(by=["imdb_score"], ascending=False)


chart = alt.Chart(cast_data).mark_rect(color='cyan').encode(
    x=alt.X("movie_title", sort="-y"),
    y=alt.Y("num_critic_for_reviews", axis=alt.Axis(title="votes")),
    tooltip=["movie_title", "num_critic_for_reviews"]

)
st.altair_chart(chart, use_container_width=True)
chart = alt.Chart(cast_data).mark_rect(color='cyan').encode(
    x=alt.X("movie_title", sort="-y"),
    y=alt.Y("num_voted_users", axis=alt.Axis(title="votes")),
    tooltip=["movie_title", "num_voted_users"]

)
st.altair_chart(chart, use_container_width=True)

c_vote = df[df["actor_2_name"].str.contains(selected_cast)].sort_values(by=["num_critic_for_reviews"], ascending=False)["num_critic_for_reviews"].iloc[0]
st.write(f"Movies featuring {selected_cast}:")

for critic, uservote in zip(cast_data["num_critic_for_reviews"], cast_data["num_voted_users"]):
    st.write(f"- The total critic vote  {critic}/ The total fan vote {uservote}")
############################################################################################################################
cast_names = df["actor_3_name"].str.split(",", expand=True).stack().str.strip()
selected_cast = st.selectbox("Select a cast member", cast_names)


cast_data = df[df["actor_3_name"].str.contains(selected_cast)].sort_values(by=["imdb_score"], ascending=False)


chart = alt.Chart(cast_data).mark_bar(color='cyan').encode(
    x=alt.X("movie_title", sort="-y"),
    y=alt.Y("num_critic_for_reviews", axis=alt.Axis(title="votes")),
    tooltip=["movie_title", "num_critic_for_reviews"]

)
st.altair_chart(chart, use_container_width=True)
chart = alt.Chart(cast_data).mark_bar(color='cyan').encode(
    x=alt.X("movie_title", sort="-y"),
    y=alt.Y("num_voted_users", axis=alt.Axis(title="votes")),
    tooltip=["movie_title", "num_voted_users"]

)
st.altair_chart(chart, use_container_width=True)

c_vote = df[df["actor_3_name"].str.contains(selected_cast)].sort_values(by=["num_critic_for_reviews"], ascending=False)["num_critic_for_reviews"].iloc[0]
st.write(f"Movies featuring {selected_cast}:")

for budget, gross in zip(cast_data["num_critic_for_reviews"], cast_data["num_voted_users"]):
    st.write(f"- The total critic vote  {budget}/ The total fan vote {gross}")


#############
st.subheader("TOP 250 MOVIES BASED ON THE IMDB RATING ")
criterion = st.selectbox("Select a criterion for top movies", ["imdb_score"])

# Get the top 250 movies based on the selected criterion
top_250 = df.sort_values(by=criterion, ascending=False).head(250)

# Display the top 250 movies
st.subheader(f"Top 250 Movies based on {criterion}:")
st.dataframe(top_250[['movie_title', criterion]])

######################################################################
st.subheader("THE HIGHEST WORD THAT THE GENRE TAKES PLACES")
image=Image.open('word cloud.png')
st.subheader("Word Cloud for Movie Genres")
st.image(image,caption='word cloud.png')




st.subheader("DIRECTOR INFORMATION FOR MOVIE GENRE")

# User input for movie genre
genre_input = st.text_input("Enter a movie genre:", "Action|Adventure|Fantasy|Sci-Fi")

# Filter DataFrame based on the selected genre
selected_movies = df[df['genres'].str.contains(genre_input, case=False, na=False)]

# Display director information for the selected genre
if not selected_movies.empty:
    st.subheader(f"Director Information for {genre_input} Movies")

    # Sort by director_facebook_likes in descending order and get the top 10 directors
    top_directors = selected_movies.sort_values(by="director_facebook_likes", ascending=False).head(10)

    # Altair Chart
    chart = alt.Chart(top_directors).mark_bar(cornerRadius=0).encode(
    x=alt.X("director_name", sort="-y"),
    y=alt.Y("director_facebook_likes", axis=alt.Axis(title="Director Facebook Likes")),
    color=alt.Color("director_facebook_likes", scale=alt.Scale(scheme="greens")),
    tooltip=["director_name", "director_facebook_likes"]
)

    # Streamlit Chart Display
    st.altair_chart(chart, use_container_width=True)
else:
    st.warning(f"No movies found for the genre: {genre_input}")




###############################
selected_actor = st.selectbox("Select Actor 1:", df['actor_1_name'].unique())

# Filter the DataFrame based on the selected actor
filtered_df = df[df['actor_1_name'] == selected_actor]

# Create a clustered bar chart using Altair
chart = alt.Chart(filtered_df).mark_bar().encode(
    x='movie_title',
    y='gross',
    color=alt.Color('budget', scale=alt.Scale(scheme='viridis')),
    tooltip=['movie_title', 'gross', 'budget']
).properties(
    width=700,
    height=400,
    title=f'Gross and Budget for Movies Starring {selected_actor}'
)

# Display the chart using Streamlit
st.altair_chart(chart, use_container_width=True)

