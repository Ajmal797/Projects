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

df=pd.read_csv('imdb fresh1.csv')
st.title('conclusion ')
image=Image.open('imdb photo 4.jpg')
st.image(image,caption='imdb photo 4.jpg')

st.write("The goal of this all-encompassing strategy is to give directors, producers, and actors insightful information. Industry experts may make well-informed decisions regarding upcoming projects with the help of our Film Foresight Engine, which uses predictive modelling and collaborative filtering through Jaccard similarity. The boosting techniques used provide improved prediction power, leading to a more sophisticated comprehension of the elements that influence a movie's success.By the time this project comes to a finish, the convergence of these methods will provide industry stakeholders with a thorough grasp of the best movie, director, and actor pairings for upcoming projects. This multifaceted strategy seeks to support informed decision-making and contribute to the continuous success of cinematic endeavors by acting as a useful tool for decision-makers in the film industry.")
st.subheader("Challenges")
st.write("•	This film foresight engine was used only if the columns are correctly presented.\n\n•	Model Performance: To improve accuracy, it could be necessary to further optimize the current models.\n\n•	Restricted Features: To improve forecast accuracy, more features than just genres can be required.\n\n•	User feedback: To improve the recommendation system, user feedback is gathered.")
st.subheader("Future Enchancement")
st.write("Now this project is only done for movies we can exnchance this to do for tv series,reality shows,music shows and other ott platforms")