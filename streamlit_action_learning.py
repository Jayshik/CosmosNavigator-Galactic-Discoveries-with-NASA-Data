import psycopg2
import hashlib
import streamlit as st
import torch
import tensorflow as tf
import joblib
import torch.nn as nn
import pandas as pd
from PIL import Image
import numpy as np
from sklearn.impute import SimpleImputer
import random
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timezone

#---------------------------------------------------DATABASE---------------------------------------------------------
# Function to create a database connection
def create_connection():
    return psycopg2.connect(
        dbname="diabetes",
        user="postgres",
        password="root",
        host="localhost",
    )


# Function to create a table for users if it doesn't exist
def create_table(conn):
    create_query = """
    CREATE TABLE IF NOT EXISTS users (
        id SERIAL PRIMARY KEY,
        username TEXT NOT NULL,
        password TEXT NOT NULL
    );
    """
    with conn.cursor() as cursor:
        cursor.execute(create_query)
    conn.commit()


# Function to hash the password before storing it in the database
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


# Function to check if the entered password matches the hashed password in the database
def verify_password(stored_password, entered_password):
    return stored_password == hashlib.sha256(entered_password.encode()).hexdigest()


# Function to create a new user account
def create_user(username, password):
    conn = create_connection()
    with conn:
        with conn.cursor() as cursor:
            insert_query = "INSERT INTO users (username, password) VALUES (%s, %s)"
            hashed_password = hash_password(password)
            cursor.execute(insert_query, (username, hashed_password))
    conn.commit()


# Function to check if the username exists in the database
def username_exists(username):
    conn = create_connection()
    with conn:
        with conn.cursor() as cursor:
            query = "SELECT * FROM users WHERE username = %s"
            cursor.execute(query, (username,))
            return cursor.fetchone() is not None
#------------------------------------------MODEL LOADING-----------------------------------------------------------

def preprocess_image(image):
    image = image.resize((256, 256))
    image_array = np.array(image)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


class NN3HiddenLayer(torch.nn.Module):

    def __init__(self, act_function=nn.Sigmoid(), input_size=37):
        super(NN3HiddenLayer, self).__init__()

        self.input_size = input_size
        self.output_size = 1

        self.act_function = act_function

        self.input = nn.Linear(self.input_size, 20)
        self.hl1 = act_function
        self.linear_hl2 = nn.Linear(20, 10)
        self.hl2 = act_function
        self.linear_hl3 = nn.Linear(10, 5)
        self.hl3 = act_function
        self.linear_hl4 = nn.Linear(5, self.output_size)
        self.output = nn.Sigmoid()

    def forward(self, x):
        x = self.input(x)
        x = self.hl1(x)
        x = self.linear_hl2(x)
        x = self.hl2(x)
        x = self.linear_hl3(x)
        x = self.hl3(x)
        x = self.linear_hl4(x)
        return self.output(x)


# Load the PyTorch model
pytorch_model = NN3HiddenLayer()
state_dict = torch.load('MODELS/pytorch_model.pth')
pytorch_model.load_state_dict(state_dict)
pytorch_model.eval()

# Load the TensorFlow model
tensorflow_model = tf.keras.models.load_model('MODELS/tensorflow_model.h5')

# Load the scikit-learn model
scikit_model = joblib.load('MODELS/clustering_model.joblib')

def set_background_image(image_path):
    style = f"""
        <style>
        body {{
            background-image: url("{image_path}");
            background-size: cover;
            background-repeat: no-repeat;
        }}
        </style>
    """
    return style

def get_session_state():
    return st.session_state.setdefault("session_state", {})
#------------------------------------------------------LOGIN PAGE------------------------------------------------

def login_page(session_state):

    background_image_path = "pictures/picture1.jpg"  # Replace with the actual path
    st.markdown(set_background_image(background_image_path), unsafe_allow_html=True)

    image1 = Image.open('pictures/12.jpg')

    # Display the images
    st.image([image1], width=900)

    st.header("Sign In")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Sign In"):
        if username and password:
            conn = create_connection()
            with conn:
                with conn.cursor() as cursor:
                    query = "SELECT * FROM users WHERE username = %s"
                    cursor.execute(query, (username,))
                    user = cursor.fetchone()
                    if user and verify_password(user[2], password):
                        session_state['user_authenticated'] = True
                        st.experimental_rerun()  # Rerun the entire app to display the Home Screen
                    else:
                        st.error("Invalid username or password. Please try again.")
        else:
            st.warning("Please enter a username and password.")

#-------------------------------------------------SIGN UP PAGE -------------------------------------------------
def sign_up_page(session_state):
    background_image_path = "pictures/picture1.jpg"  # Replace with the actual path
    st.markdown(set_background_image(background_image_path), unsafe_allow_html=True)

    image1 = Image.open('pictures/3.jpg')

    # Display the images
    st.image([image1], width=900)

    st.header("Sign Up")
    new_username = st.text_input("Username")
    new_password = st.text_input("Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")

    if st.button("Sign Up"):
        if new_username and new_password and confirm_password:
            if new_password == confirm_password:
                if not username_exists(new_username):
                    create_user(new_username, new_password)
                    st.success("Account created! You can now sign in.")
                else:
                    st.error("Username already exists. Please choose a different username.")
            else:
                st.error("Passwords do not match. Please try again.")
        else:
            st.warning("Please enter a username and password.")

#-----------------------------------------------HOME PAGE ------------------------------------------------

def home_page(session_state):
    background_image_path = "pictures/picture1.jpg"  # Replace with the actual path
    st.markdown(set_background_image(background_image_path), unsafe_allow_html=True)

    image1 = Image.open('pictures/2.webp')

    # Display the images
    st.image([image1], width=900)

    model_choice = st.selectbox('Select Model', ['ANN', 'CNN', 'Clustering'])
    class_labels = {
        0: "Super Earth : These are rocky planets larger than Earth but smaller than Uranus and Neptune. They could potentially be more likely to support life as we know it, but it's still an active field of research.",
        1: "Hot Jupyter : These are gas giant planets, similar in characteristics to Jupiter, but they orbit very close to their star, leading to high temperatures.",
        2: "Iorn Planets : These are planets composed primarily of iron and are thought to form in high-temperature conditions where rocky materials have evaporated, leaving behind iron-rich cores.",
        3: "Lava Worlds : These are terrestrial planets so close to their stars that their surface is molten.",
        4: "Rogue Planets : These are planets that do not orbit any star and drift through space on their own. They can be ejected from their original planetary systems during the formation and evolution of a system.",
        5: "Moon : Its just moon",
        6: "Mini Neptunes : These are smaller versions of Neptune and are believed to have a solid core surrounded by an atmosphere of gas or ice. They are larger than Earth but smaller than the gas giants in our solar system.",
        7: "Ice Gaints : Similar to Uranus and Neptune in our solar system, these exoplanets are believed to have a core of rock and metal, surrounded by a mantle of water, ammonia, or methane.",
        8: "Gas Dwarfs : This is a category of planets that are smaller than gas giants like Jupiter and Saturn, but larger than terrestrial planets like Earth. Their composition is largely made up of hydrogen and helium.",
        9: "Pulser Planets : These are planets that orbit around a pulsar, which is a rapidly rotating neutron star. These are quite rare and their existence challenges current theories of planet formation.",
        10: "Circum Binary Planets : These are planets that orbit two stars rather than one, similar to Tatooine in Star Wars."
    }

    if model_choice == 'ANN':
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)

            if st.button("Predict"):
                data_tensor = torch.tensor(data.values, dtype=torch.float)

                prediction = pytorch_model(data_tensor)
                st.write(f'Prediction (PyTorch): {prediction}')

        # Convert data_tensor to DataFrame
                data_df = pd.DataFrame(data_tensor.numpy(), columns=data.columns)

        # Add the 'Prediction' column to the DataFrame
                data_df['Prediction'] = prediction

                st.write(" Predictions (PyTorch):")
                st.table(data_df)

    elif model_choice == 'CNN':
        uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_image is not None:
            # Preprocess the image (if required)
            img = Image.open(uploaded_image)
            if st.button("Predict"):
                img_array = preprocess_image(img)
                # Make prediction using the TensorFlow model
                prediction = tensorflow_model.predict(img_array)
                predicted_class = np.argmax(prediction, axis=1)[0]

                # Display the prediction to the user
                st.image(img, caption='Uploaded Image', use_column_width=True)
                predicted_label = class_labels[predicted_class]
                st.markdown(f"## Predicted {predicted_label}")
                st.markdown(f"## Predicted Class: {predicted_class}")
                # st.write(f'Predicted Planet - {predicted_label}')
                # st.write(f'Predicted Class {predicted_class}')

    elif model_choice == 'Clustering':
        uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
        if uploaded_file is not None:
            # Preprocess the data (if required)
            data = pd.read_csv(uploaded_file)

            if st.button("Predict"):
                if 'PlanetaryMassJpt' in data.columns and 'SemiMajorAxisAU' in data.columns:
                    # Get the required features for clustering
                    imputer = SimpleImputer(strategy='mean')
                    imputer.fit(data[['PlanetaryMassJpt', 'SemiMajorAxisAU']])
                    # Impute missing values with the calculated mean
                    data[['PlanetaryMassJpt', 'SemiMajorAxisAU']] = imputer.transform(
                        data[['PlanetaryMassJpt', 'SemiMajorAxisAU']])

                    logM = np.log10(data['PlanetaryMassJpt'])
                    logD = np.log10(data['SemiMajorAxisAU'])
                    # Combine the features into the input matrix
                    X = np.array([logM, logD]).T

                    if not np.isnan(X).any():

                        scikit_model = joblib.load('MODELS/clustering_model.joblib')

                        prediction = scikit_model.predict(X)
                        data['label'] = prediction
                        st.write("Found", len(np.unique(prediction)), "clusters.")
                        st.markdown(f"## we have 6 clusters based on the habitability criteria , such as temperature surface, distance from nearest star,star mass,planetary atmosphere..: cluster 1 ,2 has relatively more chnaces of habitability , others have low chance")
                        st.write("Data with cluster labels:")
                        st.write(data)
                    else:
                        st.write(
                            "Error: Input data contains missing values (NaN). Please preprocess the data to handle missing values.")
                else:
                    st.write(
                        "Error: Required columns 'PlanetaryMassJpt' and 'SemiMajorAxisAU' not found in the uploaded CSV file.")

    if st.button("Logout"):
        session_state = get_session_state()
        session_state['user_authenticated'] = False
        st.experimental_rerun()

#--------------------------------------------------ARTICLES PAGE--------------------------------------------------

def get_latest_articles(url, num_articles=5):
    # Fetch the HTML content from the latest news page URL
    response = requests.get(url)
    response.raise_for_status()

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.text, "html.parser")

    # Find the latest article links
    article_links = soup.find_all("a", class_="article-link")
    latest_article_links = [link["href"] for link in article_links[:num_articles]]

    return latest_article_links

def format_published_date(published_date):
    try:
        published_datetime = datetime.fromisoformat(published_date.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        time_difference = now - published_datetime

        if time_difference.days > 0:
            return f"{published_datetime.strftime('%B %d, %Y')}"
        elif time_difference.seconds // 3600 > 0:
            return f"{time_difference.seconds // 3600} hours ago"
        elif time_difference.seconds // 60 > 0:
            return f"{time_difference.seconds // 60} minutes ago"
        else:
            return "a few seconds ago"
    except:
        return "Unknown date"

def get_article_content(url, num_paragraphs=5):
    # Fetch the HTML content from the URL
    response = requests.get(url)
    response.raise_for_status()

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.text, "html.parser")

    # Find the article content by identifying the relevant HTML tags
    article_element = soup.find("article")
    if article_element:
        # Extract the title
        title_element = article_element.find("h1")
        title = title_element.get_text() if title_element else "No title found"

        # Extract the author and publication date
        author_element = article_element.find("span", class_="author-byline__author-name")
        author = author_element.get_text() if author_element and not author_element.get_text().isspace() else "Unknown author"

        date_element = article_element.find("time", class_="relative-date")
        date = date_element["datetime"] if date_element else "Unknown date"
        formatted_date = format_published_date(date)

        # Fetch the Open Graph metadata to get the main image URL and its caption
        og_image_element = soup.find("meta", property="og:image")
        og_caption_element = soup.find("span", class_="caption-text")
        og_image_url = og_image_element["content"] if og_image_element else ""
        og_caption = og_caption_element.get_text() if og_caption_element else ""

        # Extract the introductory paragraph before the picture
        intro_paragraph_element = article_element.find("p", class_="strapline")
        intro_paragraph = intro_paragraph_element.get_text() if intro_paragraph_element and not intro_paragraph_element.get_text().isspace() else ""

        # Combine title, author, date, and article content
        article_content = (
            f"# {title}\n\n"
            f"**By {author} - Published {formatted_date}**\n\n"
            f"{intro_paragraph}\n\n"
        )

        # Add the main image with its caption
        st.image(og_image_url, caption=og_caption, use_column_width=True)

        # Add the sentence after the image
        article_content += f"\n\n{og_caption}"

        # Get all the paragraphs and combine them into a single string
        paragraphs = article_element.find_all("p")
        all_paragraphs = [p.get_text() for p in paragraphs]
        full_article_content = "\n\n".join(all_paragraphs)

        # Add link to the full article in bold
        article_content += f"\n\n**Please read the full article [here]({url})**"

        # Display the full article content
        st.markdown(article_content)
    else:
        st.write("Failed to retrieve the article content. Please try again later.")

def display_space_links():
    space_emojis = ["🚀", "🌌", "🪐"]
    space_links = [
        ("NASA Exoplanet News", "https://exoplanets.nasa.gov/news/"),
        ("Astronomy News", "https://www.astronomy.com/tags/news/"),
        ("Science News - Astronomy", "https://www.sciencenews.org/topic/astronomy")
    ]

    st.write("\n\n")
    st.markdown("---")
    st.markdown("**Explore more exoplanet news:**")
    for i, (label, url) in enumerate(space_links):
        st.markdown(f"<span style='font-size: 2em'>{space_emojis[i]}</span> [{label}]({url})", unsafe_allow_html=True)


def article():
    st.title("Exoplanet latest news")

    # Default website URL
    url = "https://www.space.com/news"

    # Get the latest article links
    latest_article_links = get_latest_articles(url)

    # Add a button to trigger the articles retrieval
    if st.button("Get Latest Articles"):
        if latest_article_links:
            for link in latest_article_links:
                get_article_content(link, num_paragraphs=4)  # Display the first four paragraphs
        else:
            st.write("Failed to retrieve the latest article links. Please try again later.")

    # Display space-themed links at the bottom of the page
    display_space_links()


def main():
    # Add background image
    st.set_page_config(layout="wide")

    st.title("Cosmic Voyager")
    st.subheader("Unveilling the cosmic mysteries")
    # Create the database and table if they don't exist
    conn = create_connection()
    create_table(conn)

    # Initialize the user's authentication status
    session_state = get_session_state()

    # Check if the user is authenticated before displaying the Home Screen
    if session_state.get('user_authenticated', False):
        home_page(session_state)
    else:
        page = st.sidebar.selectbox("Select a page:", ["Sign In", "Sign Up", "Latest Article"])
        if page == "Sign In":
            login_page(session_state)
        elif page == "Sign Up":
            sign_up_page(session_state)
        elif page == "Latest Article":
            article()
# def set_page_zoom(zoom):
#     st.markdown(f"""
#         <style>
#             @import url('https://fonts.googleapis.com/css?family=Montserrat');
#             body {{
#                 zoom: {zoom};
#             }}
#         </style>
#     """, unsafe_allow_html=True)
#
# set_page_zoom(0.75)

if __name__ == "__main__":
    main()

