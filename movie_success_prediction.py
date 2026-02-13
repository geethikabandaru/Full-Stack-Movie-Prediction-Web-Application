import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
@st.cache_data
def load_and_preprocess_data():
    file_path = 'IMDB-Movie-Data.csv'
    df = pd.read_csv(file_path)
    df_cleaned = df.dropna()
    
    # Encode categorical variables
    genre_encoder = LabelEncoder()
    director_encoder = LabelEncoder()
    
    df_cleaned['Genre_Encoded'] = genre_encoder.fit_transform(df_cleaned['Genre'])
    df_cleaned['Director_Encoded'] = director_encoder.fit_transform(df_cleaned['Director'])
    
    return df, df_cleaned, genre_encoder, director_encoder

df, df_cleaned, genre_encoder, director_encoder = load_and_preprocess_data()

# Scale numerical features and train models
@st.cache_resource
def train_models(df_cleaned):
    scaler = StandardScaler()
    numerical_features = ['Runtime (Minutes)', 'Rating', 'Votes', 'Revenue (Millions)', 'Metascore']
    X = df_cleaned[['Genre_Encoded', 'Director_Encoded'] + numerical_features]
    y = df_cleaned['Success']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train_scaled = X_train.copy()
    X_train_scaled[numerical_features] = scaler.fit_transform(X_train[numerical_features])
    
    X_test_scaled = X_test.copy()
    X_test_scaled[numerical_features] = scaler.transform(X_test[numerical_features])
    
    nb_model = GaussianNB()
    nb_model.fit(X_train_scaled, y_train)
    
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train_scaled, y_train)
    
    svm_model = SVC(probability=True)
    svm_model.fit(X_train_scaled, y_train)
    
    return nb_model, lr_model, svm_model, scaler, (X_test_scaled, y_test), numerical_features

nb_model, lr_model, svm_model, scaler, test_set, numerical_features = train_models(df_cleaned)

# Streamlit app configuration
st.set_page_config(page_title='Movie Success Rate Prediction', layout='wide')

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Prediction Tool", "Analysis Report"])

def prediction_page():
    # Custom CSS for styling
    st.markdown("""
        <style>
        .title {
            font-size: 36px;
            color: #008080;
            text-align: center;
            padding-bottom: 20px;
            text-shadow: 2px 2px #888888;
        }
        .welcome-message {
            font-size: 18px;
            color: #333;
            text-align: center;
            padding: 20px;
        }
        </style>
        """, unsafe_allow_html=True)

    st.markdown('<h1 class="title">Movie Success Rate Prediction</h1>', unsafe_allow_html=True)
    st.markdown('<p class="welcome-message">Enter details of your movie to predict its success rate.</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        movie_name = st.text_input('Movie Name')
        genre = st.selectbox('Genre', df['Genre'].unique())
        director = st.selectbox('Director', df['Director'].unique())
    with col2:
        runtime = st.number_input('Runtime (Minutes)', min_value=0, max_value=int(df['Runtime (Minutes)'].max()))
        rating = st.number_input('Rating', min_value=0.0, max_value=10.0)
        votes = st.number_input('Votes', min_value=0, max_value=int(df['Votes'].max()))
        revenue = st.number_input('Revenue (Millions)', min_value=0.0, max_value=float(df['Revenue (Millions)'].max()))
        metascore = st.number_input('Metascore', min_value=0, max_value=100)

    if st.button('Predict Success Rate'):
        with st.spinner('Predicting...'):
            input_data = pd.DataFrame({
                'Genre_Encoded': [genre_encoder.transform([genre])[0]],
                'Director_Encoded': [director_encoder.transform([director])[0]],
                'Runtime (Minutes)': [runtime],
                'Rating': [rating],
                'Votes': [votes],
                'Revenue (Millions)': [revenue],
                'Metascore': [metascore]
            })

            input_data[numerical_features] = scaler.transform(input_data[numerical_features])

            nb_pred = nb_model.predict_proba(input_data)[0][1]
            lr_pred = lr_model.predict_proba(input_data)[0][1]
            svm_pred = svm_model.predict_proba(input_data)[0][1]

            # Store in session state for recommendations
            st.session_state['last_movie'] = {
                'name': movie_name,
                'genre': genre,
                'director': director,
                'rating': rating,
                'revenue': revenue
            }

            st.subheader(f"Prediction Results for {movie_name}")
            c1, c2, c3 = st.columns(3)
            c1.metric("Naive Bayes", f"{nb_pred * 100:.2f}%")
            c2.metric("Logistic Regression", f"{lr_pred * 100:.2f}%")
            c3.metric("SVM", f"{svm_pred * 100:.2f}%")

def report_page():
    st.markdown('<h1 style="color: #008080; text-align: center;">Dataset Analysis Report</h1>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Data Snapshot", "Genre Analysis", "Financial Insights", "Model Performance", "Movie Recommendations"])
    
    with tab1:
        st.subheader("Dataset Overview")
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m1.metric("Total Movies", len(df))
        col_m2.metric("Successful Movies", len(df[df['Success'] == 1]))
        col_m3.metric("Avg Rating", round(df['Rating'].mean(), 2))
        
        st.write(f"Columns: {', '.join(df.columns)}")
        st.dataframe(df.head(10))
        
        st.subheader("Statistical Summary")
        st.write(df.describe())

    with tab2:
        st.subheader("Individual Genre Popularity")
        # Split and explode genres
        all_genres = df['Genre'].str.split(',').explode().str.strip()
        genre_dist = all_genres.value_counts()
        fig_genre = px.pie(values=genre_dist.values, names=genre_dist.index, title="Distribution of Individual Genres")
        st.plotly_chart(fig_genre, use_container_width=True)
        
        st.subheader("Top 10 Genre Combinations")
        genre_counts = df['Genre'].value_counts().head(10)
        fig = px.bar(genre_counts, x=genre_counts.index, y=genre_counts.values, 
                     labels={'x': 'Genre Combination', 'y': 'Number of Movies'},
                     color=genre_counts.values, color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Revenue vs Rating Distribution")
        fig3 = px.scatter(df_cleaned, x='Rating', y='Revenue (Millions)', 
                         color='Success', hover_data=['Title'],
                         trendline="ols", title="Impact of Rating on Revenue")
        st.plotly_chart(fig3, use_container_width=True)
        
        st.subheader("Correlation Heatmap")
        corr = df_cleaned[numerical_features + ['Success']].corr()
        fig4, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        st.pyplot(fig4)

    with tab4:
        st.subheader("Model Validation Metrics")
        X_test, y_test = test_set
        
        models = {'Naive Bayes': nb_model, 'Logistic Regression': lr_model, 'SVM': svm_model}
        res_cols = st.columns(3)
        for i, (name, m) in enumerate(models.items()):
            y_pred = m.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            res_cols[i].metric(f"{name} Accuracy", f"{acc*100:.2f}%")
        
        st.subheader("Feature Importance (Logistic Regression)")
        importance = lr_model.coef_[0]
        feat_importance = pd.Series(importance, index=X_test.columns).sort_values(ascending=False)
        fig5 = px.bar(feat_importance, x=feat_importance.values, y=feat_importance.index, orientation='h',
                     labels={'x': 'Coefficient Value', 'y': 'Feature'}, title="What Drives Movie Success?")
        st.plotly_chart(fig5, use_container_width=True)

    with tab5:
        st.subheader("Find Similar Movies")
        if 'last_movie' in st.session_state:
            last = st.session_state['last_movie']
            st.info(f"Recommendations based on your last prediction: **{last['name']}** ({last['genre']})")
            
            # Simple content-based filtering logic
            # Filter by same genre (contains) or director
            recommendations = df[
                (df['Genre'].str.contains(last['genre'].split(',')[0])) | 
                (df['Director'] == last['director'])
            ].sort_values(by='Rating', ascending=False).head(6)
            
            # Exclude the movie itself if it exists in DB
            recommendations = recommendations[recommendations['Title'] != last['name']].head(5)
            
            if not recommendations.empty:
                for _, row in recommendations.iterrows():
                    with st.expander(f"{row['Title']} ({row['Year']})"):
                        st.write(f"**Genre:** {row['Genre']}")
                        st.write(f"**Director:** {row['Director']}")
                        st.write(f"**Rating:** {row['Rating']} ⭐")
                        st.write(f"**Revenue:** ${row['Revenue (Millions)']}M")
                        st.write(f"**Description:** {row['Description']}")
            else:
                st.write("No direct matches found. Try searching below.")
        else:
            st.write("Perform a prediction first to get personalized recommendations, or search below.")
        
        st.divider()
        search_genre = st.selectbox("Or discover by Genre", df['Genre'].str.split(',').explode().str.strip().unique())
        search_results = df[df['Genre'].str.contains(search_genre)].sort_values(by='Rating', ascending=False).head(10)
        st.table(search_results[['Title', 'Director', 'Rating', 'Revenue (Millions)']])

if page == "Prediction Tool":
    prediction_page()
else:
    report_page()

st.markdown("---")
st.markdown("<p style='text-align: center;'>Movie Success Rate Prediction Project | Analytics & Reports</p>", unsafe_allow_html=True)