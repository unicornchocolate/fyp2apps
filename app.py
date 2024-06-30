import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer

# Load saved data and models
with open('train_data.pkl', 'rb') as train_file:
    train_data = pickle.load(train_file)

with open('test_data.pkl', 'rb') as test_file:
    test_data = pickle.load(test_file)

with open('knn_recomm.pkl', 'rb') as recomm_file:
    knn_recomm = pickle.load(recomm_file)

with open('restaurant_features_train.pkl', 'rb') as train_file:
    restaurant_features_train = pickle.load(train_file)

with open('restaurant_features_test.pkl', 'rb') as test_file:
    restaurant_features_test = pickle.load(test_file)

with open('us_city_user_rating_train.pkl', 'rb') as train_file:
    us_city_user_rating_train = pickle.load(train_file)

with open('us_city_user_rating_test.pkl', 'rb') as test_file:
    us_city_user_rating_test = pickle.load(test_file)


# Collaborative-based recommendations function (adjusted)
def collaborative_based_recommendations(user_id_local, restaurant_features_train_local,
                                        restaurant_features_test_local, us_city_user_rating_train_local,
                                        us_city_user_rating_test_local, data_source='train'):
    # Choose the data source
    if data_source == 'train':
        user_data = restaurant_features_train_local
    elif data_source == 'test':
        user_data = restaurant_features_test_local
    else:
        return pd.DataFrame()

    # Check if the user exists in the data source
    if user_id_local not in user_data.index:
        return pd.DataFrame()

    # Find the index of the user
    user_index = user_data.index.get_loc(user_id_local)

    # Get the user vector
    user_vector = user_data.iloc[user_index].values.reshape(1, -1)

    # Find recommendations for the user using the KNN model trained on the training data
    distances, indices = knn_recomm.kneighbors(user_vector, n_neighbors=11)

    # Filter out the user's own vector from the recommendations
    recommended_indices = indices.flatten()[1:]

    # Get the recommended restaurant names
    recommended_restaurant_names = restaurant_features_train.columns[recommended_indices]

    # Create a DataFrame for the recommendations
    recommended_restaurants = pd.DataFrame({
        'Restaurant Name': recommended_restaurant_names
    })

    # Merge with the original dataset to get the restaurant IDs from both train and test datasets
    all_restaurants = pd.concat([us_city_user_rating_train_local[['Restaurant Name', 'Cuisines', 'Rating']],
                                 us_city_user_rating_test_local[['Restaurant Name', 'Cuisines', 'Rating']]]).drop_duplicates()

    recommended_restaurants = recommended_restaurants.merge(
        all_restaurants,
        on='Restaurant Name', how='left'
    )

    # Return recommendations DataFrame
    return recommended_restaurants


# Content-based recommendations function
def content_based_recommendations(attribute_local, value_local):
    if attribute_local == 'Cuisines':
        # Filter training data for the specified Cuisines
        train_data_sample = train_data[train_data['Cuisines'].str.contains(value_local, case=False, na=False)].copy()
    elif attribute_local == 'City':
        # Filter training data for the specified City
        train_data_sample = train_data[train_data['City'] == value_local].copy()
    elif attribute_local == 'Country':
        # Filter training data for the specified Country
        train_data_sample = train_data[train_data['Country'] == value_local].copy()
    elif attribute_local == 'Diets':
        # Filter training data for the specified Diets
        train_data_sample = train_data[train_data['Diets'].str.contains(value_local, case=False, na=False)].copy()
    else:
        st.error(f"Invalid attribute specified: {attribute_local}")
        return pd.DataFrame()

    if train_data_sample.empty:
        st.error(f"No restaurants found with {attribute_local} '{value_local}'")
        return pd.DataFrame()

    # Reset index for cosine similarity calculation
    train_data_sample.reset_index(drop=True, inplace=True)

    # Feature Extraction
    train_data_sample['Split'] = train_data_sample['Cuisines'].apply(lambda x: ' '.join(x.split(',')))

    # TF-IDF Vectorizer
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix_train = tfidf.fit_transform(train_data_sample['Split'])

    # Cosine Similarity
    cosine_sim = linear_kernel(tfidf_matrix_train, tfidf_matrix_train)

    # Get the index of the restaurant with the specified attribute
    if attribute_local == 'Cuisines' or attribute_local == 'Diets':
        idx = train_data_sample[train_data_sample[attribute_local].str.contains(value_local, case=False, na=False)].index[0]
    else:
        idx = 0  # for City and Country, use the first index (arbitrary choice)

    # Calculate similarity scores with other restaurants
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the restaurants based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Top 10 similar restaurants

    # Get indices of the top similar restaurants
    rest_indices = [i[0] for i in sim_scores]

    # Retrieve information for the recommended restaurants
    recommended_restaurants = train_data_sample.iloc[rest_indices][['Restaurant Name', 'Cuisines', 'Rating']].copy()

    return recommended_restaurants


# Hybrid recommendations function
def hybrid_recommendations(user_id_local, attribute_local, value_local):
    # Collaborative Filtering Recommendations
    collab_recommendations = collaborative_based_recommendations(user_id_local, restaurant_features_train,
                                                                 restaurant_features_test,
                                                                 us_city_user_rating_train,
                                                                 us_city_user_rating_test, data_source='test')

    # Content-Based Filtering Recommendations
    content_recommendations = content_based_recommendations(attribute_local, value_local)

    # Combine both recommendations
    combined_recommendations = pd.concat([collab_recommendations,
                                          content_recommendations]).drop_duplicates(subset='Restaurant Name')

    # Return only 'Restaurant Name' columns, limited to 10 recommendations
    combined_recommendations['Rank'] = range(1, len(combined_recommendations) + 1)
    return combined_recommendations[['Rank', 'Restaurant Name', 'Cuisines', 'Rating']].head(10)


# Streamlit app begins
def main():
    st.title('**where2dine**')
    st.markdown('***Discover Your Culinary Adventure with where2dine! Explore a world of flavors at your fingertips. '
               'Start your exploration today and let where2dine guide you to your next unforgettable dining experience.***')

    st.markdown(
        f"""
               <style>
               .stApp {{
                   background-image: url("https://www.moonpalacecancun.com/all_inclusive_dining_df484d3211.jpg");
                   background-size: cover;
               }}
               </style>
               """,
        unsafe_allow_html=True
    )

    # Country selection
    country_list = ['United Kingdom', 'France', 'Japan', 'Brazil', 'South Korea']
    selected_country = st.selectbox('**Select Country**', country_list)

    # City selection (as a select box)
    city_list = train_data['City'].unique().tolist()
    selected_city = st.selectbox('**Select City**', city_list)

    # Cuisines selection
    cuisines_list = ['Italian', 'Asian', 'Pizza', 'Seafood']
    selected_cuisine = st.selectbox('**Select Cuisine**', cuisines_list)

    # Diet selection (as a select box)
    diet_list = train_data['Diets'].dropna().unique().tolist()
    selected_diet = st.selectbox('**Select Diet**', diet_list)


    if st.button('**Get Recommendations**'):
        # Predefined user ID (replace with dynamic user input if needed)
        user_id = 348

        # Determine which attribute was selected and call hybrid_recommendations accordingly
        if selected_country:
            hybrid_recommendations_result = hybrid_recommendations(user_id, 'Country', selected_country)
            st.subheader(f"Recommended Restaurants Special For You :")
            if not hybrid_recommendations_result.empty:
                for index, row in hybrid_recommendations_result.iterrows():
                    st.markdown(f"**{row['Rank']}. {row['Restaurant Name']}**")
                    st.markdown(f"*Cuisines: {row['Cuisines']}*")
                    try:
                        rating = int(row['Rating'])
                        st.markdown(f"Rating: {'⭐' * rating}")
                    except (ValueError, TypeError):
                        st.markdown("Rating: N/A")
                    st.markdown("---")
            else:
                st.warning(f"No recommendations found for 'Country' '{selected_country}'")

        elif selected_city:
            hybrid_recommendations_result = hybrid_recommendations(user_id, 'City', selected_city)
            st.subheader(f"Recommended Restaurants Special For You :")
            if not hybrid_recommendations_result.empty:
                for index, row in hybrid_recommendations_result.iterrows():
                    st.markdown(f"**{row['Rank']}. {row['Restaurant Name']}**")
                    st.markdown(f"*Cuisines: {row['Cuisines']}*")
                    try:
                        rating = int(row['Rating'])
                        st.markdown(f"Rating: {'⭐' * rating}")
                    except (ValueError, TypeError):
                        st.markdown("Rating: N/A")
                    st.markdown("---")
            else:
                st.warning(f"No recommendations found for 'City' '{selected_city}'")

        elif selected_cuisine:
            hybrid_recommendations_result = hybrid_recommendations(user_id, 'Cuisines', selected_cuisine)
            st.subheader(f"Recommended Restaurants Special For You :")
            if not hybrid_recommendations_result.empty:
                for index, row in hybrid_recommendations_result.iterrows():
                    st.markdown(f"**{row['Rank']}. {row['Restaurant Name']}**")
                    st.markdown(f"*Cuisines: {row['Cuisines']}*")
                    try:
                        rating = int(row['Rating'])
                        st.markdown(f"Rating: {'⭐' * rating}")
                    except (ValueError, TypeError):
                        st.markdown("Rating: N/A")
                    st.markdown("---")
            else:
                st.warning(f"No recommendations found for 'Cuisine' '{selected_cuisine}'")

        elif selected_diet:
            hybrid_recommendations_result = hybrid_recommendations(user_id, 'Diets', selected_diet)
            st.subheader(f"Recommended Restaurants Special For You :")
            if not hybrid_recommendations_result.empty:
                for index, row in hybrid_recommendations_result.iterrows():
                    st.markdown(f"**{row['Rank']}. {row['Restaurant Name']}**")
                    st.markdown(f"*Cuisines: {row['Cuisines']}*")
                    try:
                        rating = int(row['Rating'])
                        st.markdown(f"Rating: {'⭐' * rating}")
                    except (ValueError, TypeError):
                        st.markdown("Rating: N/A")
                    st.markdown("---")
            else:
                st.warning(f"No recommendations found for 'Diet' '{selected_diet}'")

        else:
            st.error("Please select an attribute and provide a corresponding value.")


if __name__ == '__main__':
    main()
