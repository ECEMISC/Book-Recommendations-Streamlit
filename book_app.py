import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langdetect import detect, LangDetectException
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim import corpora
from gensim.models.ldamodel import LdaModel
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langdetect import detect, LangDetectException
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim import corpora
from gensim.models.ldamodel import LdaModel
import pyLDAvis
import streamlit.components.v1 as components
import warnings
import pyLDAvis.gensim as gensimvis

warnings.filterwarnings('ignore')

nltk.download('stopwords')
nltk.download('wordnet')
import streamlit as st
import base64



# Ana sayfa arka plan rengini turuncu yapmak için CSS
st.markdown(
    """
    <style>
    .stApp {
        background-color: #FFA500; /* Turuncu renk */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar'a görsel ekle
st.sidebar.image("amazon_logo.JPG", use_column_width=True)

import streamlit as st

# CSS ile sidebar rengini değiştirme
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background-color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# Page selection

st.markdown("""
    <style>
    [data-testid="stSidebar"] h1 {
        background: linear-gradient(to right, black, orange);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

st.sidebar.title("Recommender Options")

page = st.sidebar.radio("Choose a method:", ["Content-Based Recommender", "Personalised Recommender System", "Topic Modeling Recommender", "Topic Modeling Visualizer"])

# Load and prepare data
@st.cache_data
def load_data():
    df = pd.read_csv("sampled_cleandata.csv")  # Artık bu dosya zaten küçük
    return df

df = load_data()

# Görselleştirme ayarları
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
df.head()

# Page 1: Content-Based Recommender
if page == "Content-Based Recommender":
    import pandas as pd
    import streamlit as st
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    st.title("📘 Content-Based Book Recommender")

    # Prepare unique books and lowercase key text fields
    df_unique = df[['Title', 'description', 'categories']].drop_duplicates(subset='Title').reset_index(drop=True)
    df_unique['Title'] = df_unique['Title'].str.lower()
    df_unique['description'] = df_unique['description'].fillna('').str.lower()
    df_unique['categories'] = df_unique['categories'].astype(str).str.lower()

    # Combine description and category
    df_unique['combined'] = df_unique['description'] + " " + df_unique['categories']

    # TF-IDF vectorization
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df_unique['combined'])

    # Cosine similarity matrix
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Recommendation function
    def content_based_recommender(title, cosine_sim, dataframe):
        indices = pd.Series(dataframe.index, index=dataframe['Title'])
        indices = indices[~indices.index.duplicated(keep='last')]
        if title not in indices:
            return None, f"'{title}' not found. Please check the spelling."
        book_index = indices[title]
        similarity_scores = pd.DataFrame(cosine_sim[book_index], columns=["score"])
        book_indices = similarity_scores.sort_values("score", ascending=False)[1:11].index
        results = dataframe.iloc[book_indices][['Title', 'description']].copy()
        results['Similarity Score'] = similarity_scores.iloc[book_indices].values
        results = results[['Title', 'Similarity Score', 'description']]
        results.rename(columns={"Title": "Title", "description": "Description"}, inplace=True)
        return results.reset_index(drop=True), None

    # User input
    title_input = st.text_input("Enter a book title to get similar recommendations:")

    if title_input:
        recommendations, error = content_based_recommender(title_input.lower(), cosine_sim, df_unique)

        if error:
            st.warning(error)
        else:
            st.success(f"Top recommendations similar to '{title_input.title()}':")

            # Show each result in a clean card-like layout
            for i, row in recommendations.iterrows():
                st.markdown(f"### 📖 {row['Title'].title()}")
                st.markdown(f"**Similarity Score:** {round(row['Similarity Score'], 3)}")

                # Limit description to first 100 words
                full_desc = row["Description"]
                short_desc = " ".join(full_desc.split()[:50])

                # If longer than 100 words, show expandable
                if len(full_desc.split()) > 100:
                    st.markdown(short_desc + "...")
                    with st.expander("Show full description"):
                        st.write(full_desc)
                else:
                    st.markdown(full_desc)

                st.markdown("---")  # line separator between results

# Page 2: Personalised Recommender System

elif page == "Personalised Recommender System":
    st.title("📚 Personalised Book Recommender")

    df['Title'] = df['Title'].str.lower().apply(lambda x: re.sub(r'[^\w\s]', '', x)).str.strip()

    def is_english(text):
        try:
            return detect(text) == 'en'
        except LangDetectException:
            return False

    df_filtered = df[df['Title'].apply(is_english)]
    df_unique = df_filtered[['Title', 'description']].drop_duplicates(subset='Title').reset_index(drop=True)

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df_unique['description'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    def smart_recommendation_for_user(user_id, df, df_unique, cosine_sim, top_n=2):
        user_books = df[df['User_id'] == user_id]
        read_titles = user_books['Title'].str.lower().str.strip().unique()
        liked_books = user_books[user_books['review/score'] >= 4.0]['Title'].str.lower().str.strip().unique()
        title_to_index = pd.Series(df_unique.index, index=df_unique['Title']).drop_duplicates()
        index_to_title = pd.Series(df_unique['Title'], index=df_unique.index)
        recommendations = []

        if len(liked_books) > 0:
            for book in liked_books:
                if book in title_to_index:
                    idx = title_to_index[book]
                    sim_scores = list(enumerate(cosine_sim[idx]))
                    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
                    already_read = set(read_titles)
                    filtered_scores = [(i, score) for i, score in sim_scores if index_to_title[i] not in already_read and index_to_title[i] != book]
                    for i, score in filtered_scores[:top_n]:
                        recommendations.append({
                            'User': user_id,
                            'Based On': book,
                            'Recommended Book': index_to_title[i],
                            'Similarity Score': round(score, 3),
                            'Reason': 'Similar content (according to book liked)'
                        })
        else:
            for book in read_titles:
                if book in title_to_index:
                    idx = title_to_index[book]
                    sim_scores = list(enumerate(cosine_sim[idx]))
                    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
                    already_read = set(read_titles)
                    filtered_scores = [(i, score) for i, score in sim_scores if index_to_title[i] not in already_read and index_to_title[i] != book]
                    scored_books = []
                    for i, score in filtered_scores:
                        title = index_to_title[i]
                        avg_rating = df[df['Title'].str.lower().str.strip() == title]['review/score'].mean()
                        scored_books.append((title, round(avg_rating, 2), score))
                    top_recommendations = sorted(scored_books, key=lambda x: x[1], reverse=True)[:top_n]
                    for title, avg_score, sim in top_recommendations:
                        recommendations.append({
                            'User': user_id,
                            'Based On': book,
                            'Recommended Book': title,
                            'Similarity Score': round(sim, 3),
                            'Reason': f'Similar content (highest rated by community)'
                        })

        return pd.DataFrame(recommendations)

    user_id_input = st.text_input("Enter your User ID:")
    if user_id_input:
        recs = smart_recommendation_for_user(user_id_input, df_filtered, df_unique, cosine_sim)
        if not recs.empty:
            st.success("Here are your personalised recommendations:")

            # 💄 Better styled table (bold headers, no index)
            def styled_recommendation_table(df):
                styled_df = df.style.set_table_styles(
                    [{
                        'selector': 'th',
                        'props': [('font-weight', 'bold'), ('text-align', 'left'), ('font-size', '15px')]
                    }]
                ).hide(axis="index")
                return styled_df


            # Render table as clean HTML (bold headers + no index)
            def render_html_table(df):
                st.markdown("""
                    <style>
                    table {
                        width: 100%;
                        border-collapse: collapse;
                    }
                    th {
                        background-color: #f2f2f2;
                        padding: 10px;
                        text-align: left;
                        font-weight: bold;
                        font-size: 16px;
                    }
                    td {
                        padding: 10px;
                        text-align: left;
                        font-size: 15px;
                    }
                    tr:nth-child(even) {
                        background-color: #fafafa;
                    }
                    </style>
                """, unsafe_allow_html=True)

                html_table = df.to_html(index=False, escape=False)
                st.markdown(html_table, unsafe_allow_html=True)


            # Ve sonra çağır:
            render_html_table(recs)

# Page 3: Topic Modeling Recommender

elif page == "Topic Modeling Recommender":
    st.title("🧠 Topic Modeling-Based Book Recommender")

    from langdetect import detect, LangDetectException
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    def is_english(text):
        try:
            return detect(text) == 'en'
        except LangDetectException:
            return False

    def preprocess(text):
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        words = text.split()
        words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and len(word) > 2]
        return words

    # 1. Topic modeling için kullanıcı yorumlarını al
    df_comments = df[df['review/text'].apply(is_english)].copy()
    df_comments['tokens'] = df_comments['review/text'].fillna('').apply(preprocess)

    texts = df_comments['tokens'].tolist()
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=10, random_state=42, passes=10, alpha='auto')

    def extract_keywords(topic_str):
        return ", ".join([word.split("*")[-1].replace('"', "").strip() for word in topic_str.split("+")])

    topic_descriptions = {idx: extract_keywords(topic) for idx, topic in lda_model.print_topics(num_words=8)}

    def get_dominant_topic(ldamodel, corpus):
        dominant_topics = []
        for bow in corpus:
            topic_probs = ldamodel.get_document_topics(bow)
            dominant_topic = max(topic_probs, key=lambda x: x[1])[0]
            dominant_topics.append(dominant_topic)
        return dominant_topics

    df_comments['dominant_topic'] = get_dominant_topic(lda_model, corpus)

    # Her kullanıcı için dominant topic hesapla
    user_topics = df_comments.groupby('User_id')['dominant_topic'].value_counts().unstack(fill_value=0)

    # 2. Kitap açıklamalarını işle
    df_books = df[df['description'].apply(is_english)].copy()
    df_books['tokens'] = df_books['description'].fillna('').apply(preprocess)

    book_texts = df_books['tokens'].tolist()
    book_dict = corpora.Dictionary(book_texts)
    book_corpus = [dictionary.doc2bow(text) for text in book_texts]  # Aynı dictionary kullanılmalı

    df_books['book_topic'] = get_dominant_topic(lda_model, book_corpus)
    books_with_topic = df_books[['Title', 'description', 'book_topic']].drop_duplicates(subset='Title')


    # 3. Kullanıcının topic'ine göre kitap öner
    def recommend_books_by_topic(user_id, user_topics, books_with_topic, top_n=5):
        if user_id not in user_topics.index:
            return pd.DataFrame([{"Message": f"No topic data found for user {user_id}."}]), None
        dominant_topic = user_topics.loc[user_id].idxmax()
        read_books = df[df['User_id'] == user_id]['Title'].unique()
        unread_books = books_with_topic[~books_with_topic['Title'].isin(read_books)]
        recommendations = unread_books[unread_books['book_topic'] == dominant_topic].head(top_n).reset_index(drop=True)
        return recommendations, dominant_topic


    # 4. Arayüz: kullanıcı ID girişi
    user_id_topic = st.text_input("Enter your User ID to get topic-based recommendations:")
    if user_id_topic:
        topic_recs, top_topic = recommend_books_by_topic(user_id_topic, user_topics, books_with_topic)
        if top_topic is not None:
            st.markdown(f"**User's most engaged topic:** Topic #{top_topic}")
            if top_topic in topic_descriptions:
                st.markdown(f"**Topic keywords:** {topic_descriptions[top_topic]}")


        def render_html_table(df):
            st.markdown("""
                <style>
                table {
                    width: 100%;
                    border-collapse: collapse;
                }
                th {
                    background-color: #f2f2f2;
                    padding: 10px;
                    text-align: left;
                    font-weight: bold;
                    font-size: 16px;
                }
                td {
                    padding: 10px;
                    text-align: left;
                    font-size: 15px;
                }
                tr:nth-child(even) {
                    background-color: #fafafa;
                }
                </style>
            """, unsafe_allow_html=True)
            html_table = df.to_html(index=False, escape=False)
            st.markdown(html_table, unsafe_allow_html=True)


        # 5. Öneri gösterimi (description gösteriliyor, ama model topic'i yorumlardan çıkardı)
        if not topic_recs.empty and "Message" not in topic_recs.columns:
            for _, row in topic_recs.iterrows():
                st.markdown(f"### 📖 {row['Title'].title()}")

                # 'description' sadece görsel olarak gösteriliyor
                full_text = row['description']
                short_text = " ".join(full_text.split()[:50])
                st.markdown(short_text + "...")

                with st.expander("Show full description"):
                    st.write(full_text)

                st.markdown("---")
        else:
            render_html_table(topic_recs)

        # 5. Öneri gösterimi: description'la birlikte
        if not topic_recs.empty and "Message" not in topic_recs.columns:
            for _, row in topic_recs.iterrows():
                st.markdown(f"### 📖 {row['Title'].title()}")
                full_text = row['description']
                short_text = " ".join(full_text.split()[:50])
                st.markdown(short_text + "...")
                with st.expander("Show full description"):
                    st.write(full_text)
                st.markdown("---")
        else:
            render_html_table(topic_recs)


# Page 4: Topic Modeling Visualizer

if page == "Topic Modeling Visualizer":
    st.title("Interactive Topic Visualization")

    from langdetect import detect, LangDetectException
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    # ✅ 1. Fonksiyonları önce tanımla
    def is_english(text):
        try:
            return detect(text) == 'en'
        except LangDetectException:
            return False

    def preprocess(text):
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        words = text.split()
        words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and len(word) > 2]
        return words

    # ✅ 2. Sonra veriyi filtrele
    df_vis = df[df['review/text'].apply(is_english)].copy()
    df_vis['tokens'] = df_vis['review/text'].fillna('').apply(preprocess)

    texts = df_vis['tokens'].tolist()
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    # ✅ 3. LDA modeli eğit
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=10, random_state=42, passes=10, alpha='auto')

    # ✅ 4. pyLDAvis görselleştirme
    with st.spinner('Preparing interactive topic visualization...'):
        vis_data = gensimvis.prepare(lda_model, corpus, dictionary, n_jobs=1)
        html_string = pyLDAvis.prepared_data_to_html(vis_data)

    st.markdown("<div style='margin-top: 40px;'></div>", unsafe_allow_html=True)
    components.html(html_string, width=1300, height=800, scrolling=True)
    st.markdown("<div style='margin-bottom: 40px;'></div>", unsafe_allow_html=True)

    st.markdown("<p style='font-size: 18px; margin-top: 30px;'>🔍 Click on any topic bubble to explore its most representative keywords interactively.</p>", unsafe_allow_html=True)
