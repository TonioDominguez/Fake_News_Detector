# Basic Libraries
import streamlit as st
import time
import pickle
import numpy as np
import pandas as pd
import plotly.express as px

#Vectorizer librarie

from sklearn.feature_extraction.text import CountVectorizer

# Preprocessing for modeling

import re
import nltk
nltk.download('stopwords') 
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.porter import PorterStemmer

import gensim 
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim import corpora

# Importing Naive Bayes Model for ML

with open("model_NB.pkl", "rb") as file:
    model_NB = pickle.load(file)
    
with open("vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)
    
# For Fake News Detector functionality

stop_words = list(stopwords.words("spanish"))

stop_words.extend(["según", "ahora", "después", "todas", "toda", "todo", "todos", "sólo", "solo", "sido", "están", "estan", "hacer", "hecho", "puede", "tras", "cabe", "bajo", "durante", "mediante", "cada", "me", "lunes", "martes", "miércoles", "jueves", "viernes", "sabado", "domingo"])



# Page configuration
st.set_page_config(page_title="Fake News Detector", page_icon=":newspaper:", layout="wide")

# Background color

page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
background-color: #f7f6de;
}

[data-testid="stSidebarContent"] {
background-image: linear-gradient(to right, #fcfc1e, #f7f6de);
}

[data-testid="stHeader"] {
background-color: rgba(0, 0, 0, 0);
}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

# Home Page Top Sidebar
top_sidebar_placeholder = st.sidebar.empty()

top_sidebar_placeholder.markdown(
    f"""
    <div style="text-align:center">
        <h1 style="color: #00008B;">FAKE NEWS DETECTOR <br> (Spanish Edition)</h1>
        <img src="https://i.imgur.com/norkb4Y.png" alt="FN Miniature" style="width: 100%;">
    </div>
    """,
    unsafe_allow_html=True)

#Sidebar
st.sidebar.title("")
page = st.sidebar.radio("Navigation Menu", ["What is FAKE NEWS DETECTOR about?", "Use FAKE NEWS DETECTOR!", "Understanding FND", "About the Project"])

with st.spinner("This will take a second..."):
    time.sleep(1)
    
    if page == "What is FAKE NEWS DETECTOR about?":
        st.markdown(
        """
        <div style="text-align:center">
            <img src="https://i.imgur.com/6ZS6kSq.png" alt="Home banner" style="width: 75%;">
            <br>
            <br>
            <h1 style="font-size: 55px; color: #00008B; font-family: 'Lato', sans-serif;">FAKE NEWS PREDICTOR PROJECT</h1>
            <h3 style="font-size: 30px; color: #010168; font-family: 'Lato', sans-serif;">Make sure that what you are being told is truth</h3>
            <br>    
        </div>
        <div style="text-align:center">
        <div style="text-align: justify; color: black; max-width: 1000px; margin: 0 auto;">
        <p>
            In today's world, we're bombarded with information constantly. It's everywhere — social media, digital newspapers, tv news, instant messaging  —  . But here's the thing: <strong>not all of it is reliable</strong>. Sometimes what seems legit might be totally bogus. That's why it's so important to double-check stuff. Don't just swallow everything you see or hear. 
        </p>
        <p>
            <strong>Take a step back, ask questions, and look for different angles</strong>. 
        </p>    
        <p>    
            It's kinda like sifting through a pile of junk to find that one golden nugget. Sure, it takes a bit of effort, but it's worth it. Because when you're armed with the truth, you're not just informed, you're empowered. And that's what we need to navigate this crazy world - a healthy dose of skepticism and a keen eye for the real deal.
        </p>
        <p>
            <strong>So, let's embrace our inner skeptics</strong>. Let's challenge the status quo and demand accountability from those who shape the narratives we consume. In a world where information is power, it's up to us to wield that power responsibly. And that starts with being discerning consumers of information.
        </p>
        <br>
        </div>
        </div>
        <div style="text-align:center">
            <h3 style="font-size: 30px; color: #010168; font-family: 'Lato', sans-serif;">But what happens when we're drowning in information overload?</h3>
        <br>
        </div>
        <div style="text-align:center">
        <div style="text-align: justify; color: black; max-width: 1000px; margin: 0 auto;">
        <p>
            This is where technology can be our greatest ally. With the right tools and strategies, <strong>technology can help us sift through the noise</strong> and find the signal—the truth—in the midst of the chaos. From fact-checking websites and browser extensions that detect misinformation to algorithms that curate personalized news feeds based on our interests and credibility ratings, technology offers a multitude of resources to help us discern fact from fiction.
        </p>
        <br>
        </div>
        </div>
        <div style="text-align:center">
            <h3 style="font-size: 30px; color: #010168; font-family: 'Lato', sans-serif;">For all these reasons, I present my humble project</h3>
        <br>
        </div>
        <div style="text-align:center">
        <div style="text-align: justify; color: black; max-width: 1000px; margin: 0 auto;">
        <p>
            Here you have <strong>FAKE NEWS DETECTOR!</strong> A small personal project that can help you as a tool to separete fact from fiction. 
        </p>
         <p>
            Using a machine learning <strong>supervised classification algorithm</strong> (in this case <a href="https://www.turing.com/kb/an-introduction-to-naive-bayes-algorithm-for-beginners">Naive Bayes</a>, chosen from among several models for their accuracy and reliability). This detector analyzes and evaluates texts (article titles and news extracts), identifying patterns and features that reveal their authenticity. Based on a <strong>dataset with 2000 entries</strong> (labeled as fake or real news) our algorithm has been trained to detect signals of misinformation, bias and manipulation.
        </p>
        <p>
            My goal is to provide you with an additional layer of security and confidence in the task of <strong>evaluate if an information is thruth or fake</strong>. Whether you're scrolling through social media, reading an article or browsing a website, this <strong>FAKE NEWS DETECTOR</strong> can offer you a reliable guide you to discern what they are telling you.
        </p>
        <br>
        </div>
        </div>
        """,
        unsafe_allow_html=True)
    
    elif page == "Use FAKE NEWS DETECTOR!":
        st.markdown(
        """
        <div style="text-align:center">
            <img src="https://i.imgur.com/VPKY5jK.png" alt="FND banner" style="width: 75%;">
            <br>
            <br>
            <h1 style="font-size: 55px; color: #00008B; font-family: 'Lato', sans-serif;">FAKE NEWS DETECTOR App</h1>
            <h3 style="font-size: 30px; color: #010168; font-family: 'Lato', sans-serif;">Check below the veracity of your text</h3>
            <h3 style="font-size: 22px; color: black;"><u>Keep in mind a few points for a better use</u></h3>
            <br>
            <p style="font-size: 16px; color: black;">
                1. The text must be in Spanish
                <br>
                2. It is recommended that the length of the text does not exceed 30 words.
                <br>
                3. ML model has been trained on 2019 data. Its accuracy may decrease depending on the current terms.
            </p>
        </div>
        """,
        unsafe_allow_html=True)
        
        # Text cleaner function
        
        def text_cleaner(text):
                words = []
                new_text = []

                for word in gensim.utils.simple_preprocess(text):
                    if word not in gensim.parsing.preprocessing.STOPWORDS and word not in stop_words and len(word) > 4:
                        words.append(word)
                new_text = " ".join(words)

                return new_text
        
        # FND in action!
        
        col1, col2, col3 = st.columns([1,3,1])
        
        with col1:
            pass
        
        with col2:
        
            text = st.text_area("", height=100, placeholder="Insert text extract here!", max_chars=500)
        
            if st.button("Analyze"):
                new_text = text_cleaner(text)
                df = pd.DataFrame({"text":[new_text]})
                text_ml = df["text"]
                text_vect = vectorizer.transform(text_ml)
                text_prediction = model_NB.predict(text_vect)
                text_probability = model_NB.predict_proba(text_vect).tolist()
                resul = np.argmax(text_probability)
                resul = round(text_probability[0][resul]*100,2)
            
                if text_prediction == 0:
                    fake_result = f"""<h3>This text seems to be <span style='color:red'>FAKE</span> with an {resul}% accuracy</h3>"""
                    st.markdown(fake_result, unsafe_allow_html=True)
                elif text_prediction == 1:
                    real_result = f"""<h3>This text seems to be <span style='color:green'>REAL</span> with an {resul}% accuracy</h3>"""
                    st.markdown(real_result, unsafe_allow_html=True)
        with col3:
            pass
    
    elif page == "Understanding FND":
        tab1, tab2, tab3, tab4 = st.tabs(["Dataset Preprocessing", "EDA", "Sentiment Analysis", "ML Model"])
        
        with tab1:
            st.markdown(
                """
                <div style="text-align:center">
                    <img src="https://i.imgur.com/CHXC1dS.png" alt="Dataset banner" style="width: 75%;">
                    <br>
                    <br>
                    <h1 style="font-size: 55px; color: #00008B; font-family: 'Lato', sans-serif;">UNDERSTANDING OUR STARTING POINT</h1>
                    <h3 style="font-size: 30px; color: #010168; font-family: 'Lato', sans-serif;">Dataset preprocessing for training my ML model</h3>
                    <br>    
                </div>
                <div style="text-align:center">
                <div style="text-align: justify; color: black; max-width: 1000px; margin: 0 auto;">
                <p>
                    For this project I have used a group of <strong>Spanish data</strong> published in <a href="https://www.kaggle.com/datasets/arseniitretiakov/noticias-falsas-en-espaol">Kaggle</a>. It consists of three csv: one with 1000 fake news, another with 1000 real news and the third with 1000 fake and real news. I used the first two datasets to build the main dataset for training and testing my classification model, and the remaining one was used to assess its accuracy. All of them consist of text excerpts (headlines, tweets, and other social media messages).
                </p>
                <p>
                This project is based on interpreting human language to distinguish whether it belongs to a group (Fake News) or not. That's why the data processing I did focused on transforming all these initial texts into vectors that the machine could understand and process. This process is called <strong>NLP (Natural Language Processing)</strong>.
                </p>
                <p>
                So, for this particular task, I used a specific NLP library called <strong>NLTK (Natural Language Toolkit)</strong> to tokenize the texts and extracting keywords (smaller elements that could find a stronger relationship among themselves) that my ML model could recognize for categorization.
                </p>
                <p>
                Also I decided to use <strong>Sentiment Analysis</strong> to check out a hunch I was getting: fake news seems to be written with a lot of anger and negativity. So, I used another NLP tool called <strong>Sentiment Analysis Spanish</strong> (it's trained on 800,000 reviews from recommendation websites) to rate all the entries in my dataset on a positivity scale.
                </p>
                <br>
                </div>
                </div>
                """,
                unsafe_allow_html=True)
            
            st.markdown(
                """
                <div style="text-align:center">
                    <br>
                    <h3 style="font-size: 30px; color: #010168; font-family: 'Lato', sans-serif;">Final Dataset</h3>
                    <br>    
                </div>
                <div style="text-align:center">
                <div style="text-align: justify; color: black; max-width: 1000px; margin: 0 auto;">
                <p>
                    After all data cleaning and wrangling, I ended up with a single dataset containing columns useful for both thorough data exploration and training the fake news predictor model.     
                </p>
                <br>
                </div>
                </div>
                """,
                unsafe_allow_html=True)

            

            cola1, cola2, cola3 = st.columns([1,3,1])
            
            with cola1:
                pass
            
            with cola2:
                df = pd.read_csv("to_modeling.csv", index_col=0)
                st.dataframe(df, width=1000, hide_index=False)
                st.markdown(
                """
                <div style="text-align:justify">
                <p style="font-size: 16px; ; color: black;">
                    <strong>Explanation of variables:</strong><br>
                </p> 
                <p style="font-size: 14px; ; color: black;">
                    &#x25AA; <u><i>id</i></u>: Unique ID of each entry.<br>
                    &#x25AA; <u><i>text</i></u>: Original excerpts from headlines, tweets, and entries from other social media platforms.<br>
                    &#x25AA; <u><i>state</i></u>: Numeric classifier indicating the status of the information (0 for Fake and 1 for Real).<br>
                    &#x25AA; <u><i>type_news</i></u>: Equivalent string to the <i>state</i>, for better visual description of the type.<br>
                    &#x25AA; <u><i>tags</i></u>: List of keyword strings after tokenizing <i>text</i>.<br>
                    &#x25AA; <u><i>n_of_tags</i></u>: Tag count per entry to assess the length of each text after tokenization.<br>
                    &#x25AA; <u><i>text_ml</i></u>: Union of tags composing the text that will be analyzed by the ML model.<br>
                    &#x25AA; <u><i>numerical_sentiment</i></u>: Numerical value (from 0 to 1) that indicates the level of text positivity.<br>
                    &#x25AA; <u><i>sentiment</i></u>: Emotion label assigned to the news according to the value range on <i>numerical_sentiment</i>.<br>
                </p>
                </div>
                """,
                unsafe_allow_html=True)    
                                    
            with cola3:
                pass
            
        with tab2:
            st.markdown(
                """
                <div style="text-align:center">
                    <img src="https://i.imgur.com/wi4MAre.png" alt="EDA banner" style="width: 75%;">
                    <br>
                    <br>
                    <h1 style="font-size: 55px; color: #00008B; font-family: 'Lato', sans-serif;">INTERPRETING THE DATA IN DEPTH</h1>
                    <h3 style="font-size: 30px; color: #010168; font-family: 'Lato', sans-serif;">Number of Tags Distribution</h3>
                    <br>    
                </div>
                <div style="text-align:center">
                <div style="text-align: justify; color: black; max-width: 1000px; margin: 0 auto;">
                <p>
                    After breaking down the raw texts into tokens, I got a bunch of keywords per entry. These tags are the crucial elements the model reads for its classification system. That's why I checked out the distribution of this variable.
                </p>
                </div>
                </div>
                """,
                unsafe_allow_html=True)
            
            colb1, colb2, colb3 = st.columns([1,3,1])
            
            with colb1:
                pass
            
            with colb2:
                st.components.v1.html(open("histogram.html", "r").read(), height=450, scrolling=False)
                st.markdown(
                """
                <div style="text-align:center">
                <div style="text-align: justify; color: black; max-width: 800px; margin: 0 auto;">
                <p style="font-size: 14px;">
                    After cleaning out the most pronounced outliers, we noticed that the distribution looks kinda like a normal one, but it's leaning a bit to the left. Most of the tags per entry fall between <strong>16</strong> and <strong>21</strong>, with the highest value hitting <strong>19</strong>, clocking in at <strong>344</strong> entries.
                </p>
                </div>
                </div>
                """,
                unsafe_allow_html=True)
                
            with colb3:
                pass
            
            st.markdown(
                """
                <div style="text-align:center">
                    <br>
                    <h3 style="font-size: 30px; color: #010168; font-family: 'Lato', sans-serif;">Frequency Tags</h3>
                    <br>    
                </div>
                <div style="text-align:center">
                <div style="text-align: justify; color: black; max-width: 1000px; margin: 0 auto;">
                <p>
                    With the complete list of keywords obtained from the original texts, I set out to observe the frequency of repeated terms to find some sort of pattern that would help me understand which are the best trigger words to guide the classification.
                </p>
                <p>
                    Analyzing the frequency of repeated terms provided us with a clearer insight into the words that appear most frequently in our data. Identifying these keywords allowed us to better understand the <strong>most relevant themes and concepts</strong> for our classification. Some words appeared with surprisingly high frequency, suggesting that they are particularly important in defining the content of the texts.
                </p>
                </div>
                </div>
                """,
                unsafe_allow_html=True)
            
            colc1, colc2, colc3, colc4 = st.columns([1,2,2,1])
            
            with colc1:
                pass
            
            with colc2:
                st.components.v1.html(open("fake_tag_frequency.html", "r").read(), height=450, scrolling=False)
                st.markdown(
                    """
                    <div style="text-align:center">
                    <img src="https://i.imgur.com/yN79sDx.png" alt="Fake WC" style="width: 65%;">
                    </div>
                    """,
                    unsafe_allow_html=True)

            with colc3:
                st.components.v1.html(open("real_tag_frequency.html", "r").read(), height=450, scrolling=False)
                st.markdown(
                    """
                    <div style="text-align:center">
                    <img src="https://i.imgur.com/7USerf9.png" alt="Fake WC" style="width: 65%;">
                    </div>
                    """,
                    unsafe_allow_html=True)
            with colc4:
                pass
            
            st.markdown(
                """
                <div style="text-align:center">
                <div style="text-align: justify; color: black; max-width: 850px; margin: 0 auto;">
                <br>
                <br>
                <p style="font-size: 14px;">
                    It's interesting to see the results shown by these plots because we can observe an ideological trend in the frequency. Although some words like <strong>"España"</strong>, <strong>gobierno"</strong> or <strong>"Madrid"</strong> appear equally repeated among the most frequent in the texts, we find distiguished keywords in the fake entries: <strong>"Sánchez"</strong>, <strong>"inmigrantes"</strong>, <strong>"Podemos"</strong>. These words matches with topics about which the alt-right in Spain has relentlessly spread misinformation. 
                </p>
                <p style="font-size: 14px;">
                    And this pattern can be corroborated by taking a look at the respective word clouds. In real news, informative and descriptive words stand out, indicating that they are used more accurately to report facts. However, fake news highlights other words like <strong>"mulsulmanes"</strong>, <strong>"feministas"</strong>, and <strong>"Islam"</strong> topics that can be added to the already mentioned terms preferred by the alt-right.
                </p>
                </div>
                </div>
                """,
                unsafe_allow_html=True)
            
        with tab3:
            st.markdown(
                """
                <div style="text-align:center">
                    <img src="https://i.imgur.com/YWT2es7.png" alt="SA banner" style="width: 75%;">
                    <br>
                    <br>
                    <h1 style="font-size: 55px; color: #00008B; font-family: 'Lato', sans-serif;">ANALYZING THE SENTIMENTALITY</h1>
                    <h3 style="font-size: 30px; color: #010168; font-family: 'Lato', sans-serif;">Type of feelings per text</h3>
                    <br>    
                </div>
                <div style="text-align:center">
                <div style="text-align: justify; color: black; max-width: 1000px; margin: 0 auto;">
                <p>
                    After observing the pattern I discovered while exploring which were the most repeated keywords in both fake and real news, I began to consider how fake news is disseminated. Typically, misinformation is spread with the intention of discrediting something or someone as a form of propaganda. According to the <strong>Goebbels principle</strong>: "Propaganda must be limited to a few basic ideas and must be repeated incessantly. It must present the <strong>same ideas from different perspectives over and over again</strong>."
                </p>
                <p>
                    But what about the emotion behind that intention? It occurred to me to analyze the feelings behind fake news, aware that they are written from hatred towards what is different and fear of losing privileges.
                </p>
                <p>
                    So, I took advantage of the tokenized texts through <strong>NLTK (NLP)</strong>, which I was going to use to train the model, to take another step in data exploration and measure the level of emotion through <strong>Sentiment Analysis (NLP)</strong>. Thanks to the sentiment-spanish library, I was able to assign a numerical value to each text ranging from 0 to 1 on a positivity scale. Then, I grouped the resulting values into another variable, allowing me to label the emotion as <strong>very negative</strong>, <strong>negative</strong>, <strong>neutral</strong>, <strong>positive</strong>, and <strong>very positive</strong>.
                </p>
                </div>
                </div>
                """,
                unsafe_allow_html=True)
            
            cold1, cold2, cold3 = st.columns([1,3,1])
        
            with cold1:
                pass
        
            with cold2:
                st.components.v1.html(open("text_sentiment_analysis.html", "r").read(), height=450, scrolling=False)
                st.markdown(
                """
                <div style="text-align:center">
                <div style="text-align: justify; color: black; max-width: 800px; margin: 0 auto;">
                <p style="font-size: 14px;">
                    The analysis first tells us something that is quite obvious: <strong>the majority of news, whether fake or true, are very negative</strong>. They make up 46.7% of the total texts in our dataset.
                </p>
                <p style="font-size: 14px;">
                    The second major piece of evidence supports the hypothesis about the <strong>predominance of extreme negativity in fake news</strong>. Within the very negative category, there are 530 fake and 404 true news articles, with a difference of 126 points in favor of fake news. This significant difference is unique among the sentiment groups, which distribute their proportions more evenly. Additionally, we notice that true information has a higher count in the other sentiment categories.
                </p>
                <p style="font-size: 14px;">
                    Based on the analysis, we can say that <strong>fake news focuses its discourse on extreme negativity</strong>, while real news, in attempting to inform more objectively, distribute their tone across a wider spectrum.
                </p>
                </div>
                </div>
                """,
                unsafe_allow_html=True)
            
            with cold3:
                pass
        
        with tab4:
            st.markdown(
                """
                <div style="text-align:center">
                    <img src="https://i.imgur.com/UBpqW3V.png" alt="Dataset banner" style="width: 75%;">
                    <br>
                    <br>
                    <h1 style="font-size: 55px; color: #00008B; font-family: 'Lato', sans-serif;">CLASSIFYING THE TRUTHFULNESS</h1>
                    <h3 style="font-size: 30px; color: #010168; font-family: 'Lato', sans-serif;">Choosing the right model</h3>
                    <br>    
                </div>
                <div style="text-align:center">
                <div style="text-align: justify; color: black; max-width: 1000px; margin: 0 auto;">
                <p>
                    Once I processed the data to train and test my model, it was time to choose the most suitable one.
                </p>
                <p>
                First, I allocated 75% of our dataset for training and 25% for testing, a corpus that I vectorized using <strong>CountVectorizer</strong>, an implementation from <strong>Scikit Learn</strong> that takes all text elements and converts them into a token count matrix. Then, I prepared the vectorized data to test the ML models.
                </p>
                <p>
                I tried five types of algorithms to see which one gives us the best accuracy results:
                </p>
                <p>
                &#x25AA; <i>LOGISTIC REGRESSION</i><br>
                &#x25AA; <i>RANDOM FOREST</i><br>
                &#x25AA; <i>XGBOOST</i><br>
                &#x25AA; <i>DECISION TREES</i><br>
                &#x25AA; <i>NAIVE BAYES</i><br>
                </p>
                <p>
                The one that gave me the best results was <strong>Naive Bayes</strong>. This is a supervised learning algorithm based on Bayes' theorem and <strong>primarily used for text classification</strong>. It's a <strong>probabilistic classifier</strong>, meaning it predicts based on the probability of an object.
                </p>
                <br>
                </div>
                </div>
                """,
                unsafe_allow_html=True)
            
            cole1, cole2, cole3 = st.columns([1,3,1])
        
            with cole1:
                pass
            
            with cole2:
                st.markdown(
                    """
                    <div style="text-align:center">
                    <img src="https://i.imgur.com/yByEBRt.png" alt="NB Report" style="width: 65%;">
                    </div>
                    """,
                    unsafe_allow_html=True)
            
            with cole3:
                pass
            
            st.markdown(
                """
                <div style="text-align:center">
                    <br>
                    <h3 style="font-size: 30px; color: #010168; font-family: 'Lato', sans-serif;">Naive Bayes Performance</h3>
                    <br>    
                </div>
                <div style="text-align:center">
                <div style="text-align: justify; color: black; max-width: 1000px; margin: 0 auto;">
                <p>
                    Let's see how the model performed during testing. Below, I'll measure its accuracy on the sample by extracting its <strong>confusion matrix</strong> and <strong>ROC curve</strong>, essential metrics for evaluating its precision.
                </p>
                </div>
                </div>
                """,
                unsafe_allow_html=True)
            
            colf1, colf2, colf3, colf4 = st.columns([1,2,2,1])
            
            with colf1:
                pass
            
            with colf2:
                st.markdown(
                    """
                    <div style="text-align:center">
                    <img src="https://i.imgur.com/wg4BMbr.png" alt="Conf Matrix" style="width: 65%;">
                    </div>
                    """,
                    unsafe_allow_html=True)
            
            with colf3:
                st.markdown(
                    """
                    <div style="text-align:center">
                    <img src="https://i.imgur.com/VmIYHbc.png" alt="ROC Curve" style="width: 65%;">
                    </div>
                    """,
                    unsafe_allow_html=True)
            
            with colf4:
                pass
            
            st.markdown(
                """
                <div style="text-align:center">
                <div style="text-align: justify; color: black; max-width: 850px; margin: 0 auto;">
                <br>
                <br>
                <p style="font-size: 14px;">
                    The results obtained are quite good. In the <strong>Confusion Matrix</strong> (on the left), the heat map indicates that we achieve good numbers on the sample with True Positives and True Negatives. And the <strong>ROC Curve</strong> (on the right) also shows us that the curve's elbow is above the random line between 0 and 1, and the <strong>Area Under Curve (AUC)</strong> is wide enough to give us good results. 
                </p>
                </div>
                </div>
                """,
                unsafe_allow_html=True)
            
            st.markdown(
                """
                <div style="text-align:center">
                    <br>
                    <h3 style="font-size: 30px; color: #010168; font-family: 'Lato', sans-serif;">External Data Testing</h3>
                    <br>    
                </div>
                <div style="text-align:center">
                <div style="text-align: justify; color: black; max-width: 1000px; margin: 0 auto;">
                <p>
                    So, apparently, this model is working correctly. The only thing left is to test it with information external to the one I used to train it. For this purpose, I'll use the third dataset that I downloaded from the collection I found on Kaggle. It's a dataset with 2000 entries labeled as true or false.
                </p>
                <p>
                    And here there are the results:
                </p>
                </div>
                </div>
                """,
                unsafe_allow_html=True)
            
            colg1, colg2, colg3 = st.columns([1,3,1])
            
            with colg1:
                pass
            
            with colg2:
                df2 = pd.read_csv("test_to_streamlit.csv", index_col=0)
                st.dataframe(df2, width=1000, hide_index=False)
                st.markdown(
                """
                <div style="text-align:justify">
                <p style="font-size: 16px; ; color: black;">
                    <strong>Explanation of variables:</strong><br>
                </p> 
                <p style="font-size: 14px; ; color: black;">
                    &#x25AA; <u><i>class</i></u>: Check box marked means it is a Real News; Check box unmarked means it is a Fake News.<br>
                    &#x25AA; <u><i>text</i></u>: Text in its totallity.<br>
                    &#x25AA; <u><i>type_news</i></u>: Model prediction.<br>
                </p>
                </div>
                """,
                unsafe_allow_html=True)
                
            with colg3:
                pass
            
            st.markdown(
                """
                <div style="text-align:center">
                <div style="text-align: justify; color: black; max-width: 1000px; margin: 0 auto;">
                <p>
                    Finally, the final results in predictions with external information are quite good. In total, the <strong>accuracy for Fake News is 93.8% with a failure rate of 6.2%</strong>; and for <strong>Real News, the accuracy is 92.9% with a failure rate of 7.1%</strong>.
                </p>
                </div>
                </div>
                """,
                unsafe_allow_html=True)
            
    elif page == "About the Project":
        st.markdown(
        """
        <div style="text-align:center">
            <img src="https://i.imgur.com/X4bZYwR.png" alt="About banner" style="width: 75%;">
            <br>
            <br>
            <h1 style="font-size: 55px; color: #00008B; font-family: 'Lato', sans-serif;">LETS TALK ABOUT FEELINGS</h1>
            <h3 style="font-size: 30px; color: #010168; font-family: 'Lato', sans-serif;">Hope you found this project interesting!</h3>
            <br>    
        </div>
        <div style="text-align:center">
        <div style="text-align: justify; color: black; max-width: 1000px; margin: 0 auto;">
        <p>
            Hello, I'm Toño Domínguez, and this is my final project for the <strong>Data Analyst Bootcamp</strong> at <strong>Ironhack</strong>, which has been my life for the past two months. With this work, I aim to summarize a large part of the knowledge acquired during this time and, in a way, put a final signature on my introduction to the world of data science. 
        </p>
        <p>
            The theme of this project is not random. I am a journalist by profession — yes, I know, it's rare... a journalist becoming a data analyst? — and the field of information, especially politics, is one of my great passions. That's why developing a final project that <strong>combines the world of information with the world of data</strong> made a lot of sense to me. 
        </p>    
        <p>    
            My Fake News predictor is not perfect — of course not — I need to update it to make it more effective with more recent news — it struggles a bit as it is trained with information from five years ago — and find a way to overcome the geographical barrier.
        </p>
        <p>
            And yet... it works! I'm very proud of it because developing a tool that <strong>can help us discern what is real and what is fictional is</strong>, I believe, one of the greatest needs we have today.
        </p>
        <p>
            And I'm also proud of myself, hell yes!
        </p>
        <p>
            I hope you find it interesting and that it inspires you, as I have found many other projects that have inspired me to develop things that until recently I wouldn't have believed I was capable of.
        </p>
        <p>
           Best regards and <strong>VIVA LA DATA</strong>! 
        </p>
        <br>
        </div>
        </div>
        <div style="text-align:center">
            <h3 style="font-size: 30px; color: #010168; font-family: 'Lato', sans-serif;">Sources consulted</h3>
        <br>
        </div>
        <div style="text-align:center">
        <div style="text-align: justify; color: black; max-width: 1000px; margin: 0 auto;">
        <p>
            A few sites I have turned to for enlightenment in the creation process:
        </p>
        <p>
        &#x25AA; <u><a href="https://www.kaggle.com/datasets/arseniitretiakov/noticias-falsas-en-espaol">Kaggle</a></u>: Where I found the datasets that started it all.<br>
        &#x25AA; <u><a href="https://github.com/restevesd/Clasificador/blob/main/Art%C3%ADculo_Julio.ipynb">Roberto Esteves Github</a></u>: Inquiry about a project that also developed a fake news classifier.<br>
        &#x25AA; <u><a href="https://github.com/sentiment-analysis-spanish/sentiment-spanish">sentiment-analysis-spanish</a></u>: Library for performing sentiment analysis NLP.<br>
        &#x25AA; <u><a href="https://github.com/TonioDominguez?tab=repositories">My Github repositories</a></u>: Where I reviewed old labs to refresh my memory on working with supervised classification models.<br>
        &#x25AA; <u><a href="https://docs.streamlit.io">Streamlit Docs</a></u>: Documentation checked to know how to tame this beast.<br>
        &#x25AA; <u><a href="https://chat.openai.com/">ChatGPT</a></u>: The snitch that kept whispering to me what I was doing wrong when the code errors were driving me crazy.<br>
        </p>
        <br>
        </div>
        </div>
        <div style="text-align:center">
            <h3 style="font-size: 30px; color: #010168; font-family: 'Lato', sans-serif;">Can't forget the acknowledgments</h3>
        <br>
        </div>
        <div style="text-align:center">
        <div style="text-align: justify; color: black; max-width: 1000px; margin: 0 auto;">
         <p>
            Acknowledgments to several people for being there, not only during the final project but throughout this journey in the Bootcamp.
        </p>
        <p>
            To my classmates <strong>Carlos</strong>, <strong>Óscar</strong>, <strong>Axier</strong> and <strong>Víctor</strong>. Thank you for motivating me with your work throughout the entire time and for being such great colleagues.
        </p>
        <p>
            To <strong>Xisca</strong> for that masterclass on Big Data and the help with the SQL block. It was a pleasure to meet such a whirlwind of a person!
        </p>
        <p>
            To <strong>Isi</strong>. I don't have enough words to thank you for all your work and dedication with us. I can't imagine having had a better mentor than you.
        </p>
        <p>
           To <strong>Gali</strong>, the guy who one day suggested to me that maybe Ironhack could be a good idea. And who, a month before it started, was more excited than I was. 
        </p>
        <p>
           To <strong>Cristina</strong>, for being by my side all this time and helping me navigate my frustrations until they became achievements. 
        </p>
        <br>
        </div>
        </div>
        """,
        unsafe_allow_html=True)  