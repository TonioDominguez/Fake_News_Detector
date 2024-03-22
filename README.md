<div style="text-align:center">
            <img src="https://i.imgur.com/6ZS6kSq.png" alt="Home banner" style="width: 75%;">
            <br>
            <h1 style="font-size: 55px; color: #00008B; font-family: 'Lato', sans-serif;">FAKE NEWS PREDICTOR PROJECT :newspaper::mag:</h1>
            <br>
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
        </div>
        </div>
        <br>
        <div style="text-align:center">
            <h3 style="font-size: 30px; color: #010168; font-family: 'Lato', sans-serif;">But what happens when we're drowning in information overload?</h3>
        </div>
        <div style="text-align:center">
        <div style="text-align: justify; color: black; max-width: 1000px; margin: 0 auto;">
                    <br>
        <p>
            This is where technology can be our greatest ally. With the right tools and strategies, <strong>technology can help us sift through the noise</strong> and find the signal—the truth—in the midst of the chaos. From fact-checking websites and browser extensions that detect misinformation to algorithms that curate personalized news feeds based on our interests and credibility ratings, technology offers a multitude of resources to help us discern fact from fiction.
        </p>
        </div>
        </div>
        <br>
        <div style="text-align:center">
            <h3 style="font-size: 30px; color: #010168; font-family: 'Lato', sans-serif;">For all these reasons, I present my humble project</h3>
        </div>
        <br>
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
        
---

<h1 style="font-size: 55px; color: #00008B; font-family: 'Lato', sans-serif;">PROJECT STRUCTURE :open_file_folder:</h1>
<br>

<p>
  This project is developed through a bunch of Python notebooks.
</p>

<ol>
  <li>
    <strong><a href="https://github.com/TonioDominguez/Fake_News_Detector/blob/main/fake_news_preprocessing.ipynb">FAKE NEWS PREPROCESSING & EDA:</a></strong> A first notebook where I clean and process the datasets that I'm going to use to build my prediction model. I also perform EDA (Exploratory Data Analysis) on crucial variables and Sentiment Analysis.
  </li>
  
  <li>
    <strong><a href="https://github.com/TonioDominguez/Fake_News_Detector/blob/main/fake_news_classifier.ipynb">ML DEVELOPING & TESTING:</a></strong> Here, I select, fine-tune, and measure the results of the ML model that best fits the project. I also perform tests of the complete model with a dataset external to the training data.
  </li>
  
  <li>
    <strong><a href="https://github.com/TonioDominguez/Fake_News_Detector/blob/main/predictor_fakenews_input.ipynb">FAKE NEWS DETECTOR PER SOLO IMPUTS:</a></strong> As a summary, I create this Notebook by importing the complete model and adding some functions that clean the user input to process their text and test it.
  </li>
</ol>

<p>
  As a final step, I develop a <strong><a href="https://github.com/TonioDominguez/Fake_News_Detector/blob/main/app.py">Streamlit app</a></strong> where I thoroughly develop the storytelling of the project creation and allow you to use the predictor.
</p>
<br>

---

<h1 style="font-size: 55px; color: #00008B; font-family: 'Lato', sans-serif;">PROJECT DEVELOPMENT TIME ⏰</h1>
<br>

<p>
    <strong>10 days</strong> from <strong>03/12/24</strong> to <strong>03/22/24</strong>
  </p>
  <br>

---

<h1 style="font-size: 55px; color: #00008B; font-family: 'Lato', sans-serif;">SOURCES :fountain:</h1>
<br>
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

---
        
<div style="text-align:center">
            <h1 style="font-size: 55px; color: #00008B; font-family: 'Lato', sans-serif;">LETS TALK ABOUT FEELINGS :star:</h1>
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
