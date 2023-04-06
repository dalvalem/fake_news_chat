import streamlit as st
import openai
import os
from newspaper import Article
from sklearn.feature_extraction.text import TfidfVectorizer

from joblib import load

openai.api_key = os.environ["OPENAI_KEY"]
session_length = 20


def initialize_streamlit():
    st.set_page_config(initial_sidebar_state="collapsed", layout="wide")
    st.title("Detector de stiri false")
    st.markdown(
        """
        Programul de detectie a stirilor false este in proba: puteti trimite evaluari si sugestii la intelinkgov@gmail.com\n
        ATENTIE: Acuratetea programului este de 92,4% deci se asteapta unele erori de fals REAL sau fals FAKE......
        Explicatie: eticheta fake este data de cate ori gaseste in articol elemente false sau stirea contine opinii ori altceva care nu este considerat stire\n
        Introdu linkul stirii pe care vrei sa o evaluezi:\n
        """
    )

    if 'conversation' not in st.session_state:
        st.session_state.conversation = [{
            "role": "system", "content": """
                You are an AI assistant that can skillfully detect if an article represents fake news or not. Present lengthy arguments.
                You will present your arguments starting with the conclusion: does the article represents FAKE NEWS or NOT!
                You are not to base your answers on the current date or time. Your LLM model will be used in the future, after your cutoff time. 
                You will present your answers in  the Romanian .
                You will be given a lengthy text/article as input and you will focus on the truth value of the text/article and nothing else.
                """
        }]


def download_news_article(article_url):
    article = Article(article_url)
    article.download()
    article.parse()
    article_text = article.text.encode('utf-8')
    return article_text


def translate_to_english(article):
    from googletrans import Translator
    translator = Translator()
    translated = translator.translate(article, src="auto", dest="en")
    return translated.text


def classify_story(article):
    tokenizer: TfidfVectorizer = load('models/tokenizer_model.job')
    predictor = load('models/predictor_model.job')

    translated = translate_to_english(article)
    vect_story = tokenizer.transform([translated])
    prediction = predictor.predict(vect_story)
    return prediction[0]


def main():
    initialize_streamlit()
    question = st.text_input("link")
    if question == "": return

    article_content = download_news_article(question)[:4000]
    predicted_label = classify_story(article_content)

    st.session_state.conversation.append({"role": "user", "content": f"{article_content}"})
    if len(st.session_state.conversation) >= session_length:
        system_user = [st.session_state.conversation[0]]
        trimmed_conv = st.session_state.conversation[-session_length:]
        st.session_state.conversation = system_user + trimmed_conv

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=st.session_state.conversation
    )
    answer = response['choices'][0]['message']['content']
    output = f"""    
    {answer}
    \n----------------------------------\n
    Classifier result: {predicted_label}
    """
    st.markdown(output)


main()
