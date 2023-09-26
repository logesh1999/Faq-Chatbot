import streamlit as st
import json
import pandas as pd
import numpy
import re
import gensim
import pprint
from gensim import corpora
from gensim.parsing.preprocessing import remove_stopwords
import sklearn
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
import gensim.downloader as api


def main():
    html_temp = """
    <div style="background-color:cyan;padding:10px">
    <h2 style="color:black;text-align:center;">FAQ</h2>
    </div>
    """
    # response = requests.get('https://www.google.co.in/')
    st.markdown(html_temp, unsafe_allow_html=True)

    df = pd.read_csv(r'faq.csv')
    df.columns = ['Questions', 'Answers']
    index_sim = None
    max_sim = None

    def clean_sentence(sentence, stopwords=False):

        sentence = sentence.lower().strip()
        sentence = re.sub(r'[^a-z0-9\s]', '', sentence)

        if stopwords:
            sentence = remove_stopwords(sentence)
        return sentence

    def get_cleaned_sentences(df, stopwords=False):

        sents = df[['Questions']]
        cleaned_sentences = []

        for index, row in df.iterrows():
            cleaned = clean_sentence(row['Questions'], stopwords)
            cleaned_sentences.append(cleaned)

        return cleaned_sentences

    cleaned_sentences = get_cleaned_sentences(df, stopwords=True)
    s = pd.DataFrame(cleaned_sentences)
    cleaned_sentences_with_stopwords = get_cleaned_sentences(df, stopwords=False)
    k = pd.DataFrame(cleaned_sentences_with_stopwords)
    sentences = cleaned_sentences_with_stopwords
    sentence_words = [[word for word in document.split()]
                      for document in sentences]
    dic = corpora.Dictionary(sentence_words)

    bow_corpus = [dic.doc2bow(text) for text in sentence_words]

    s = st.text_input('Enter your Query')
    print(s)
    max_sim_final = -1
    question_orig = s
    question = clean_sentence(question_orig, stopwords=False)
    question_embedding = dic.doc2bow(question.split())

    def retrieveAndPrintFAQAnswers(question_embedding, sentence_embedding, sentences, max_sim_final):

        max_sim = -1
        index_sim = -1
        topTenList = []
        topTenDict = {}
        topTenDictSorted = {}

        for index, faq_embedding in enumerate(sentence_embedding):

            sim = cosine_similarity(faq_embedding, question_embedding)[0][0]
            print(index, sim, sentences[index])
            # topTenDict[str(sim)].append(index)
            if sim > max_sim:
                max_sim = sim
                index_sim = index
                max_sim_final = sim
                topTenList.append(index)
                print(index_sim)
        # topTenDictSorted = sorted(topTenDict.items(), key =
        #      lambda kv:(kv[1], kv[0]))
        topTenList.sort(reverse=True)
        return topTenList[0:10]

    # v2w_model = gensim.models.KeyedVectors.load('w2vecmodel.mod')
    v2w = None
    try:
        v2w_model = gensim.models.KeyedVectors.load("./w2vecmodel.mod")
        print('Loaded w2v model')
    except:
        v2w_model = api.load('word2vec-google-news-300')
        v2w_model.save("./w2vecmodel.mod")
        print('Saved glove model')

    w2vec_embedding_size = len(v2w_model['computer'])

    def getWordVec(word, model):

        samp = model['computer']
        vec = [0] * len(samp)

        try:

            vec = model[word]

        except:

            vec = [0] * len(samp)

        return (vec)

    def getPhraseEmbedding(phrase, embeddingmodel):
        samp = getWordVec('computer', embeddingmodel)
        vec = numpy.array([0] * len(samp))
        den = 0
        for word in phrase.split():
            den = den + 1
            vec = vec + numpy.array(getWordVec(word, embeddingmodel))
        return vec.reshape(1, -1)

    sent_embedding = []

    for sent in cleaned_sentences:
        sent_embedding.append(getPhraseEmbedding(sent, v2w_model))
    question_embedding = getPhraseEmbedding(question, v2w_model)
    index_sim = retrieveAndPrintFAQAnswers(question_embedding, sent_embedding, cleaned_sentences, max_sim_final)

    result = ''
    if st.button('Search'):
        print('\n  search keyword is ' + s)
        index_sim = retrieveAndPrintFAQAnswers(question_embedding, sent_embedding, cleaned_sentences, max_sim_final)
        print('\n ', index_sim, "\n", max_sim_final)
        # print(df)
        print(index_sim)
        if s != "":
            # with open(r"FAQ final.CSV", encoding='utf-8') as f:
            # st.header('your searched queries', anchor= None)
            st.success('your searched queries')
            # st.error('No values found')
            # st.write('**----------------------------------------------------------------------**')
            for i in (index_sim):
                print(i, end=" ")
                st.write('------------------------------------------------------------------------------------------')
                st.write(df.iloc[i, 0])
                st.write(df.iloc[i, 1])

            if s not in df.iloc[index_sim, 0]:
                st.write(
                    "Go to google [link](https://www.google.co.in/search?q=" + s.replace(' ', '%20') + "&source=null)")

    with open(r"faq.csv", encoding='utf-8') as d:
        data = d
        st.write('--------------------------------------')
        st.header("Here we have repeated FAQ's", anchor=None)
        # st.write('**-----------------------------------------------------------------------------------------**')


if __name__ == '__main__':
    main()









