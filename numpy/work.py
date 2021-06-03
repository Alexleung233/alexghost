# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 23:55:51 2021

@author: Alex leung
"""

from gensim.corpora import Dictionary
from gensim.models import ldamodel
from gensim.matutils import kullback_leibler, jaccard, hellinger, sparse2full
import numpy
import jieba.posseg as jp, jieba


with open('红楼梦.txt',encoding='utf-8') as f:
    te=f.read()
    te1=te.replace('\n', '').replace('\u3000', '')
    texts=te1.split('------------')

   
with open('stop_words.txt') as f:
   st=f.read()
   stopwords=st.split('\n')
   



words_ls = []
for text in texts:
    words = [word.word for word in jp.cut(text) if word.word != ' ' and word.word not in stopwords]
    words_ls.append(words)
    

qian=words_ls[1:81]
hou=words_ls[81:121]


be=[]
for i in qian:
    for j in i:
        be.append(j)


la=[]
for i in hou:
    for j in i:
        la.append(j)


all=be+la


dictionary = Dictionary(words_ls)
print("字典：",dictionary)
corpus = [dictionary.doc2bow(text) for text in words_ls]
print("语料库：",corpus)
    
    
    
model = ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=2)

model.show_topics()
    


xe1 = model.id2word.doc2bow(be)
lda_xe1 = model[xe1]

xe2 = model.id2word.doc2bow(la)
lda_xe2 = model[xe2]

xe3 = model.id2word.doc2bow(all)
lda_xe3 = model[xe3]


print("lda_qianbashi:",lda_xe1)
print("lda_housishi:",lda_xe2)
print("lda_housishi:",lda_xe3)


from gensim.matutils import kullback_leibler, jaccard, hellinger, sparse2full
print("1:",jaccard(lda_xe1, lda_xe2))
print("2:",jaccard(lda_xe1, lda_xe3))
print("3:",jaccard(lda_xe2, lda_xe3))


print("1:",hellinger(lda_xe1, lda_xe2))
print("2:",hellinger(lda_xe1, lda_xe3))
print("3:",hellinger(lda_xe2, lda_xe3))
    

print("1:",kullback_leibler(lda_xe1, lda_xe2))
print("2:",kullback_leibler(lda_xe1, lda_xe3))
print("3:",kullback_leibler(lda_xe2, lda_xe3))
    
model.get_document_topics(xe3)
    
    
print("1:",jaccard(xe1, xe2))
print("2:",jaccard(xe1, xe3))
print("3:",jaccard(xe2, xe3))    
    
    
    
print("1:",jaccard(be, la))
print("2:",jaccard(be, all))
print("3:",jaccard(la, all))    
    
    
    
    
    
def make_topics_bow(topic):
    # takes the string returned by model.show_topics()
    # split on strings to get topics and the probabilities
    topic = topic.split('+')
    #print("topic:",topic)
    # list to store topic bows
    topic_bow = []
    for word in topic:
        # split probability and word
        prob, word = word.split('*')
        # get rid of spaces
        word = word.replace(" ","")
        #print("word:",word)
        # convert to word_type
        #print("2：",model.id2word.doc2bow(["bank"]))
        #print("example:",model.id2word.doc2bow([word.replace('"','')]))
        word = model.id2word.doc2bow([word.replace('"','')])[0][0]
        topic_bow.append((word, float(prob)))
    return topic_bow

topic_qian, topic_hou = model.show_topics()    
print("topic_qian[1]:",topic_qian[1])
print("topic_hou[1]:",topic_hou[1])
qian_distribution = make_topics_bow(topic_qian[1])
print("前八十权重:",qian_distribution)
hou_distribution = make_topics_bow(topic_hou[1])
print("后四十权重:",hou_distribution)
    

print("3:",hellinger(qian_distribution, hou_distribution))







    