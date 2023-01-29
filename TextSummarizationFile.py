# Databricks notebook source
!pip install pandas
!pip install nltk
!pip install numpy
!pip install rouge

# COMMAND ----------

import pandas as pd
from pyspark.sql import SparkSession
import numpy
import requests
from collections import Counter
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from numpy.linalg import norm
from rouge import Rouge
import matplotlib.pyplot as plt

from random import normalvariate
import math


# COMMAND ----------

#Input total documents required
documents_total = 1000

# COMMAND ----------

class Preprocess:
    def __init__(self):
        pass
    
    #Method to read articles from the dataset
    def dataprocess(self, filepath):
        articles_dir = filepath+"/News+Articles"
        sections = ['business', 'entertainment', 'politics', 'sport', 'tech']
        summaries_dir = filepath+"/Summaries"
        
        #Declaring arrays to add the articles, summaries and document ids seperately
        articles = []
        summaries = []
        file_ids = []
        
        for section in sections:
            file = '001'
            article_filepath = articles_dir + "/" + section + '/' + file +'.txt'
            summary_filepath = summaries_dir + "/" + section + '/' + file +'.txt'

            response1 = requests.get(article_filepath)
            response2 = requests.get(summary_filepath)
            
            while response1.status_code == 200:
                art = response1.text
                summ = response2.text
                articles.append(".".join([line.rstrip() for line in art.splitlines()]))
                summaries.append('.'.join([line.rstrip() for line in summ.splitlines()]))
                file_ids.append(section + '/' + file)
                
                file_no = int(file)
                file_no +=1
                file = str(file_no)
                ##########
                if file == str((documents_total//5)+1):
                    break
                ###########
                n = 3-len(file)
                for _ in range(n):
                    file = '0'+file
                article_filepath = articles_dir + "/" + section + '/' + file +'.txt'
                summary_filepath = summaries_dir + "/" + section + '/' + file +'.txt'
                response1 = requests.get(article_filepath)
                response2 = requests.get(summary_filepath)
        
        #Creating a map of file ids and articles
        article = list(map(lambda x,y: (x,y),file_ids,articles))
        #Creating a map of file ids and summaries
        summary = list(map(lambda x,y: (x,y),file_ids, summaries))
        
        #Creating a datframe of the articles and summaries
        articles_df = spark.createDataFrame(article,["Category/ID", "Article"])
        summaries_df = spark.createDataFrame(summary,["Category/ID","Summary"])
    
        return articles_df, summaries_df
    
    #Method to preprocess the data
    def process(self, document, doc_id):
        sentences = document.split(".")
        doc = set()
        wordlist = []
        sentence_data = []
        for idx,sentence in enumerate(sentences):
            #creating sentence ids for every sentence in the document
            sentence_id = doc_id+'/'+ str(idx)
            
            #Performing Tokenization of words
            words = word_tokenize(sentence)
            if len(words)<1:
                continue
            
            #Filtering words by removing stopwords
            words = [word for word in words if word.lower() not in stopwords.words('english') and word.isalpha()]
            
            #Performing word lemmatizations
            lemmatizer = nltk.stem.WordNetLemmatizer()
            words = [lemmatizer.lemmatize(word) for word in words]
            
            #collecting all unique words of the document
            for word in words:
                doc.add(word)
            
            if words:
                #wordlist array has tuples of sentence ids and the preprocesed words in that sentence.
                wordlist.append((sentence_id,words))
                #wordlist array has tuples of sentence and that particular sentence id.
                sentence_data.append((sentence_id,sentence))
        
        return wordlist, doc, sentence_data


# COMMAND ----------

#Reading the dataset path
dataset_path = "https://bigdatadataset6350.s3.us-west-1.amazonaws.com/dataset"
data = Preprocess()
articles_df, summaries_df = data.dataprocess(dataset_path)
#Displaying dataframes for news articles
display(articles_df)
articles_df.count()

# COMMAND ----------

class Tfidf:
    def __init(self):
        pass
      
    #Method to calculate the term frequency
    def term_frequency(self,wordlist):
        tf = Counter(wordlist[1])
        return (wordlist[0],tf)
    
    #Calculating the inverse document frequency using the document frequency vector
    def inverse_docfreq(self,sent_ids,dfvector):
        return numpy.log10(numpy.reciprocal(dfvector) * sent_ids)
        
    #Creating term frequency matrix and document frequency matrix
    def termfreq_matrix(self,tf,doclist):
        return [tf.get(word,0) for word in doclist]

    def docfreq_matrix(self,tf,doclist):
        return [1.0 if tf.get(word,0.0)>0 else 0.0 for word in doclist]


# COMMAND ----------

class SVD:
    def __init__(self):
        pass
    
    def unit_vector(self,k):
        before_norm = [normalvariate(0, 1) for _ in range(k)]
        norms = 0
        for val in before_norm:
            norms += (val*val)
        y = math.sqrt(norms)
        Normalized = [x / y for x in before_norm]
        return Normalized

    def decomposition(self,m, mat):
        row, col = mat.shape
        k = min(row,col)
        prev_V = self.unit_vector(k)
        if row>col:
            B = numpy.dot(mat.T, mat)
        else:
            B = numpy.dot(mat, mat.T)
        curr_V = numpy.dot(B, prev_V)
        curr_V /= norm(curr_V)


        while abs(numpy.dot(curr_V, prev_V)) <= 1 - 1e-10:
            prev_V = curr_V
            curr_V = numpy.dot(B, prev_V)
            curr_V /= norm(curr_V)
        v = curr_V
        u1 = numpy.dot(m, v)
        sig = norm(u1)
        u = u1/ sig
        return v, u, sig
      
    #Method to calculate singular value decomposition using power method
    def SingularValueDecomp(self,matrix):
        row, col = matrix.shape
        svd = []
        end = min(row,col)
        idx = 0
        while idx< end :
            mat = matrix.copy()
            for value, u, v in svd[:idx]:
                mat -= value * numpy.outer(u, v)
            if row>col:
                V, U, ST = self.decomposition(matrix, mat)
            else:
                U, V, ST = self.decomposition(matrix.T, mat)
            svd.append((ST, U, V))
            idx +=1
        Singular_Value, U_singular, V_singular = [numpy.array(x) for x in zip(*svd)]
        return Singular_Value, U_singular.T, V_singular

# COMMAND ----------

term_doc = Tfidf()
svd_obj = SVD()

# COMMAND ----------

class LSA:
    def __init__(self):
        pass
    
    #Method to calculate Latent Semantic Analysis
    def lsa1(self,sentenceRDD, doclist):
        #Creating a term frequency rdd having key as sentence id and value as the term frequency map of that sentence
        tf = sentenceRDD.map(lambda x: term_doc.term_frequency(x))
        dfvector = tf.map(lambda x: term_doc.docfreq_matrix(x[1],doclist)).reduce(lambda x, y: numpy.array(x) + numpy.array(y))
        #rdd of term frequency matrix
        tf = tf.map(lambda x: (x[0], term_doc.termfreq_matrix(x[1], doclist))).sortByKey()
        sentence_ids = tf.keys().collect()
        term_frequency_mat = tf.values()
        term_frequency_mat = numpy.array(numpy.transpose(term_frequency_mat.collect()))
        #Creating inverse document matrix 
        idf_mat = term_doc.inverse_docfreq(len(sentence_ids),dfvector)
        idf_mat = numpy.array(numpy.transpose(idf_mat))
        idf_mat = numpy.reshape(idf_mat, (-1,1))
        #Creating tf-idf matrix
        tf_idf_mat = numpy.multiply(term_frequency_mat,idf_mat)
        #U, S, VT = numpy.linalg.svd(tf_idf_mat, full_matrices=0)
        
        #getting the singular matrices using the implemented singular value decomposition
        S,U,VT = svd_obj.SingularValueDecomp(tf_idf_mat)
        return U, S, VT

# COMMAND ----------

lsa_obj = LSA()

# COMMAND ----------

class Extract:
    def __init__(self):
        self.dimension = 3
        self.reduction = 1/1
    
    #Ranking each sentence in the document based on the singular matrices generated after doing SVD
    def rank_sentences(self, S, VT):

        dim = max(self.dimension, int(len(S)*self.reduction))
        sigma = tuple(s**2 if i < dim else 0.0 for i, s in enumerate(S))

        sentence_ranks = []
        
        for cols in VT.T:
            ranks = sum(s*v**2 for s, v in zip(sigma, cols))
            sentence_ranks.append(math.sqrt(ranks))

        return sentence_ranks
    
    def compute_sentence(self, sentencerdd, rank):
        n = sentencerdd.count()
        ranks = iter(rank)
        #sentence_ranked rdd that contains key as (sentence id, sentence) and value as the sentence rank in the document
        sentence_ranked = sentencerdd.map(lambda x: (x, next(ranks)))
        
        #Sorting the ranks in descending order and getting the top n/3 sentences where n is the total no. of sentences in the document
        s = sentence_ranked.sortBy(lambda x: -x[1]).take(math.ceil(n/3))
        summaries_ranked = sc.parallelize(s)
        #sorting the highly ranked sentences in based of the sentence id so that the summary is coherent
        summaries = summaries_ranked.map(lambda x: x[0]).sortByKey()
        
        #final summary as a string
        summary = ".".join(summaries.values().collect())
        #print(summary)
        return summary
        

# COMMAND ----------

extract = Extract()

# COMMAND ----------

final_summary = []
sc = spark.sparkContext

for rows in articles_df.collect():
    document = rows['Article']
    doc_id = rows['Category/ID']
    wordlist, doclist, sentence_Data = data.process(document, doc_id)
    #creating rdd having keys as sentence ids and values as sentence words
    sentence_rdd = sc.parallelize(wordlist)
    #creating rdd having keys as sentence ids and values as sentences
    sentence_data_rdd = sc.parallelize(sentence_Data)
    
    #Implementing LSA on the sentence rdd
    U, S, VT = lsa_obj.lsa1(sentence_rdd, doclist)
    #extracting summaries using the Extract class
    ranks = extract.rank_sentences(S, VT)
    summaries = extract.compute_sentence(sentence_data_rdd, ranks)
    final_summary.append((doc_id, summaries))

summary = sc.parallelize(final_summary)
#Dataframe having document ID and system generated final summary after processing
finalSummary = summary.toDF(["Category/ID","Final Summary"])

#total_df dataframe have the document ID, News Article, Given Summary, System generated summary
total_df = articles_df.join(summaries_df,["Category/ID"]).join(finalSummary,["Category/ID"])
display(total_df)
    
    

# COMMAND ----------

#Rouge is a library that compares summaries in terms of Precision, Recall and F_measure.
rouge = Rouge()
recall = []
precision = []
f_measure = []
for rows in total_df.collect():
    #get_scores function will give recall, precision and f_measure by implementing unigram, bigram and longest matching sequence
    scores = rouge.get_scores(rows['Summary'], rows['Final Summary'])
    recall.append(max(scores[0]['rouge-1']['r'],scores[0]['rouge-2']['r'],scores[0]['rouge-l']['r']))
    precision.append(max(scores[0]['rouge-1']['p'],scores[0]['rouge-2']['p'],scores[0]['rouge-l']['p']))
    f_measure.append(max(scores[0]['rouge-1']['f'],scores[0]['rouge-2']['f'],scores[0]['rouge-l']['f']))
print("Average and maximum recall:")
print(numpy.mean(recall), max(recall))
print("Average and maximum precision:")
print(numpy.mean(precision), max(precision))
print("Average and maximum F_measure:")
print(numpy.mean(f_measure), max(f_measure))
recall1 = iter(recall)
precision1 = iter(precision)
f_measure1 = iter(f_measure)
final = total_df.rdd.map(lambda x: [x[0],x[1],x[2],x[3],next(recall1), next(precision1), next(f_measure1)] )
#output dataframe maps the respective recall, precision and F_measure for that particular documents
output = final.toDF(["Category/ID","Article", "Reference Summary", "LSA Summary", "Recall", "Precision", "F_measure"])
display(output)
plt.scatter(recall, [i for i in range(len(recall))])
plt.title("Recall")
plt.xlabel("Recall values")
plt.ylabel("Document IDs")
plt.show()
plt.scatter(precision, [i for i in range(len(precision))])
plt.title("Precision")
plt.xlabel("Precision values")
plt.ylabel("Document IDs")
plt.show()
plt.scatter(f_measure, [i for i in range(len(f_measure))])
plt.title("F_measure")
plt.xlabel("F_measure values")
plt.ylabel("Document IDs")
plt.show()

# COMMAND ----------


