#通过训练Doc2Vec模型以便将文本向量化供后续模型训练
import nltk,gensim,string,csv,time,torch
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.test_doc2vec import ConcatenatedDoc2Vec
import os
from multiprocessing import cpu_count

def Preprocessing(text):
    text = text.lower() # 将所有的单词转换成小写字母

    for c in string.punctuation:
        text = text.replace(c," ")  # 将标点符号转换成空格

    wordList = nltk.word_tokenize(text)  # 分词

    filtered = [w for w in wordList if (w == "no" or ( w not in stopwords.words('english')) and w != "xxxx")] # 删除停顿词和脱敏词

    ps = PorterStemmer()
    filtered = [ps.stem(w) for w in filtered]  # 提取词干
    wl = WordNetLemmatizer()
    filtered = [wl.lemmatize(w) for w  in filtered]  # 词形还原
    return filtered
#对影像报告进行分词、提取词干、删除停用词和脱敏词等操作

def get_text_feature(texts):
    model_dbow = Doc2Vec.load(os.path.dirname(os.path.abspath(__file__))+'/doc2vector_dbow.doc2vec')
    model_dm = Doc2Vec.load(os.path.dirname(os.path.abspath(__file__))+'/doc2vector_dm.doc2vec')
    model = ConcatenatedDoc2Vec([model_dm, model_dbow])
    result = []
    i = 1
    for text in texts:
        if i % 1000 == 0:
            print(i,"texts of",len(texts),"texts have been loaded")
        i += 1
        result.append(model.infer_vector(document=Preprocessing(text),alpha=0.025,min_alpha=0.025,steps=100))
    return (torch.tensor(result)+1)/2
#批量将文本转化为向量并返回

if __name__ == '__main__':
    f = open('F:/data/cxr/report/indiana_reports.csv',encoding='utf-8')
    cxr = list(csv.reader(f))[1:]
    read = [Preprocessing(cxr[i][6]) for i in range(len(cxr)) if cxr[i][6] != ""]
    f1 = open('F:/data/mpx/x/report.csv',encoding='utf-8')
    mpx = list(csv.reader(f1))[1:]
    read += [Preprocessing(mpx[i][1]) for i in range(len(mpx))]
    documents = [TaggedDocument(read[i],[i]) for i in range(len(read))]

    model_dbow = Doc2Vec(documents=documents,dm=0,vector_size=3000,workers=cpu_count(),alpha=0.025,min_alpha=0.025)
    model_dm = Doc2Vec(documents=documents, dm=1, vector_size=3000, workers=cpu_count(), alpha=0.025, min_alpha=0.025)
    model_dbow.wv.similarity()
    for epoch in range(10):
        begin = time.time()
        model_dbow.train(documents,total_examples=model_dbow.corpus_count,epochs=10)
        model_dbow.alpha -= 0.002
        model_dbow.min_alpha = model_dbow.alpha
        print("epoch:{e} running time: {n}".format(e=str(epoch), n=str(time.time() - begin)))

    for epoch in range(10):
        begin = time.time()
        model_dm.train(documents, total_examples=model_dm.corpus_count, epochs=10)
        model_dm.alpha -= 0.002
        model_dm.min_alpha = model_dm.alpha
        print("epoch:{e} running time: {n}".format(e=str(epoch), n=str(time.time() - begin)))


    model_dbow.save('doc2vector_dbow.doc2vec')
    model_dm.save('doc2vector_dm.doc2vec')
#使用Doc2Vec训练cxr所有的报告文本并将训练模型保存

