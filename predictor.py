import pickle
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.data import Field,BucketIterator,TabularDataset
import torchtext
from flask import Flask
from flask import request,jsonify,render_template
from nltk.tokenize import sent_tokenize,word_tokenize
from gensim.summarization.summarizer import summarize 
from gensim.summarization import keywords
from torchtext import data
from torchtext import vocab
import pickle
import pandas as pd
import re
import numpy as np
from werkzeug.utils import secure_filename
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] ='/home/talha/Desktop/fakeit/'

class Model(nn.Module):
    def __init__(self,num_layers,pad_index,batch_size,vocab_size,embedding_matrix,embedding_dimensions,hidden_size,bidirectional):
        super().__init__()
        self.embedding_dimensions = embedding_dimensions
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.embedding_layer = nn.Embedding(vocab_size,self.embedding_dimensions,padding_idx = pad_index)
        self.embedding_layer.weight = nn.Parameter(torch.tensor(embedding_matrix,dtype=torch.float32))
        self.embedding_layer.weight.requires_grad = False
        self.lstm_layer = nn.LSTM(embedding_dimensions,hidden_size,bidirectional = self.bidirectional,num_layers = self.num_layers,batch_first = True)
        self.output_layer = nn.Linear(hidden_size,1)
        
        
    def forward(self,x):
        embedding_outputs = self.embedding_layer(x)  
        h_n_1,c_n_1 = self.__init__hidden()
        output,(h_n_1,c_n_1) = self.lstm_layer(embedding_outputs,(h_n_1,c_n_1))        
        linear_output_1 = self.output_layer(h_n_1[-1])
        return linear_output_1
    
    def __init__hidden(self):
        return nn.Parameter(torch.zeros(1,self.batch_size,self.hidden_size,dtype=torch.float32,device=device),requires_grad=True),nn.Parameter(torch.zeros(1,self.batch_size,self.hidden_size,dtype=torch.float32,device=device),requires_grad=True)


model_cpu = pickle.load(open('./model_cpu.pkl','rb'))
loaded_field = pickle.load(open('./text_field.pkl','rb')) 

def predict(text):
    text = str(text)
    text = text.lower()
    split_text = text.split()
    tokenized_text = []
    for token in split_text:
        if token not in dict(loaded_field.stoi).keys():
            tokenized_text.append(loaded_field.stoi['<unk>'])
        else:
            tokenized_text.append(loaded_field.stoi[token])
    clone_list = []
    for i in range(32):
        clone_list.append(tokenized_text)

    clone_list = torch.tensor(clone_list,device = device)
    predictions = model_cpu(clone_list)
    output = torch.sigmoid(predictions[0][0])
    if output.item() > 0.6:
        return "Fake"
    else:
        return "True"
    return output

def return_accuracy(logits,label):
    sigmoid = nn.Sigmoid()(logits)
    predictions = torch.round(sigmoid)
    predictions = predictions.view(32)
    return (predictions == label).sum().float()/float(label.size(0))

def summarize_text(article):
    summarized_article = summarize(article,word_count = 50)
    return summarized_article

def clean_contractions(text, mapping):
    text = text.lower()
    specials = ["’", "‘", "´", "`"]
    for s in specials:
        text = text.replace(s, "'")
    text = ' '.join([mapping[t] if t in mapping else mapping[t.lower()] if t.lower() in mapping else t for t in text.split(" ")])
    return text

def remove_newlines(sent):
    sent = re.sub(r'\s+', " ", sent )
    return sent


def create_embedding_matrix(field,embeddings):  
    embedding_matrix = np.random.rand(len(field.vocab.stoi),100)
    for string,index in field.vocab.stoi.items():
        if not  all(x == 0 for x in embeddings[string].tolist()):
            embedding_matrix[index] = embeddings[string] 
    return embedding_matrix

@app.route('/form',methods=['POST','GET'])
def model_form():
    return render_template('index.html')


@app.route('/train',methods=['POST','GET'])
def train():
    learning_rate = float(request.form['learning_rate'])
    epochs = int(request.form['epochs'])
    csv_file = request.files['data']
    csv_file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'data.csv'))

    train_df = pd.read_csv('./data.csv')
    valid = train_df[15000:]
    df_trn = pd.DataFrame({'text':train_df['title'].values, 'labels':train_df['label'].values})
    df_val = pd.DataFrame({'text':valid['title'].values, 'labels':valid['label'].values})
    df_trn = df_trn.sample(frac=1)
    df_val = df_val.sample(frac=1)
    contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not", "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not", "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",  "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am",'i\'m':'i am', "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have","you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have", 'colour': 'color', 'centre': 'center', 'favourite': 'favorite', 'travelling': 'traveling', 'counselling': 'counseling', 'theatre': 'theater', 'cancelled': 'canceled'}
    df_trn['text'] = df_trn['text'].apply(lambda x: clean_contractions(str(x),contraction_mapping))
    df_val['text'] = df_val['text'].apply(lambda x: clean_contractions(str(x),contraction_mapping))
    df_trn['text'] = df_trn['text'].apply(lambda x: remove_newlines(str(x)))
    df_val['text'] = df_val['text'].apply(lambda x: remove_newlines(str(x)))
    df_trn.to_csv('train.csv',index = False)
    df_val.to_csv('validation.csv',index = False)
    tokenizer = lambda s: s.lower().split()
    text1 = data.Field(tokenize=tokenizer,batch_first=True,include_lengths=True)
    label = data.Field(sequential=False, use_vocab=False, pad_token=None, unk_token=None)
    fields = [('text',text1),('labels',label)]
    train_data, valid_data = data.TabularDataset.splits(path='./',train='train.csv',validation = 'validation.csv',format='csv',fields=fields,skip_header=True)
    text1.build_vocab(train_data,valid_data)
    label.build_vocab(train_data,valid_data)
    embeddings = pickle.load(open('./embeddings.pkl','rb'))
    text1.build_vocab(train_data,valid_data,vectors = embeddings)
    label.build_vocab()
    train_itr,valid_itr = data.BucketIterator.splits((train_data,valid_data),batch_size = 32,sort_key = lambda x: len(x.text),sort_within_batch = True,device = device)
    embedding = create_embedding_matrix(text1,embeddings)
    model = Model(pad_index = text1.vocab.stoi[text1.pad_token],
            batch_size = 32,
            vocab_size = len(text1.vocab),
            embedding_matrix = embedding,
            embedding_dimensions = 100,
            hidden_size = 512,
            bidirectional = False,
            num_layers = 1
            )
    model = model.to(device = device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(),lr = learning_rate) 
    

    model.train()
    for epoch in range(epochs):
        print("Epoch {} out of {}".format(epoch,epochs))
            
        epoch_train_loss = 0
        epoch_train_accuracy = 0
            
        epoch_valid_loss = 0
        epoch_valid_accuracy = 0
            
        for batch in train_itr:
            model.train()
            optimizer.zero_grad()
                
            text = batch.text[0]
            label = batch.labels
                
            if text.size()[0] != 32:
                continue
                
            text.to(device)
            label.to(device)

            label = torch.tensor(label,dtype= torch.float32,device = device)
            predictions = model(text)

            loss = criterion(torch.sigmoid(predictions),label.unsqueeze(1))
                
            loss.backward()
            optimizer.step()                

            batch_loss = loss.item()/len(batch)
            batch_accuracy = return_accuracy(predictions,label)
                
            epoch_train_loss += loss.item()
            epoch_train_accuracy += batch_accuracy.item()

        print("Epoch Train Accuracy: ",epoch_train_accuracy/len(train_itr))
        print("Epoch Train Loss: ",epoch_train_loss/len(train_itr))
    return render_template('train.html',accuracy=epoch_train_accuracy/len(train_itr),epoch=epochs)


@app.route('/get_prediction',methods=['POST','GET'])
def get_predictions():
    data = request.get_json()
    summarized_text = data['text']
    if len(sent_tokenize(summarized_text)) > 5 or len(word_tokenize(summarized_text)) > 50: 
        summarized_text = summarize_text(summarized_text)
    predictions = predict(summarized_text)
    print(predictions)
    return jsonify({'prediction':predictions})


if __name__ == '__main__':
    app.run(debug=True)