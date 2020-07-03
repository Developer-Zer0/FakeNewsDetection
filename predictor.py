import pickle
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.data import Field,BucketIterator,TabularDataset
import torchtext
from flask import Flask
from flask import request,jsonify
from nltk.tokenize import sent_tokenize,word_tokenize
from gensim.summarization.summarizer import summarize 
from gensim.summarization import keywords

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
app = Flask(__name__)

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

def summarize_text(article):
    summarized_article = summarize(article,word_count = 50)
    return summarized_article
    
@app.route('/get_prediction',methods=['POST','GET'])
def get_predictions():
    data = request.get_json()
    summarized_text = data['text']
    if len(sent_tokenize(summarized_text)) > 5 or len(word_tokenize(summarized_text)) > 50: 
        summarized_text = summarize_text(summarized_text)
    predictions = predict(summarized_text)
    return jsonify({'prediction':predictions})


if __name__ == '__main__':
    app.run(debug=True)