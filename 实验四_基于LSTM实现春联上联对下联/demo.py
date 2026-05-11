import json
import torch
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)
def couplet_match(s, device='cpu'):
    # 将字符串转为数值
    with open('./save/word2idx.json', 'r') as f:
        word2idx = json.load(f)
    with open('./save/idx2word.json', 'r') as f:
        idx2word = json.load(f)
    model_path = './save/best_model.pt'
    model = torch.load(model_path)
    model.to(device)
    # x = [word2idx[word] for word in s]
    x = []
    for word in s:
        if word2idx.get(word) != None:
            x.append(word2idx[word])
    if len(x) == 0: return ''
    # 将数值向量转为tensor
    x = torch.from_numpy(np.array(x).reshape(-1, 1))

    y = model(x)
    y = y.argmax(axis=1)
    # r = ''.join([idx2word.get(str(idx.item())) for idx in y])
    r = ''
    for idx in y:
        word = idx2word.get(str(idx.item()))
        if word == "UNK":
            word = idx2word.get(str(np.random.randint(0, len(idx2word) - 1)))
        r += word
    print('上联：%s，下联：%s' % (s, r))
    return r

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/execute_function", methods=["POST"])
def execute_function():
    data = request.json
    function_name = data["function_name"]
    input_text = data["input_text"]
    result = ""
    if function_name == "analyze":
        result = couplet_match(input_text)
    return result

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
    # couplet_match('')