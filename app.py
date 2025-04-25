from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    
    if file.filename == '':
        return "No selected file"
    
    # Читаем текст из файла
    text = file.read().decode('utf-8')
    
    # Обрабатываем текст для TF-IDF
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform([text])
    
    # Получаем слова и их TF-IDF значения
    words = vectorizer.get_feature_names_out()
    tfidf_values = X.toarray()[0]
    
    # Создаем DataFrame для удобного отображения
    df = pd.DataFrame({'word': words, 'tfidf': tfidf_values})
    
    # Сортируем по убыванию idf (в данном случае tfidf)
    df = df[df['tfidf'] > 0].sort_values(by='tfidf', ascending=False).head(50)
    
    return render_template('index.html', tables=[df.to_html(classes='data')], titles=df.columns.values)

if __name__ == '__main__':
    app.run(debug=True)
