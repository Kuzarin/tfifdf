Реализация веб-приложения. В качестве интерфейса выполнена страница с формой для загрузки текстового файла, после загрузки и обработки файла отображается таблица с 50 словами с колонками:

-    слово
-    tf, сколько раз это слово встречается в тексте
-    idf, обратная частота документа

Вывод упорядочен по уменьшению idf.

Ознакомиться с tfidf можно здесь: https://ru.wikipedia.org/wiki/TF-IDF.

Для справки:
1. Устанавливаем дополнительные библиотеки
pip install Flask pandas scikit-learn nltk

2. Запускаем приложение 
python3 app.py

3. Переходим в браузере по адресу http://127.0.0.1:5000/, чтобы увидеть интерфейс загрузки файла.
