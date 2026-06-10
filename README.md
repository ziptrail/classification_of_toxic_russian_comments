# Classification of Russian Toxic Comments

Веб-приложение на Flask для определения токсичности русскоязычных комментариев.  
Поддерживаются три архитектуры нейросетей: **LSTM**, **GRU**, **Conv1D**.  
Для каждого текста предсказываются вероятности четырёх классов:  
`normal`, `insult`, `threat`, `obscenity`.


## Требования

- Python 3.10 или выше
- TensorFlow 2.18 (для совместимости с сохранёнными моделями)


## Установка и запуск

### 1. Клонируйте репозиторий
```bash
git clone https://github.com/ziptrail/classification_of_toxic_russian_comments.git
cd classification_of_toxic_russian_comments
```

### 2. Установите зависимости
```bash
pip install flask==3.1.0 tensorflow==2.21.0 h5py==3.12.1 numpy==2.1.3
```

### 3. Скачайте модели

- **LSTM:** (https://cloud.mail.ru/public/1zRQ/MBEBt8T6b)  
- **GRU:** (https://cloud.mail.ru/public/9zWo/xHh6mfn5G)  
- **Conv1D:** (https://cloud.mail.ru/public/V95x/Bssk7myMp)

### 4. Запустите приложение
```bash
python app.py
```

И перейдите по адресу http://127.0.0.1:5000 (или другой порт, в зависимости от того, какой он у вас указан)

Контакты

Автор: Александр Кочелаев 
E-mail: sanya.kochelaev@mail.ru
