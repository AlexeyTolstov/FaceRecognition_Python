# Распознавание лиц на Python

### Установка
``` bash
git clone https://github.com/AlexeyTolstov/FaceRecognition_Python
cd FaceRecognition_Python
pip install -r requirements.txt
```

### Создание датасета
В папке `Dataset` нужно разместить фотографии по папкам. На фотографиях должно быть четко видно лицо и желательно делать фотографии под разным ракурсом.

Структура папок `Dataset`
``` 
Dataset ->
    Name1 ->
        1.jpg
        ...
    Name2 ->
        1.jpg
        ...
```

Вы можете также воспользоваться готовым датасетом, представленным в репозитории по умолчанию


### Обучение модели

Для обучения модели вам необходимо запустить файл `training_model.py`.

```
python training_model.py
```
По окончанию создастся файл `face_enc`. Здесь и содержится модель.

### Запуск модели

Для запуска программы вам необходимо запустить файл `video_recognition.py`.

```
python video_recognition.py
```

В коде вы можете изменить камеру или указать видео, которое нужно распознавать.

### Результат

![Видео с результатом](Result.gif)
