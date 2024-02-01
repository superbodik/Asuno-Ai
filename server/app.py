# Импорты 
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from flask_dance.contrib.discord import make_discord_blueprint, discord
from werkzeug.security import generate_password_hash, check_password_hash
from textblob import TextBlob
from datetime import datetime
from langdetect import detect
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from flask_socketio import SocketIO, emit
from discord_app import discord_bp, profile_bp
import openai
import sqlite3

#Загрузка вспомагательных библиотек 
import logging
import os
import asyncio
import datetime
import pyttsx3

# Загружаем SpaCy для русского и украинского языков
import spacy

# Запуск Discord end Flask 2 in 1
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
import threading

engine = pyttsx3.init()
nlp_ru = spacy.load("ru_core_news_sm")

# Инициализация Flask приложения
app = Flask(__name__, static_url_path='/static')
app.config['SECRET_KEY'] = '2f45d0f56a2e5a38781e189ae8bc56ea'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
db = SQLAlchemy(app)

# Инициализация Flask-Login
login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Инициализация Flask-Dance
discord_bp = make_discord_blueprint(
    client_id='1198418658444189856',
    client_secret='Ro5AnZYCZloid8W9LWfwjisFf3t0NvyM',
)
app.register_blueprint(discord_bp, url_prefix='/discord_login')

socketio = SocketIO(app)

# Конфигурация журнала
logging.basicConfig(filename='app.log', level=logging.INFO)

# Модель пользователя
class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)
    average_rating = db.Column(db.Float, default=0.0)
    moods = db.relationship('Mood', backref='author', lazy=True)

# Модель настроения
class Mood(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    sentiment = db.Column(db.String(10), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    rating = db.Column(db.Integer, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow, nullable=False)

def train_neural_network():
    if os.path.exists('sentiment_model.h5'):
        logging.info('Neural network model already trained')
        return

    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

    # Векторизация текста
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    # Создание модели нейронной сети
    model = models.Sequential()
    model.add(layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))

    # Компиляция модели
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Обучение модели
    model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

    # Оценка модели
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test Accuracy: {accuracy}')

    # Сохранение модели
    model.save('sentiment_model.h5')

    # Журнал об успешном обучении
    logging.info('Neural network trained successfully!')

# Функция загрузки пользователя для Flask-Login
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Маршрут для главной страницы
@app.route('/')
def index():
    return render_template('index.html')

# Маршрут для регистрации нового пользователя
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        hashed_password = generate_password_hash(password)
        new_user = User(username=username, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        flash('Your account has been created!', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')

# Маршрут для входа пользователя
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = User.query.filter_by(username=username).first()

        if user and check_password_hash(user.password, password):
            login_user(user, remember=True)
            flash('Logged in successfully!', 'success')
            return redirect(url_for('analyze_mood'))

        flash('Login unsuccessful. Please check your username and password.', 'danger')
        return jsonify({'is_authenticated': False})

    return render_template('login.html')
# Маршрут для выхода пользователя
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

# Функция для определения языка и анализа текста
def detect_language(text):
    supported_languages = ['ru']
    
    try:
        language = detect(text)
        
        if language not in supported_languages:
            return {
                'language_info': {'language': 'unsupported', 'confidence': 1.0},
                'message': 'Анализ текста доступен только для русского языка.'
            }
            
        return {'language_info': {'language': language, 'confidence': 1.0}}
    except Exception as e:
        return {'language_info': {'language': 'unknown', 'confidence': 0.0, 'message': str(e)}}

#

@app.route('/analyze_mood', methods=['GET', 'POST'])
@login_required
def analyze_mood():
    response = None

    if request.method == 'POST':
        text = request.form['text']
        rating = int(request.form['rating'])

        # Определение языка текста
        language_info = detect_language(text)

        if 'language' not in language_info or language_info['language'] != 'ru':
            return render_template('analyze_mood.html', error_message='Анализ текста доступен только для русского языка.')

        # Продолжение с анализом только для русского языка
        doc = nlp_ru(text)

        # Выполнение анализа настроений с использованием TextBlob
        analysis = TextBlob(text)

        # Получение полярности настроения (-1 до 1) и уверенности (используем абсолютное значение для простоты)
        polarity = analysis.sentiment.polarity
        confidence = abs(polarity)

        # Определение настроения на основе полярности с оценкой от 1 до 10
        if polarity > 0:
            sentiment = 'положительное'
            rating = int(polarity * 5 + 5)  
        elif polarity < 0:
            sentiment = 'отрицательное'
            rating = int(-polarity * 5 + 5)  
        else:
            sentiment = 'нейтральное'
            rating = 5  

        # Создание нового объекта Mood и добавление его в базу данных
        new_mood = Mood(sentiment=sentiment, confidence=confidence, rating=rating, author=current_user)
        db.session.add(new_mood)

        # Создание нового объекта Message и добавление его в базу данных
        new_message = Message(content=text, user_id=current_user.id)
        db.session.add(new_message)

        # Пересчитывание среднего рейтинга
        current_user.average_rating = calculate_average_rating(current_user)
        db.session.commit()

        flash('Настроение успешно проанализировано!', 'success')

        # Получение обновленного среднего рейтинга пользователя
        average_rating = current_user.average_rating

        # Анализ текста с использованием OpenGPT
        opengpt_response = opengpt.complete(text, max_tokens=50)  # Замените 50 на желаемое количество токенов

        # Добавьте результат OpenGPT в response
        response = {
            'sentiment': sentiment,
            'confidence': confidence,
            'rating': rating,
            'average_rating': average_rating,
            'message': 'Настроение успешно проанализировано!',
            'opengpt_response': opengpt_response
        }

        # Журнал об успешном анализе настроения
        logging.info('Mood analysis successful!')

    return render_template('analyze_mood.html', response=response)
@app.route('/train_neural_network')
@login_required
def train_neural_network_route():
    # Отправляем событие для отображения всплывающего окна
    socketio.emit('show_popup', {'data': 'Training the neural network...'})
    
    train_neural_network()
    
    # Завершаем обучение и отправляем событие для закрытия всплывающего окна
    socketio.emit('hide_popup', {'data': 'Training complete!'})
    
    flash('Neural network trained successfully!', 'success')
    return redirect(url_for('index'))
    
# Функция для вычисления среднего рейтинга настроения
def calculate_average_rating(user):
    moods = Mood.query.filter_by(user_id=user.id).all()

    if not moods:
        return None

    total_rating = sum(mood.rating for mood in moods)
    average_rating = total_rating / len(moods)

    return round(average_rating)

#DISCORD APP AI
app.register_blueprint(profile_bp, url_prefix='/user')

#open-ai lib

openai.api_key = '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'

def generate_response(input_text):
    opengpt_response = openai.Completion.create(
        engine="gpt-3.5-turbo-0613",
        prompt=input_text,
        max_tokens=50,
        temperature=0.8  # Увеличьте значение temperature для более "эмоциональных" ответов
    )

    return opengpt_response.choices[0].text.strip()


#DISCORD LOGICA
import disnake
from disnake.ext import commands
import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import pygame


ffmpeg_path = r'C:\ffmpeg-master-latest-win64-gpl\bin\ffmpeg.exe'
AudioSegment.converter = ffmpeg_path
AudioSegment.ffmpeg = ffmpeg_path
AudioSegment.ffprobe = r'C:\ffmpeg-master-latest-win64-gpl\bin\ffprobe.exe'


anime_voice_file_path = ''

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.run_async()

intents = disnake.Intents.all()
bot = commands.Bot(command_prefix="!", intents=intents)

@bot.event
async def on_ready():
    print(f'Logged in as {bot.user.name} ({bot.user.id})')
    await bot.change_presence(activity=disnake.Game(name="私はAIボットです"), status=disnake.Status.online)

@bot.event
async def on_voice_state_update(member, before, after):
    if after.channel and after.channel.id == 1170667312454565889:
        # Пользователь присоединился к указанному голосовому каналу
        print(f'{member.display_name} joined voice channel {after.channel.name}')

        # Получение данных об общении
        # В данном примере просто используем member.display_name
        data = f'{member.display_name} joined voice channel {after.channel.name}\n'

        # Запись данных в текстовый файл
        with open('voice_data.txt', 'a', encoding='utf-8') as file:
            file.write(data)

@bot.event
async def on_message(message):
    # Обработка команд в текстовом канале
    await bot.process_commands(message)

    # Проверка, что сообщение отправлено в личные сообщения боту
    if isinstance(message.channel, disnake.DMChannel) and message.author != bot.user:
        try:
            # Ваш код обработки сообщений в личные сообщения
            recognized_text = message.content
            response = await generate_response(recognized_text)
            await message.author.send(f"Ответ Asuno: {response}")

        except Exception as e:
            print(f"Произошла ошибка при обработке сообщения в личные сообщения: {e}")
    
    # Проверка, что сообщение отправлено в текстовый канал
    elif isinstance(message.channel, disnake.TextChannel) and message.author != bot.user:
        try:
            # Ваш код обработки текстовых сообщений в текстовом канале
            recognized_text = message.content
            response = await generate_response(recognized_text)
            await message.channel.send(f"Ответ Asuno: {response}")

        except Exception as e:
            print(f"Произошла ошибка при обработке текстового сообщения: {e}")

@bot.command(name='ping')
async def ping(ctx):
    await ctx.send('Pong!')

async def listen_and_respond(voice_channel):
    recognizer = sr.Recognizer()

    print("Готов к прослушиванию...")

    while voice_channel.is_connected():
        try:
            with sr.Microphone() as source:
                audio = recognizer.listen(source, timeout=None)

            recognized_text = recognizer.recognize_google(audio, language="ru-RU")
            print(f"Распознанный текст: {recognized_text}")
          # Проверка наличия команды /leave
            if recognized_text.lower().startswith("/leave"):
                await voice_channel.disconnect()
                print("Бот отключен от голосового канала.")
                break

            # Find answer based on keywords
            answer = find_answer(recognized_text)

            if answer:
                await voice_channel.send(answer)
            else:
                # If no keyword match, use GPT-3 response
                response = generate_response(recognized_text)
                await voice_channel.send(f"Ответ Asuno: {response}")

        except sr.UnknownValueError:
            print("Не удалось распознать речь.")
        except sr.RequestError as e:
            print(f"Произошла ошибка при запросе к Google Web Speech API: {e}")
        except Exception as e:
            print(f"Произошла неизвестная ошибка: {e}")

        await asyncio.sleep(1)  # Ждем 1 секунду перед следующей попыткой

@bot.command(name='connect')
async def connect(ctx):
    try:
        voice_state = ctx.author.voice
        if voice_state is None:
            await ctx.send("Вы не находитесь в голосовом канале.")
            return

        channel = voice_state.channel
        voice_channel = await channel.connect()

        recognizer = sr.Recognizer()

        print("Готов к прослушиванию...")

        while voice_channel.is_connected():
            try:
                with sr.Microphone() as source:
                    audio = recognizer.listen(source, timeout=None)

                recognized_text = recognizer.recognize_google(audio, language="ru-RU")
                print(f"Распознанный текст: {recognized_text}")

                # Проверка наличия команды /leave
                if recognized_text.lower().startswith("/leave"):
                    await voice_channel.disconnect()
                    print("Бот отключен от голосового канала.")
                    break

                # Используйте await при вызове generate_response
                response = await generate_response(recognized_text)


            except sr.UnknownValueError:
                print("Не удалось распознать речь.")
            except sr.RequestError as e:
                print(f"Произошла ошибка при запросе к Google Web Speech API: {e}")
            except Exception as e:
                print(f"Произошла неизвестная ошибка: {e}")

            await asyncio.sleep(1)  # Ждем 1 секунду перед следующей попыткой
    except Exception as e:
        print(f"Произошла ошибка: {e}")

@bot.command(name='speak')
async def speak(ctx):
    try:
        channel = ctx.author.voice.channel
        voice_channel = await channel.connect()

        await speak_async("Привет, это тест!", voice_channel)

    except Exception as e:
        print(f"Произошла ошибка: {e}")

@bot.command(name='database_check')
async def database_check(ctx):
    query = "Как дела?"
    answer = find_answer(query)
    await ctx.send(f"Запрос: {query}\nОтвет из базы данных: {answer}")

 
def create_database():
    try:
        # Создаем базу данных и таблицу для примера
        connection = sqlite3.connect('answers_database.db')
        cursor = connection.cursor()

        # Создаем таблицу с колонками question и answer
        cursor.execute('''CREATE TABLE IF NOT EXISTS answers
                          (question TEXT PRIMARY KEY, answer TEXT)''')

        # Вставляем несколько примеров
        cursor.executemany('''INSERT OR REPLACE INTO answers (question, answer)
                               VALUES (?, ?)''', [
                               ('Как дела?', 'Отлично, спасибо!'),
                               ('Как твои?', 'Отлично, спасибо!'),
                               ('Кто такая Юки Асуна?', 'Я персонаж из Sword Art Online'),
                               ('Что делаешь', 'Общаюсь с вами!'),
                               ('привет', 'Привет! Чем я могу помочь?')])

        connection.commit()
        connection.close()

        print("Database created and populated.")

    except Exception as e:
        print(f"Error creating database: {e}")

# Вызываем функцию создания базы данных

create_database()
async def find_answer_async(query):
    connection = sqlite3.connect('answers_database.db')
    cursor = connection.cursor()

    try:
        # Проверка наличия конкретного запроса в базе данных
        cursor.execute("SELECT answer FROM answers WHERE question = ?", (query,))
        result = cursor.fetchone()
        print("SQLite query result:", result)
    except Exception as e:
        print(f"Error querying SQLite database: {e}")
    finally:
        connection.close()

    # Если найден конкретный ответ, вернуть его, в противном случае использовать ключевые слова
    if result:
        return result[0]
    else:
        # Если конкретный ответ не найден, проверяем ключевые слова
        query_lower = query.lower()
        keyword_responses = {
            'Асуно': 'Что?',
            'aсуно': 'Что?',
            'aсуна': 'Что?',
            'Юкки': 'А????',
            'привет': 'Привет! Чем я могу помочь?',
            'как твои дела': 'У меня все отлично, спасибо!',
            'как дела': 'У меня все отлично, спасибо!',
            'как дела твои ': 'У меня все отлично, спасибо!',
            'как тебя зовут': 'Меня зовут Юки Асуна. Чем я могу вам помочь?',
            'что делаешь': 'Общаюсь с вами!',
            'кто такая юки асуна': 'Я Ai вайфу от разработчика @super_bodik'
            # Добавьте больше ключевых слов по мере необходимости
        }

        for keyword, response in keyword_responses.items():
            if keyword in query_lower:
                return response

        # Если ничего не найдено, вернуть None
        return None

def find_answer(query):
    connection = sqlite3.connect('answers_database.db')
    cursor = connection.cursor()

    try:
        # Проверка наличия конкретного запроса в базе данных
        cursor.execute("SELECT answer FROM answers WHERE question = ?", (query,))
        result = cursor.fetchone()
        print("SQLite query result:", result)
    except Exception as e:
        print(f"Error querying SQLite database: {e}")
    finally:
        connection.close()

    # Если найден конкретный ответ, вернуть его, в противном случае использовать ключевые слова
    if result:
        return result[0]
    else:
        # Если конкретный ответ не найден, проверяем ключевые слова
        query_lower = query.lower()
        keyword_responses = {
            'Асуно': 'Что?',
            'Юкки': 'А????',
            'привет': 'Привет! Чем я могу помочь?',
            'как твои дела': 'У меня все отлично, спасибо!',
            'как дела': 'У меня все отлично, спасибо!',
            'как дела твои ': 'У меня все отлично, спасибо!',
            'как тебя зовут': 'Меня зовут Юки Асуна. Чем я могу вам помочь?',
            'что делаешь': 'Общаюсь с вами!',
            'кто такая юки асуна': 'Я Ai вайфу от разработчика @super_bodik'
            # Добавьте больше ключевых слов по мере необходимости
        }

        for keyword, response in keyword_responses.items():
            if keyword in query_lower:
                return response

        # Если ничего не найдено, вернуть None
        return None
        
async def generate_response_async(recognized_text):
    try:
        # Первым делом, пытаемся найти ответ в базе данных
        database_answer = await find_answer_async(recognized_text)
        if database_answer:
            return database_answer

        # Если не найдено, используем GPT-3.5-turbo
        response = await asyncio.to_thread(openai.ChatCompletion.create, model="gpt-3.5-turbo", messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": recognized_text}])
        print("Generated response:", response.choices[0].message['content'])
        return response.choices[0].message['content']
    except Exception as e:
        print(f"Error generating response: {e}")
        return "An error occurred while generating a response."

        
async def generate_response(recognized_text):
    try:
        response = await generate_response_async(recognized_text)
        
        # Передаем текст для озвучки
        await speak_async(response)
        
        return response
    except Exception as e:
        print(f"Error generating response: {e}")
        return "An error occurred while generating a response."

mp3_file = 'temp.mp3'

try:
    if not os.path.exists(mp3_file):
        print(f"Ошибка: Файл '{mp3_file}' не найден.")
    else:
        sound = AudioSegment.from_mp3(mp3_file)
        sound.export('temp.wav', format='wav')
except Exception as e:
    print(f"Произошла ошибка во время обработки аудио: {e}")

with app.app_context():
    async def speak_async(text, voice_channel):
        try:
            with app.app_context():
                await voice_channel.send(text)
        except Exception as e:
            print(f"Произошла ошибка во время speak_async: {e}")

        try:
            with app.app_context():
                # Сохраняем mp3 файл
                tts = gTTS(text=text, lang='ru')
                tts.save('temp.mp3')

                # Проверяем существование файла перед его загрузкой
                if os.path.exists('temp.mp3'):
                    # Загружаем mp3 файл
                    sound = AudioSegment.from_mp3('temp.mp3')
                    sound.export('temp.wav', format='wav')

                    # Воспроизводим звук
                    play(sound)
                else:
                    print("Файл 'temp.mp3' не существует.")

        except Exception as e:
            print(f"Произошла ошибка во время speak_async: {e}")

        finally:
            # Удаляем временные файлы
            try:
                os.remove('temp.mp3')
                os.remove('temp.wav')
            except Exception as e:
                print(f"Произошла ошибка во время удаления временных файлов: {e}")
#Генерация Секретного ключа .env
import secrets

env_file_path = '.env'

if not os.path.exists(env_file_path):
    with open('.env.example', 'r') as example_file:
        content = example_file.read()
        with open('.env', 'w') as env_file:
            env_file.write(content)

# Получаем текущий секретный ключ или генерируем новый
current_secret_key = os.environ.get('FLASK_SECRET_KEY')
if current_secret_key is None:
    new_secret_key = secrets.token_hex(16)
    print(f'Generated new SECRET_KEY: {new_secret_key}')
else:
    print(f'Current SECRET_KEY: {current_secret_key}')

with open('.env', 'r') as env_file:
    lines = env_file.readlines()
    with open('.env', 'w') as env_file:
        for line in lines:
            if line.startswith('FLASK_SECRET_KEY='):
                env_file.write(f'FLASK_SECRET_KEY={current_secret_key}\n')
            else:
                env_file.write(line)

# Если файл запускается напрямую, а не импортируется
load_dotenv()

app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'fallback_secret_key')

executor = ThreadPoolExecutor(max_workers=2)


# Обновление функции run_flask
def run_flask():
    app.config['ENV'] = 'development'
    app.config['DEBUG'] = True
    app.config['USE_RELOADER'] = False

    # Асинхронная инициализация Flask-SocketIO с использованием eventlet
    socketio.init_app(app, async_mode='eventlet')

    # Запуск Flask и SocketIO
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, use_reloader=False)

    # Запуск Discord-бота в основном потоке
    run_discord()

def run_discord():
    bot.run('MTE5ODcyODMwMjgwNjk2NjM0Mg.G12jkl.n3uv7ykUcG5JBAmEtjUVMF5FWiuPWQlMCCAu1U')

# Запуск Flask в основном потоке
if __name__ == '__main__':
    with app.app_context():
        db.create_all()

    flask_thread = threading.Thread(target=run_flask)
    discord_thread = threading.Thread(target=run_discord)
    
    flask_thread.start()
    discord_thread.start()

    flask_thread.join()
    discord_thread.join()