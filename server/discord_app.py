from flask import Flask, render_template, redirect, url_for
from flask_dance.contrib.discord import make_discord_blueprint, discord
from flask_login import LoginManager
from your_profile_module import profile_bp

app = Flask(__name__)
app.config['SECRET_KEY'] = 'Ro5AnZYCZloid8W9LWfwjisFf3t0NvyM'  # Замените на свой секретный ключ

# Настройка Flask-Login
login_manager = LoginManager(app)
login_manager.login_view = 'discord.login'

# Настройка Flask-Dance для Discord
discord_bp = make_discord_blueprint(
    client_id='1198418658444189856',
    client_secret='Ro5AnZYCZloid8W9LWfwjisFf3t0NvyM',
)

# Регистрация Blueprint для профиля
app.register_blueprint(profile_bp, url_prefix='/profile')

# Пример использования discord_bp
@discord_bp.route('/discord_login')
def discord_login():
    # Ваш код для взаимодействия с Discord API
    return redirect(url_for('home'))  # Пример перенаправления на главную страницу

# Обработчик события при успешном входе пользователя через Discord
@discord_bp.route('/discord_logged_in')
def discord_logged_in():
    # Получение информации о пользователе
    resp = discord.get('/api/users/@me')

    if resp.ok:
        user_info = resp.json()
        # Добавьте код для обработки информации о пользователе, например, сохранение в базе данных
        return redirect(url_for('home'))
    else:
        # Обработка ошибки входа
        return 'Error logging in with Discord'

# Главная страница
@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
