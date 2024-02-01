from flask import Blueprint, render_template

profile_bp = Blueprint('profile', __name__, template_folder='templates')

@profile_bp.route('/profile')
def profile():
    return render_template('profile.html')  # Замените 'profile.html' на ваш шаблон

# Добавьте другие маршруты и функции по необходимости