#discord_auth.py 

import requests
from discord_app import discord_bp

client_id = '1198418658444189856'
client_secret = 'Ro5AnZYCZloid8W9LWfwjisFf3t0NvyM'
redirect_uri = 'https://discord.com/api/oauth2/authorize?client_id=1198418658444189856&response_type=code&redirect_uri=http%3A%2F%2F185.229.224.238%3A5000%2F&scope=email+identify+connections+guilds+guilds.join+voice+messages.read'
code = 'код_авторизации'

data = {
    'client_id': client_id,
    'client_secret': client_secret,
    'grant_type': 'authorization_code',
    'code': code,
    'redirect_uri': redirect_uri,
    'scope': 'ваш_scope',
}

response = requests.post('https://discord.com/api/oauth2/token', data=data)
token_data = response.json()

access_token = token_data.get('access_token')
refresh_token = token_data.get('refresh_token')
