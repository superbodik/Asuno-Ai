document.getElementById('roundButton').addEventListener('click', function() {
    // Отправляем AJAX-запрос на сервер
    var xhr = new XMLHttpRequest();
    xhr.open('GET', '/check_authentication', true);

    xhr.onload = function() {
        if (xhr.status === 200) {
            var isAuthenticated = JSON.parse(xhr.responseText).is_authenticated;

            if (isAuthenticated) {
                alert('Пользователь аутентифицирован!');
                // Ваш код для обработки аутентифицированного пользователя
            } else {
                alert('Пользователь не аутентифицирован!');
                // Ваш код для обработки неаутентифицированного пользователя
            }
        }
    };

    xhr.send();
});
document.getElementById('roundButton').addEventListener('click', function() {
    var menu = document.querySelector('.menu');
    menu.style.display = (menu.style.display === 'none' || menu.style.display === '') ? 'block' : 'none';
});