# Анализ сайта weddingpost.site и возможности интеграции

## Обзор сайта

Сайт **weddingpost.site** - это платформа для создания персонализированных свадебных приглашений с расширенным функционалом для управления гостями и мероприятием.

## Технологический стек сайта

### Frontend:
- **jQuery 2.0.3** - для DOM манипуляций и AJAX
- **Bootstrap** - для адаптивной верстки
- **Font Awesome** - иконки
- **Canvas Confetti** - анимация конфетти
- **Yandex Maps API 2.1** - интерактивные карты
- **Twemoji** - эмодзи
- **Colorbox** - модальные окна для изображений

### Backend (предположительно):
- PHP (судя по путям `/cabinet/constructor/`, `/template/invent/userinvent.php`)
- Вероятно MySQL/PostgreSQL для хранения данных

## Основные функциональные модули

### 1. **Конструктор приглашений** (`module="mymain"`)
- Перетаскивание элементов (drag & drop)
- Редактирование текста, шрифтов, цветов
- Загрузка изображений
- Фоновые паттерны и декоративные элементы
- Обратный отсчет до свадьбы

### 2. **Система меню** (`module="menu"`)
- Адаптивное меню с иконками
- Якорные ссылки на секции
- Выпадающее меню для мобильных устройств

### 3. **Приглашение** (`module="myinv"`)
- Текстовая информация о свадьбе
- Фотографии жениха и невесты
- Информация о меню
- Пожелания по подаркам
- Примечания (например, про "Горько")
- Подтверждение присутствия
- Хештеги для фото

### 4. **Система RSVP** (`module="agree"`)
- Подтверждение/отмена присутствия
- Персонализация для каждого гостя
- Автоматическое оповещение жениха и невесты

### 5. **Опросы для гостей** (`module="opros"`)
- Множественные вопросы с вариантами ответов
- Типы вопросов: один ответ / несколько ответов
- Примеры вопросов:
  - Трансфер (да/нет, до/после)
  - Пищевые предпочтения (вегетарианство, аллергии)
  - Предпочтения по алкоголю
  - Наличие детей

### 6. **Расписание мероприятий** (`module="timetable"`)
- Список событий с временем и датой
- Адреса мероприятий
- Описания событий
- Редактирование расписания

### 7. **Интерактивная карта** (`module="map"`)
- Яндекс.Карты с координатами места проведения
- Поиск адресов
- Маршруты

### 8. **Комментарии и поздравления** (`module="comment"`)
- Гости могут оставлять сообщения
- Интеграция с системой денежных подарков
- Отображение автора и времени

### 9. **Система денежных подарков** (`module="money"`)
- Выбор суммы подарка из списка
- Интеграция с платежными системами (предположительно)
- Отображение подарков в комментариях

### 10. **Палитра цветов** (`module="palitra"`)
- Рекомендации по цветовой схеме нарядов
- Настраиваемая палитра

### 11. **Добавление в календарь** (`module="ad2cal"`)
- Генерация файлов календаря (.ics)
- Интеграция с Google Calendar, Outlook и др.

## Можно ли скопировать в ваше приложение?

### ✅ **ДА, но с адаптацией**

Ваше приложение на **ASP.NET Core 9.0 Razor Pages** может интегрировать большую часть функционала, но потребуется:

### Что можно использовать напрямую:

1. **HTML/CSS структура** - можно адаптировать
2. **JavaScript библиотеки** - все совместимы с ASP.NET Core
3. **Yandex Maps API** - работает в любом веб-приложении
4. **Дизайн и стили** - можно перенести

### Что нужно переписать:

1. **Backend логика** - с PHP на C#
2. **Система аутентификации** - использовать ASP.NET Core Identity
3. **База данных** - Entity Framework Core вместо PHP PDO
4. **API endpoints** - ASP.NET Core Web API вместо PHP скриптов
5. **Загрузка файлов** - ASP.NET Core file upload handlers

## Рекомендации по интеграции

### Этап 1: Базовая структура
- ✅ Создать модели данных (Guest, Event, Question, Answer, Comment)
- ✅ Настроить Entity Framework Core
- ✅ Создать базовые Razor Pages для каждой секции

### Этап 2: Функциональность
- ✅ RSVP система с персонализацией
- ✅ Опросы с сохранением ответов
- ✅ Расписание мероприятий
- ✅ Комментарии и поздравления

### Этап 3: Интеграции
- ✅ Яндекс.Карты API
- ✅ Система загрузки изображений
- ✅ Генерация календарных файлов

### Этап 4: Продвинутые функции
- ⚠️ Конструктор приглашений (drag & drop) - сложно, но возможно
- ⚠️ Система денежных подарков - требует интеграции платежных систем
- ⚠️ Палитра цветов - просто реализовать

## Необходимые зависимости для ASP.NET Core

```xml
<PackageReference Include="Microsoft.EntityFrameworkCore.SqlServer" Version="9.0.0" />
<PackageReference Include="Microsoft.EntityFrameworkCore.Tools" Version="9.0.0" />
<PackageReference Include="Microsoft.AspNetCore.Identity.EntityFrameworkCore" Version="9.0.0" />
```

## JavaScript библиотеки (можно добавить через CDN или npm)

```html
<!-- jQuery -->
<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.6.0/jquery.min.js"></script>

<!-- Bootstrap -->
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>

<!-- Font Awesome -->
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">

<!-- Yandex Maps -->
<script src="https://api-maps.yandex.ru/2.1/?lang=ru_RU"></script>

<!-- Canvas Confetti -->
<script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.5.1/dist/confetti.browser.min.js"></script>
```

## Структура базы данных (предложение)

```sql
-- Гости
Guests (Id, Name, Email, Phone, IsAttending, ConfirmedAt, WeddingId)

-- События
Events (Id, Title, Description, DateTime, Address, Coordinates, WeddingId)

-- Вопросы опроса
Questions (Id, Text, Type, WeddingId, Order)

-- Варианты ответов
QuestionOptions (Id, QuestionId, Text, Order)

-- Ответы гостей
GuestAnswers (Id, GuestId, QuestionId, OptionId, CustomText)

-- Комментарии
Comments (Id, GuestId, Text, GiftAmount, CreatedAt, WeddingId)

-- Свадьбы
Weddings (Id, GroomName, BrideName, WeddingDate, Theme, CreatedAt, UserId)
```

## Вывод

**Можно интегрировать**, но это потребует:

1. **Переписывания backend** с PHP на C#/ASP.NET Core
2. **Создания моделей данных** и миграций EF Core
3. **Адаптации JavaScript** к новой структуре API
4. **Настройки системы аутентификации** для персонализации
5. **Реализации загрузки файлов** для изображений

**Оценка сложности**: Средняя-Высокая
**Время разработки**: 2-4 недели (в зависимости от опыта)

## Следующие шаги

1. Создать модели данных
2. Настроить Entity Framework Core
3. Реализовать базовые Razor Pages
4. Интегрировать JavaScript библиотеки
5. Добавить функциональность по модулям
