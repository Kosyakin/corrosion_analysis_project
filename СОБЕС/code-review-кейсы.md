# Code Review — кейсы для практики

Ниже даётся кейс (код с недостатками).  
**Твоя задача:** скопировать код кейса и под ним написать:
- какие **риски** видишь;
- какие **баги** возможны;
- что бы ты **улучшил** (свой вариант кода или конкретные правки).

---

## Кейс 1

**Контекст:** метод сервиса, который по id пользователя возвращает его данные из БД.

```csharp
public User GetUser(string userId)
{
    var connection = new SqlConnection(_connectionString);
    connection.Open();
    
    var query = "SELECT * FROM Users WHERE Id = " + userId;
    var command = new SqlCommand(query, connection);
    var reader = command.ExecuteReader();
    
    var user = new User();
    if (reader.Read())
    {
        user.Id = reader.GetInt32(0);
        user.Name = reader.GetString(1);
        user.Email = reader.GetString(2);
    }
    
    connection.Close();
    return user;
}
```

---

**Твой разбор (заполни ниже):**
Отсутсвует using  у sqlConnection и у reader
В запросе * - возможны утечки струтуры БД
Теоретически возможен баг, если например несколько юзеров с одним id тогда мне кажется не отработает нормально reader, но я не до конца уверен, что это так.
Id может храниться не в формате Int

**Предлагаемые улучшения (код или список правок):**


---

### Как исправить Кейс 1 (эталонный вариант)

1. **Параметризованный запрос** — убираем SQL-инъекцию: подставляем значение через `SqlParameter`.
2. **`using`** — гарантируем вызов `Dispose` у `SqlConnection` и `SqlDataReader` даже при исключении.
3. **Явный список полей** — вместо `SELECT *` перечисляем нужные колонки.
4. **Тип Id** — если в БД `uniqueidentifier`, используем `Guid`; если `bigint` — `long`. В примере ниже предполагаем, что Id в БД — `int` или совместимый тип; параметр оставлен `string` для входа, но значение передаём в запрос через параметр нужного типа.

```csharp
public User? GetUser(string userId)
{
    // Валидация входа (опционально, но полезно)
    if (string.IsNullOrWhiteSpace(userId))
        return null;

    const string query = @"
        SELECT Id, Name, Email 
        FROM Users 
        WHERE Id = @UserId";

    using var connection = new SqlConnection(_connectionString);
    connection.Open();

    using var command = new SqlCommand(query, connection);
    command.Parameters.AddWithValue("@UserId", userId); // или Add("@UserId", SqlDbType.Int).Value = int.Parse(userId);

    using var reader = command.ExecuteReader();

    if (!reader.Read())
        return null;

    return new User
    {
        Id = reader.GetInt32(0),
        Name = reader.GetString(1),
        Email = reader.GetString(2)
    };
}
```

**Замечания:**
- Возврат `User?` и `return null`, если пользователь не найден — вызывающий код явно обрабатывает отсутствие.
- `AddWithValue` удобен, но для строгой типизации лучше `command.Parameters.Add("@UserId", SqlDbType.NVarChar, 50).Value = userId` (или соответствующий тип/размер колонки Id).
- Если Id в БД — число, то парсить `userId` в `int`/`long` с проверкой и передавать в параметр типа `SqlDbType.Int`/`BigInt`.

---

## Кейс 2

**Контекст:** метод контроллера/сервиса, который по URL загружает JSON и возвращает строку с данными.

```csharp
public string LoadData(string url)
{
    using var client = new HttpClient();
    var response = client.GetAsync(url).Result;
    response.EnsureSuccessStatusCode();
    return response.Content.ReadAsStringAsync().Result;
}
```

---

**Твой разбор (заполни ниже):**

**Риски:**


**Возможные баги:**


**Предлагаемые улучшения (код или список правок):**

```csharp
public async Task<string> LoadData(string url)
{
    if (string.IsNullOrEmpty(url))
        throw new ArgumentNullException(nameof(url));
    using var client = new HttpClient();
    var response = await client.GetAsync(url);

    response.EnsureSuccessStatusCode();

    return await response.Content.ReadAsStringAsync();
}
```

--- Я вот еще думаю надо ли мне тут указывать try catch finally

---

## Кейс 3

**Контекст:** метод читает текстовый файл по пути и возвращает его содержимое; при ошибке должен вернуть пустую строку.

```csharp
public string ReadConfig(string path)
{
    try
    {
        var stream = File.OpenRead(path);
        var reader = new StreamReader(stream);
        var content = reader.ReadToEnd();
        return content;
    }
    catch (Exception)
    {
        return string.Empty;
    }
}
```

---

**Твой разбор (заполни ниже):**

**Риски:**


**Возможные баги:**


**Предлагаемые улучшения (код или список правок):**

public string ReadConfig(string path)
{
    if(string.NullOrEmpty(path))
        throw new ArgumentNullException();
    if(!System.IO.File.Exists(path))
        throw new ArgumentException();

    try
    {
        using var stream = File.OpenRead(path);
        using var reader = new StreamReader(stream);
        var content = reader.ReadToEnd();
        return content;
    }
    catch (Exception ex)
    {
        // Необходимо как-то логировать
        return string.Empty;
    }
}

---

## Кейс 4

**Контекст:** сервис создания заказа — сохраняет заказ в БД и отправляет письмо подтверждения.

```csharp
public class OrderService
{
    public void CreateOrder(Order order)
    {
        var repository = new SqlOrderRepository();
        repository.Save(order);

        var emailService = new SmtpEmailService();
        emailService.SendConfirmation(order.CustomerEmail, order.Id);
    }
}
```

---

**Твой разбор (заполни ниже):**

**Риски:**


**Возможные баги:**


**Предлагаемые улучшения (код или список правок):**
public class OrderService
{
    // Метод ничего не возвращает, хотя бы bool
    public void CreateOrder(Order order)
    {
        // Проверка на null Order(добавить)
        using var repository = new SqlOrderRepository();
        repository.Save(order);
        // Не знаю использует ли данный сервис IDisposeble
        var emailService = new SmtpEmailService();
        emailService.SendConfirmation(order.CustomerEmail, order.Id);
    }
}
```

---

## Кейс 5

**Контекст:** метод собирает из списка строк одну строку — полное имя через пробел (например, для отчёта).

```csharp
public string BuildFullName(IEnumerable<string> nameParts)
{
    string result = null;
    foreach (var part in nameParts)
        result += part + " ";
    return result?.Trim();
}
```

---

**Твой разбор (заполни ниже):**

**Риски:**


**Возможные баги:**


**Предлагаемые улучшения (код или список правок):**

public string BuildFullName(IEnumerable<string> nameParts)
{
    if(nameParts==null)
        return string.Empty;
    return string.Join(" ",nameParts);
}
```

---

## Кейс 6

**Контекст:** кнопка в UI по нажатию отправляет данные на сервер. Обработчик события.

```csharp
private async void OnSaveButtonClicked(object sender, EventArgs e)
{
    var data = _textBox.Text;
    var response = await _httpClient.PostAsync("/api/save", new StringContent(data));
    response.EnsureSuccessStatusCode();
}
```

---

**Твой разбор (заполни ниже):**

**Риски:**


**Возможные баги:**


**Предлагаемые улучшения (код или список правок):**


