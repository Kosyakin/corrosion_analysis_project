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


---

## Кейс 7 — развёрнутый (50+ строк, найти от 5 типов ошибок)

**Контекст:** сервис получает пользователя по email, дергает внешний API за доп. данными и пишет результат в лог-файл.

**Задача:** найти **не менее 5 разных по смыслу** проблем (безопасность, ресурсы, асинхронность, null, исключения, производительность, дизайн и т.д.).

```csharp
public class UserSyncService
{
    private string _connectionString = "Server=.;Database=AppDb;";
    private string _logPath = "C:\\Logs\\sync.log";

    public void SyncUserByEmail(string email)
    {
        var connection = new SqlConnection(_connectionString);
        connection.Open();

        var query = "SELECT * FROM Users WHERE Email = '" + email + "'";
        var cmd = new SqlCommand(query, connection);
        var reader = cmd.ExecuteReader();

        int userId = 0;
        string userName = "";
        if (reader.Read())
        {
            userId = reader.GetInt32(0);
            userName = reader.GetString(1);
        }
        connection.Close();

        var client = new HttpClient();
        var url = "https://api.external.com/user/" + userId;
        var json = client.GetStringAsync(url).Result;
        client.Dispose();

        var data = JsonSerializer.Deserialize<ExternalData>(json);
        var line = DateTime.Now + " | " + userId + " | " + data.Status + " | " + data.Score;

        try
        {
            var stream = File.Open(_logPath, FileMode.Append);
            var writer = new StreamWriter(stream);
            writer.WriteLine(line);
            writer.Close();
        }
        catch (Exception)
        {
        }

        if (data.Score > 100)
        {
            var updateCmd = new SqlCommand("UPDATE Users SET Flag = 1 WHERE Id = " + userId, connection);
            connection.Open();
            updateCmd.ExecuteNonQuery();
        }
    }
}
```

---

**Твой разбор — перечисли по категориям (найди от 5 направлений):**

**1. Безопасность:**


**2. Ресурсы / IDisposable:**


**3. Асинхронность:**


**4. Null / граничные случаи:**


**5. Исключения:**


**6. Прочее (производительность, дизайн, дублирование и т.д.):**


**Кратко — что исправил бы в первую очередь (топ-3):**

public async Task SyncUserByEmailAsync(string email)
{
    using var connection = new SqlConnection(_connectionString);
    connection.Open();

    var query = "SELECT * FROM Users WHERE Email = @email";

    var cmd = new SqlCommand(query, connection);
    cmd.Parameters.AddWithValue("@email", email);
    using var reader = cmd.ExecuteReader(){
    int userId = 0;
    string userName = "";
    if (reader.Read())
    {
        userId = reader.GetInt32(0);
        userName = reader.GetString(1);
    }
  }
    using var client = new HttpClient();
    
    var url = $"https://api.external.com/user/{userId}" ;
    var json = await client.GetStringAsync(url);
    var data = await JsonSerializer.DeserializeAsync<ExternalData>(json);
    var line = $"{DateTime.Now} | {userId} | {data.Status} |  {data.Score}";
    


    try
    {
        using (var stream = File.Open(_logPath, FileMode.Append))
        {
            var writer = new StreamWriter(stream);
            await writer.WriteLineAsync(line);
        }
    }
    catch (Exception Ex)
    {
        // Добавление лога 
    }

    if (data.Score > 100)
    {
        var updateCmd = new SqlCommand("UPDATE Users SET Flag = 1 WHERE Id = @userId", connection);
        updateCmd.Parameters.AddWithValue("@userId", userId);
        await updateCmd.ExecuteNonQueryAsync();
    }
}
```

---

## Кейс 8 — API / контроллер (другая направленность)

**Контекст:** веб-API принимает запрос на создание заказа, валидирует, сохраняет и отправляет в очередь уведомлений. Нужно найти от 5 типов проблем.

```csharp
[ApiController]
[Route("api/orders")]
public class OrdersController : ControllerBase
{
    private readonly string _connStr = ConfigurationManager.ConnectionStrings["Default"].ConnectionString;

    [HttpPost]
    public IActionResult CreateOrder([FromBody] CreateOrderRequest request)
    {
        if (request == null)
            return BadRequest();        if (request.Amount <= 0 || request.CustomerId == Guid.Empty)
            return BadRequest("Invalid data");

        var order = new Order();
        order.Id = Guid.NewGuid();
        order.CustomerId = request.CustomerId;
        order.Amount = request.Amount;
        order.CreatedAt = DateTime.Now;

        using (var conn = new SqlConnection(_connStr))
        {
            conn.Open();
            var sql = "INSERT INTO Orders (Id, CustomerId, Amount, CreatedAt) VALUES ('" +
                order.Id + "', '" + order.CustomerId + "', " + order.Amount + ", '" + order.CreatedAt + "')";
            var cmd = new SqlCommand(sql, conn);
            cmd.ExecuteNonQuery();
        }

        var client = new HttpClient();
        client.Timeout = TimeSpan.FromSeconds(3);
        var content = new StringContent(JsonSerializer.Serialize(new { OrderId = order.Id }), Encoding.UTF8, "application/json");
        var response = client.PostAsync("http://internal-service/notify", content).Result;
        if (!response.IsSuccessStatusCode)
        {
            // откатываем заказ при ошибке уведомления
            using (var conn = new SqlConnection(_connStr))
            {
                conn.Open();
                new SqlCommand("DELETE FROM Orders WHERE Id = '" + order.Id + "'", conn).ExecuteNonQuery();
            }
        }

        return Ok(new { order.Id });
    }
}
```

**Твой разбор (заполни завтра):**

**1. Безопасность:**  
**2. Ресурсы / IDisposable:**  
**3. Асинхронность:**  
**4. Null / граничные случаи:**  
**5. Исключения / транзакции:**  
**6. Прочее:**  
**Топ-3 исправления:**

---

## Кейс 9 — фоновая обработка / worker

**Контекст:** воркер раз в минуту забирает задачи из таблицы, обрабатывает и обновляет статус. Найти от 5 типов проблем.

```csharp
public class TaskProcessorService
{
    private SqlConnection _connection;
    private bool _running = true;

    public void Start()
    {
        _connection = new SqlConnection("Server=.;Database=Tasks;");
        _connection.Open();

        while (_running)
        {
            var cmd = new SqlCommand("SELECT * FROM PendingTasks WHERE Status = 0", _connection);
            var reader = cmd.ExecuteReader();

            while (reader.Read())
            {
                var taskId = reader.GetGuid(0);
                var payload = reader.GetString(1);
                reader.Close();

                try
                {
                    ProcessTask(taskId, payload);
                    MarkCompleted(taskId);
                }
                catch (Exception ex)
                {
                    MarkFailed(taskId, ex.Message);
                }

                reader = cmd.ExecuteReader();
            }

            Thread.Sleep(60000);
        }
    }

    private void MarkCompleted(Guid taskId)
    {
        var cmd = new SqlCommand("UPDATE PendingTasks SET Status = 1 WHERE Id = '" + taskId + "'", _connection);
        cmd.ExecuteNonQuery();
    }

    private void MarkFailed(Guid taskId, string message)
    {
        var cmd = new SqlCommand("UPDATE PendingTasks SET Status = 2, Error = '" + message.Replace("'", "''") + "' WHERE Id = '" + taskId + "'", _connection);
        cmd.ExecuteNonQuery();
    }

    private void ProcessTask(Guid taskId, string payload) { /* ... */ }
}
```

**Твой разбор (заполни завтра):**

**1. Безопасность:**  
**2. Ресурсы / соединение:**  
**3. Reader / циклы:**  
**4. Исключения / повторная обработка:**  
**5. Прочее (потоки, конфиг и т.д.):**  
**Топ-3 исправления:**

---

## Кейс 10 — сервис с кэшем и внешним вызовом

**Контекст:** сервис отдаёт настройки пользователя; при первом запросе грузит из API и кэширует в словаре. Найти от 5 типов проблем.

```csharp
public class UserSettingsService
{
    private Dictionary<Guid, UserSettings> _cache = new Dictionary<Guid, UserSettings>();
    private static readonly HttpClient _httpClient = new HttpClient();

    public UserSettings GetSettings(Guid userId)
    {
        if (_cache.ContainsKey(userId))
            return _cache[userId];

        var url = "https://config-api.company.com/users/" + userId + "/settings";
        var response = _httpClient.GetAsync(url).Result;
        response.EnsureSuccessStatusCode();

        var json = response.Content.ReadAsStringAsync().Result;
        var settings = JsonSerializer.Deserialize<UserSettings>(json);

        _cache[userId] = settings;
        return settings;
    }

    public void InvalidateCache(Guid userId)
    {
        _cache.Remove(userId);
    }

    public void PreloadCache(IEnumerable<Guid> userIds)
    {
        foreach (var id in userIds)
            GetSettings(id);
    }
}
```

**Твой разбор (заполни завтра):**

**1. Потокобезопасность / кэш:**  
**2. Асинхронность:**  
**3. Null / граничные случаи:**  
**4. Ресурсы / конфигурация:**  
**5. Прочее (производительность, дизайн):**  
**Топ-3 исправления:**