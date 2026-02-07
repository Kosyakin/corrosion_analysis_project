# Внедрение зависимостей (Dependency Injection) в .NET — кратко

## Что такое DI

**Что это:** Паттерн проектирования, при котором объект не создаёт свои зависимости сам, а получает их извне (обычно через конструктор). В .NET есть встроенный DI-контейнер `Microsoft.Extensions.DependencyInjection`.

**Зачем:** Слабая связанность (loose coupling), легко подменять реализации, удобно тестировать (можно подставить мок).

---

## Три времени жизни (Service Lifetime)

### Transient

**Что это:** Новый экземпляр создаётся при **каждом** запросе из контейнера.

**Когда использовать:** Лёгкие, stateless-сервисы без общего состояния.

```csharp
builder.Services.AddTransient<IEmailSender, SmtpEmailSender>();
```

### Scoped

**Что это:** Один экземпляр на **область действия** (scope). В ASP.NET Core scope = один HTTP-запрос.

**Когда использовать:** Сервисы, которые должны быть общими в рамках одного запроса (например, `DbContext`).

```csharp
builder.Services.AddScoped<IOrderRepository, OrderRepository>();
```

### Singleton

**Что это:** Один экземпляр на **всё время жизни приложения**. Создаётся при первом обращении.

**Когда использовать:** Потокобезопасные сервисы с дорогой инициализацией (кэш, настройки, HttpClient-фабрика).

```csharp
builder.Services.AddSingleton<ICacheService, MemoryCacheService>();
```

---

## Способы внедрения

### Через конструктор (основной и рекомендуемый)

```csharp
public class OrderService
{
    private readonly IOrderRepository _repo;

    public OrderService(IOrderRepository repo)
    {
        _repo = repo;
    }
}
```

### Через метод (Method Injection)

```csharp
public void Process([FromServices] ILogger<OrderController> logger)
{
    logger.LogInformation("Processing...");
}
```

### Через свойство / Service Locator (антипаттерн, избегать)

```csharp
// Не рекомендуется — скрывает зависимости
var service = serviceProvider.GetRequiredService<IMyService>();
```

---

## Регистрация сервисов — Program.cs

```csharp
var builder = WebApplication.CreateBuilder(args);

// Регистрация зависимостей
builder.Services.AddScoped<IOrderRepository, OrderRepository>();
builder.Services.AddTransient<IEmailSender, SmtpEmailSender>();
builder.Services.AddSingleton<ICacheService, MemoryCacheService>();

var app = builder.Build();
```

---

## Частые ловушки

| Ловушка | Описание |
|---|---|
| **Captive Dependency** | Singleton держит ссылку на Scoped/Transient — scoped-объект живёт дольше, чем должен. Правило: зависимость должна жить **не меньше**, чем потребитель. |
| **Service Locator** | Вместо явного внедрения — ручной вызов `GetService<T>()`. Скрывает зависимости, усложняет тестирование. |
| **Слишком много параметров** | Конструктор с 7+ зависимостями — признак нарушения SRP. Нужно разбить класс на несколько. |

---

## Типичные вопросы на собесе

**В: Чем Scoped отличается от Transient?**
О: Transient — новый экземпляр при каждом resolve. Scoped — один экземпляр в рамках scope (HTTP-запроса в ASP.NET Core). Два вызова `GetService<T>()` внутри одного запроса вернут **один и тот же** Scoped-объект, но **разные** Transient-объекты.

**В: Можно ли внедрять Scoped-сервис в Singleton?**
О: Нельзя напрямую — это Captive Dependency. Scoped-объект «застрянет» в Singleton навсегда. Решение: внедрять `IServiceScopeFactory` и создавать scope вручную.

**В: Как зарегистрировать несколько реализаций одного интерфейса?**
О: Можно зарегистрировать несколько и внедрить `IEnumerable<IMyService>` — контейнер вернёт все. Если нужна одна — последняя регистрация побеждает.

**В: Что делает `TryAdd*`?**
О: Регистрирует сервис только если он **ещё не зарегистрирован**. Полезно в библиотеках, чтобы не перезаписывать пользовательскую регистрацию.
