# Entity Framework Example

- **Теория и типичные вопросы на собеседовании:** [EntityFramework-основы-и-собеседование.md](EntityFramework-основы-и-собеседование.md)
- **Пример кода:** инициализация `DbContext` и использование — в `Program.cs`, контекст в `Data/AppDbContext.cs`, сущность в `Models/Product.cs`.

## Запуск

```bash
cd EntityFrameworkExample
dotnet run
```

Используется провайдер **InMemory** (реальная БД не нужна). Для SQL Server раскомментируйте блок в `Program.cs` и настройте строку подключения.
