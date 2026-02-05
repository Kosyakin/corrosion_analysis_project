using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.DependencyInjection;
using EntityFrameworkExample.Data;
using EntityFrameworkExample.Models;

// --- Инициализация: регистрация DbContext в DI ---
var services = new ServiceCollection();

// Вариант 1: InMemory (для тестов/демо, без реальной БД)
services.AddDbContext<AppDbContext>(options =>
    options.UseInMemoryDatabase("DemoDb"));

// Вариант 2: SQL Server (раскомментировать при наличии строки подключения)
// services.AddDbContext<AppDbContext>(options =>
//     options.UseSqlServer("Server=.;Database=DemoDb;Trusted_Connection=True;TrustServerCertificate=True;"));

var provider = services.BuildServiceProvider();

// --- Использование: создаём БД (для InMemory), добавляем данные, читаем ---
using (var scope = provider.CreateScope())
{
    var db = scope.ServiceProvider.GetRequiredService<AppDbContext>();

    // Создание БД (для InMemory — сразу готова; для реальной БД — миграции или EnsureCreated)
    await db.Database.EnsureCreatedAsync();

    // Добавление
    db.Products.Add(new Product { Name = "Книга", Price = 500 });
    db.Products.Add(new Product { Name = "Стол", Price = 3000 });
    await db.SaveChangesAsync();

    // Чтение: запрос выполняется при перечислении (ToListAsync)
    var list = await db.Products
        .Where(p => p.Price > 100)
        .OrderBy(p => p.Name)
        .ToListAsync();

    Console.WriteLine($"Найдено записей: {list.Count}");
    foreach (var p in list)
        Console.WriteLine($"  {p.Id}: {p.Name}, {p.Price}");

    // Чтение без отслеживания (AsNoTracking) — только для чтения
    var cheap = await db.Products
        .AsNoTracking()
        .Where(p => p.Price < 1000)
        .FirstOrDefaultAsync();
    if (cheap != null)
        Console.WriteLine($"Первый дешёвый: {cheap.Name}");
}
