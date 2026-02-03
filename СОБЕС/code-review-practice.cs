// ============================================================
// ПРАКТИКА CODE REVIEW — на собеседовании дадут что-то похожее
// Разбери каждый метод: найди недостатки и предложи улучшения
// Ответы и разбор — в code-review-prep.md
// ============================================================

// --------------- Фрагмент 1: работа с БД и строками ---------------
public class UserRepository
{
    private SqlConnection _connection;

    public User GetUserByEmail(string email)
    {
        var sql = "SELECT * FROM Users WHERE Email = '" + email + "'";
        var cmd = new SqlCommand(sql, _connection);
        var reader = cmd.ExecuteReader();
        var user = new User();
        if (reader.Read())
        {
            user.Id = reader.GetInt32(0);
            user.Email = reader.GetString(1);
        }
        return user;
    }
}

// --------------- Фрагмент 2: асинхронность ---------------
public class ApiClient
{
    public string FetchData(string url)
    {
        using var client = new HttpClient();
        var task = client.GetStringAsync(url);
        return task.Result;
    }
}

// --------------- Фрагмент 3: исключения и ресурсы ---------------
public void ProcessFile(string path)
{
    try
    {
        var stream = File.OpenRead(path);
        var text = new StreamReader(stream).ReadToEnd();
        DoSomethingWith(text);
    }
    catch (Exception)
    {
        // продолжим работу
    }
}

// --------------- Фрагмент 4: null и коллекции ---------------
public decimal CalculateTotal(List<OrderItem> items)
{
    decimal total = 0;
    foreach (var item in items)
        total += item.Price * item.Quantity;
    return total;
}

// --------------- Фрагмент 5: производительность ---------------
public string GetFullName(List<string> names)
{
    string result = null;
    foreach (var name in names)
        result += name + " ";
    return result?.Trim();
}
