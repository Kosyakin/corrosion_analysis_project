// ============================================================
// ПРАКТИКА CODE REVIEW — асинхронность и потоки (Example 1)
// Задача: найди недостатки, deadlock'и, race condition и прочие проблемы.
// Подумай, что здесь может сломаться и как это улучшить.
// ============================================================

using System;
using System.Collections.Generic;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;

public class Example1
{
    private readonly object _lock = new object();
    private readonly HttpClient _httpClient = new HttpClient(); // Мне не нравится объявление клиента без Using  и т.п. это норм или нет?
    private int _counter;
    private List<string> _cache = new List<string>();

    // ----- Метод 1 -----
    public async Task<string> LoadDataAsync(string url)
    {
        lock (_lock)
        {
            var cached = _cache.Find(x => x.StartsWith(url));
            if (cached != null)
                return cached;
        } // Возможно лучше использовать потокобезопасную список для данной задачи, но я не знаю как она точно работает, если подскажешь буду рад.

        var response = await _httpClient.GetStringAsync(url);
        
        lock (_lock)
        {
            _cache.Add(response);
        }

        return response;
    }
    // В кэш кладётся тело ответа (response), а поиск идёт по условию x.StartsWith(url) — то есть ищут элементы, начинающиеся с URL.
    // Нужно разделить ключ и значение: например, Dictionary<string, string> с URL как ключом и response как значением, либо хранить в кэше пары вида "url:response" и искать по key + ":".

    // Race condition: если два потока одновременно вызовут LoadDataAsync с одним и тем же URL и в кэше его ещё нет, оба пройдут проверку, оба сделают HTTP-запрос и оба добавят ответ в кэш. Лишний запрос и дублирование в кэше.
    // Чтобы этого избежать, нужна дополнительная синхронизация, например, SemaphoreSlim на один URL или AsyncLock/паттерн с задачей для каждого URL.



    // ----- Метод 2 -----
    public async void ProcessItemsAsync(IEnumerable<string> items) // Ничего не возвращает, может привести 
    { // Нет проверки на null, что может привести к ошибке
        foreach (var item in items)
        {
            var result = await ProcessItemAsync(item);
            Console.WriteLine(result);
        }
    }

    private Task<string> ProcessItemAsync(string item)
    {// Нет проверки на null, что может привести к ошибке
        return Task.Run(() =>
        {
            Thread.Sleep(100);// Лучше использовать Task.Delay()
            return item.ToUpper();
        });
    }// Да и в целом лучше переписать вот так:
    private async Task<string> ProcessItemAsync(string item)
    {// Нет проверки на null, что может привести к ошибке
        Task.Delay(100);// Лучше использовать Task.Delay()
        return item.ToUpper();
    }// Да и в целом лучше переписать вот так:

    // ----- Метод 3 -----
    public int IncrementCounter()
    {
        _counter++;
        return _counter;
    }

    public async Task DoWorkAsync()
    {
        var tasks = new List<Task>();
        for (int i = 0; i < 100; i++)
        {
            tasks.Add(Task.Run(async () =>
            {
                var value = IncrementCounter();
                await Task.Delay(10);
                Console.WriteLine(value);
            }));
        }
        await Task.WhenAll(tasks);
    }

    // ----- Метод 4 -----
    public string GetDataSynchronously(string url)
    {
        var task = _httpClient.GetStringAsync(url);
        return task.Result;
    }

    // ----- Метод 5 -----
    public async Task SaveToCacheAsync(string key, string value)
    {
        await Task.Delay(100);
        lock (_lock)
        {
            _cache.Add($"{key}:{value}");
        }
    }

    public async Task<string> GetFromCacheAsync(string key)
    {
        lock (_lock)
        {
            var found = _cache.Find(x => x.StartsWith(key + ":"));
            if (found != null)
                return found;
        }

        await Task.Delay(50);
        return null;
    }
}
