# Списки в .NET — виды и примеры

Краткий обзор основных типов коллекций-списков и работа с ними в C#.

---

## 1. List\<T\> — основной динамический список

Упорядоченная коллекция с доступом по индексу. Внутри — массив, при нехватке места размер увеличивается.

```csharp
// Создание
var list = new List<int>();
var list2 = new List<string> { "a", "b", "c" };
var list3 = new List<int>(capacity: 100); // начальная ёмкость

// Добавление
list.Add(1);
list.AddRange(new[] { 2, 3, 4 });
list.Insert(0, 0);           // вставка по индексу

// Доступ
int first = list[0];
int count = list.Count;

// Поиск
int index = list.IndexOf(3);
bool exists = list.Contains(2);
var found = list.Find(x => x > 2);

// Удаление
list.Remove(3);
list.RemoveAt(0);
list.RemoveAll(x => x % 2 == 0);
list.Clear();

// Перебор
foreach (var item in list)
    Console.WriteLine(item);

for (int i = 0; i < list.Count; i++)
    Console.WriteLine(list[i]);
```

**Когда использовать:** нужен именно список с индексами, добавление/удаление в конец или по индексу. Самый частый выбор.

---

## 2. LinkedList\<T\> — двусвязный список

Элементы связаны ссылками (prev/next). Вставка и удаление в любом месте — O(1), если есть ссылка на узел; доступ по индексу — O(n).

```csharp
var linked = new LinkedList<string>();
linked.AddLast("first");
linked.AddLast("second");
linked.AddFirst("zero");
linked.AddAfter(linked.First!, "half");  // после первого

var node = linked.Find("second");
if (node != null)
    linked.AddAfter(node, "between");

// Удаление по значению или по узлу
linked.Remove("zero");
linked.Remove(node);

// Перебор
foreach (var item in linked)
    Console.WriteLine(item);

// По узлам (если нужен доступ к соседям)
for (var n = linked.First; n != null; n = n.Next)
    Console.WriteLine(n.Value);
```

**Когда использовать:** частые вставки/удаления в середине при наличии узла; очередь/стек на узлах (AddFirst/AddLast + RemoveFirst/RemoveLast).

---

## 3. IList\<T\> и IList — интерфейсы

`List<T>` реализует `IList<T>`. Через интерфейс удобно принимать «любой список» в методах.

```csharp
void ProcessItems(IList<int> items)
{
    // Доступ по индексу, Count, Add, Remove, IndexOf и т.д.
    for (int i = 0; i < items.Count; i++)
        items[i] *= 2;
}

ProcessItems(new List<int> { 1, 2, 3 });
ProcessItems(new int[] { 1, 2, 3 });  // массив тоже IList<T>
```

**IList** (без generic) — устаревший, элементы типа `object`. Лучше не использовать в новом коде.

---

## 4. Array (T[]) — массив

Фиксированная длина, создаётся один раз. Самый быстрый доступ по индексу.

```csharp
var arr = new int[] { 1, 2, 3 };
var arr2 = new int[10];

arr[0] = 0;
int len = arr.Length;

// Копирование
var copy = new int[arr.Length];
Array.Copy(arr, copy, arr.Length);

// Сортировка, поиск (статическими методами)
Array.Sort(arr);
int idx = Array.IndexOf(arr, 2);
Array.Reverse(arr);
```

**Когда использовать:** длина известна и не меняется; максимальная производительность и минимум аллокаций.

---

## 5. Collection\<T\> — базовый класс для своих коллекций

Расширяемая коллекция с виртуальными методами (InsertItem, RemoveItem и т.д.). По умолчанию внутри хранит `List<T>`.

```csharp
var col = new Collection<string> { "a", "b" };
col.Add("c");
col.Insert(0, "x");
string item = col[0];
col.RemoveAt(0);
```

**Когда использовать:** когда нужно наследовать коллекцию и переопределять поведение при добавлении/удалении (валидация, уведомления).

---

## 6. ObservableCollection\<T\> — список с уведомлениями об изменениях

Реализует `INotifyCollectionChanged`. Используется в WPF/Uno/MAUI для привязки к UI: при Add/Remove/Replace список сам обновляет экран.

```csharp
var obs = new ObservableCollection<string> { "a", "b" };

obs.CollectionChanged += (sender, e) =>
{
    if (e.NewItems != null)
        Console.WriteLine($"Added: {e.NewItems[0]}");
    if (e.OldItems != null)
        Console.WriteLine($"Removed: {e.OldItems[0]}");
};

obs.Add("c");
obs.RemoveAt(0);
obs[0] = "replaced";
```

**Когда использовать:** привязка списка к списку/таблице в UI (WPF, Xamarin, MAUI).

---

## 7. ReadOnlyCollection\<T\> и AsReadOnly() — только чтение

Обёртка над `IList<T>`, запрещающая изменение. Не копирует данные.

```csharp
var list = new List<int> { 1, 2, 3 };
IReadOnlyList<int> readOnly = list.AsReadOnly();
// или
var ro = new ReadOnlyCollection<int>(list);

// readOnly[0] = 5;  // нельзя
// ro.Add(4);        // нельзя
```

**Когда использовать:** отдать наружу «неизменяемый» вид списка без копирования.

---

## 8. ImmutableList\<T\> — неизменяемый список

Любая «изменение» возвращает новый экземпляр; исходный не меняется. Потокобезопасен для чтения.

```csharp
using System.Collections.Immutable;

var imm = ImmutableList.Create(1, 2, 3);
var imm2 = imm.Add(4);   // новый список, imm не изменился
var imm3 = imm2.Remove(2);

// imm  = [1, 2, 3]
// imm2 = [1, 2, 3, 4]
// imm3 = [1, 3, 4]
```

**Когда использовать:** многопоточность, функциональный стиль, когда нельзя менять список после создания.

---

## 9. IEnumerable\<T\> и LINQ — перечисление и преобразования

Интерфейс «что можно перечислить». Не даёт Count/индекс по умолчанию (только перебор). Основа для LINQ.

```csharp
IEnumerable<int> source = new List<int> { 1, 2, 3, 4, 5 };

var doubled = source.Select(x => x * 2);
var evens = source.Where(x => x % 2 == 0);
var listFromLinq = source.Where(x => x > 2).ToList();
var arrayFromLinq = source.ToArray();

// Один проход — только перебор
foreach (var x in source)
    Console.WriteLine(x);
```

**Когда использовать:** вход/выход методов, когда достаточно «перебрать» или строить цепочки LINQ; материализация в список — `.ToList()` при необходимости.

---

## 10. Сводка: что когда выбирать

| Нужно | Тип |
|-------|-----|
| Обычный список с индексами, добавление/удаление | **List\<T\>** |
| Частые вставки/удаления в середине по узлу | **LinkedList\<T\>** |
| Длина фиксирована, максимум скорости | **T[]** |
| Привязка к UI (WPF, MAUI) | **ObservableCollection\<T\>** |
| Отдать «только чтение» без копирования | **ReadOnlyCollection\<T\>** / **IReadOnlyList\<T\>** |
| Неизменяемый список, многопоточность | **ImmutableList\<T\>** |
| Принять «любой список» в методе | **IList\<T\>** или **IEnumerable\<T\>** |
| Только перебор и LINQ | **IEnumerable\<T\>** |

---

## 11. Частые операции (на примере List\<T\>)

```csharp
var list = new List<int> { 3, 1, 4, 1, 5 };

list.Sort();                    // [1, 1, 3, 4, 5]
list.Reverse();                 // разворот на месте
bool any = list.Any(x => x > 4);
int sum = list.Sum();
var distinct = list.Distinct().ToList();
var sub = list.GetRange(1, 2);  // с индекса 1 взять 2 элемента
list.RemoveRange(0, 2);
```

Этого достаточно, чтобы уверенно говорить про списки на собеседовании по .NET.
