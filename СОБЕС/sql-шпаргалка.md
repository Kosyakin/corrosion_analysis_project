

### SELF JOIN — таблица с самой собой (например, сотрудник → руководитель)

```sql
SELECT e.Name AS Employee, m.Name AS Manager
FROM Employees e
LEFT JOIN Employees m ON e.ManagerId = m.Id;
```

### Hash Join (алгоритм выполнения JOIN)

Одна таблица (обычно меньшая) полностью читается и строится **in-memory хеш-таблица** по ключу JOIN. Вторая таблица сканируется, для каждой строки по ключу ищется совпадение в хеше — O(1). **Когда уместен:** большие объёмы, нет индексов по ключу JOIN, нужна равноправная обработка обеих таблиц. Часто выбирают для больших JOIN’ов вместо Nested Loop. Может требовать много памяти (build-таблица должна помещаться в хеш).

---

## 2. Подзапросы (subqueries)

### В WHERE (скаляр или список)

```sql
-- Один результат (скаляр)
SELECT * FROM Orders
WHERE CustomerId = (SELECT Id FROM Customers WHERE Email = 'a@b.com');

-- Список (IN)
SELECT * FROM Orders
WHERE CustomerId IN (SELECT Id FROM Customers WHERE City = 'Moscow');

-- Существование (EXISTS) — часто быстрее, т.к. не материализует список
SELECT * FROM Customers c
WHERE EXISTS (
    SELECT 1 FROM Orders o WHERE o.CustomerId = c.Id
);
```

### Сравнение с агрегатом (ALL, ANY, SOME)

```sql
-- Заказы больше, чем ВСЕ заказы клиента 1
SELECT * FROM Orders
WHERE Amount > ALL (SELECT Amount FROM Orders WHERE CustomerId = 1);

-- Заказы больше, чем ЛЮБОЙ заказ клиента 1
SELECT * FROM Orders
WHERE Amount > ANY (SELECT Amount FROM Orders WHERE CustomerId = 1);
---

## 3. Оконные функции

**Шаблон:** `Функция() OVER (PARTITION BY a, b ORDER BY c [ROWS BETWEEN ...])`

- Рамка: `UNBOUNDED PRECEDING` / `N PRECEDING` / `CURRENT ROW` / `N FOLLOWING` / `UNBOUNDED FOLLOWING`
- Без текущей (сумма «до этой строки»): `ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING`
- С текущей (нарастающий итог): `ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`
- Скользящее 3 строки: `ROWS BETWEEN 2 PRECEDING AND CURRENT ROW`

**Функции:** ROW_NUMBER, RANK, DENSE_RANK | LAG/LEAD | SUM/AVG OVER | NTILE(N), NTH_VALUE, PERCENT_RANK, CUME_DIST

---

## 4. CASE

**В строчку:** `CASE [столбец] WHEN значение THEN ... WHEN ... ELSE ... END` или `CASE WHEN условие THEN ... ELSE ... END`

- В SELECT: как выше. В агрегате: `SUM(CASE WHEN Status='Paid' THEN 1 ELSE 0 END)`. В ORDER BY: `ORDER BY CASE WHEN Priority='High' THEN 0 ELSE 1 END`.

---

## 5. Работа с датой

*Синтаксис зависит от СУБД (ниже — типичные имена).*

| Действие | SQL Server | PostgreSQL | Пример |
|----------|------------|------------|--------|
| Текущие дата/время | `GETDATE()`, `SYSDATETIME()` | `CURRENT_DATE`, `NOW()` | `SELECT GETDATE()` |
| Год/месяц/день | `YEAR(d)`, `MONTH(d)`, `DAY(d)` | то же | `YEAR(CreatedAt)` |
| Разница дней | `DATEDIFF(day, d1, d2)` | `d2::date - d1::date` или `DATE_PART('day', d2 - d1)` | `DATEDIFF(day, Start, End)` |
| Добавить интервал | `DATEADD(day, N, d)` | `d + INTERVAL 'N days'` | `DATEADD(month, 1, d)` |
| Начало дня | `CAST(d AS DATE)` | `date_trunc('day', d)::date` | убрать время |
| Начало месяца | — | `date_trunc('month', d)` | SS: `DATEFROMPARTS(YEAR(d), MONTH(d), 1)` |
| Формат в строку | `FORMAT(d, 'yyyy-MM-dd')` | `TO_CHAR(d, 'YYYY-MM-DD')` | вывод даты |

**В WHERE:** `WHERE CreatedAt >= '2024-01-01' AND CreatedAt < '2025-01-01'` (диапазон года). Для «только этот день»: `CAST(CreatedAt AS DATE) = '2024-06-15'` (или эквивалент в своей СУБД).

