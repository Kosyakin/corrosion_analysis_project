# SQL — шпаргалка: JOIN, подзапросы, оконные функции

Краткие примеры для подготовки к собеседованию (в т.ч. Т-Банк).

---

## 1. JOIN

### INNER JOIN — только совпадающие строки

```sql
SELECT o.Id, o.Amount, c.Name
FROM Orders o
INNER JOIN Customers c ON o.CustomerId = c.Id;
```

### LEFT JOIN — все из левой таблицы + совпадения справа (NULL, если нет)

```sql
SELECT c.Name, o.Id AS OrderId
FROM Customers c
LEFT JOIN Orders o ON c.Id = o.CustomerId;
```

### RIGHT JOIN — все из правой + совпадения слева

```sql
SELECT c.Name, o.Id
FROM Orders o
RIGHT JOIN Customers c ON o.CustomerId = c.Id;
```

### FULL OUTER JOIN — все из обеих (NULL где нет пары)

```sql
SELECT c.Name, o.Amount
FROM Customers c
FULL OUTER JOIN Orders o ON c.Id = o.CustomerId;
```

### CROSS JOIN — декартово произведение (каждая строка с каждой)

```sql
SELECT c.Name, p.ProductName
FROM Customers c
CROSS JOIN Products p;
```

### Несколько JOIN в одном запросе

```sql
SELECT o.Id, c.Name, p.ProductName
FROM Orders o
INNER JOIN Customers c ON o.CustomerId = c.Id
INNER JOIN OrderItems oi ON oi.OrderId = o.Id
INNER JOIN Products p ON oi.ProductId = p.Id;
```

### SELF JOIN — таблица с самой собой (например, сотрудник → руководитель)

```sql
SELECT e.Name AS Employee, m.Name AS Manager
FROM Employees e
LEFT JOIN Employees m ON e.ManagerId = m.Id;
```

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

### В SELECT (скалярный подзапрос)

```sql
SELECT 
    c.Name,
    (SELECT COUNT(*) FROM Orders o WHERE o.CustomerId = c.Id) AS OrderCount
FROM Customers c;
```

### В FROM (производная таблица)

```sql
SELECT sub.City, AVG(sub.Total) AS AvgTotal
FROM (
    SELECT c.City, SUM(o.Amount) AS Total
    FROM Customers c
    INNER JOIN Orders o ON o.CustomerId = c.Id
    GROUP BY c.City
) sub
GROUP BY sub.City;
```

### Сравнение с агрегатом (ALL, ANY, SOME)

```sql
-- Заказы больше, чем ВСЕ заказы клиента 1
SELECT * FROM Orders
WHERE Amount > ALL (SELECT Amount FROM Orders WHERE CustomerId = 1);

-- Заказы больше, чем ЛЮБОЙ заказ клиента 1
SELECT * FROM Orders
WHERE Amount > ANY (SELECT Amount FROM Orders WHERE CustomerId = 1);
```

### CTE (WITH) — подзапрос с именем, удобно для сложной логики

```sql
WITH TopCustomers AS (
    SELECT CustomerId, SUM(Amount) AS Total
    FROM Orders
    GROUP BY CustomerId
    HAVING SUM(Amount) > 10000
)
SELECT c.Name, tc.Total
FROM Customers c
INNER JOIN TopCustomers tc ON c.Id = tc.CustomerId;
```

### Несколько CTE

```sql
WITH 
Orders2024 AS (
    SELECT * FROM Orders WHERE YEAR(CreatedAt) = 2024
),
ByCustomer AS (
    SELECT CustomerId, COUNT(*) AS Cnt FROM Orders2024 GROUP BY CustomerId
)
SELECT c.Name, bc.Cnt
FROM Customers c
INNER JOIN ByCustomer bc ON c.Id = bc.CustomerId;
```

---

## 3. Оконные функции (window functions)

Синтаксис: `Функция() OVER (PARTITION BY ... ORDER BY ...)`  
Строки не схлопываются (в отличие от GROUP BY), к каждой строке добавляется значение.

### ROW_NUMBER, RANK, DENSE_RANK

```sql
SELECT 
    Name,
    Salary,
    DepartmentId,
    ROW_NUMBER() OVER (PARTITION BY DepartmentId ORDER BY Salary DESC) AS RowNum,
    RANK()       OVER (PARTITION BY DepartmentId ORDER BY Salary DESC) AS Rank,
    DENSE_RANK() OVER (PARTITION BY DepartmentId ORDER BY Salary DESC) AS DenseRank
FROM Employees;
```

- **ROW_NUMBER** — уникальный номер 1, 2, 3… внутри партиции.  
- **RANK** — при равенстве один ранг, следующий «прыгает» (1, 2, 2, 4).  
- **DENSE_RANK** — при равенстве один ранг, следующий идёт подряд (1, 2, 2, 3).

### Топ-N по группе (например, топ-3 по отделу)

```sql
WITH Ranked AS (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY DepartmentId ORDER BY Salary DESC) AS rn
    FROM Employees
)
SELECT * FROM Ranked WHERE rn <= 3;
```

### Агрегаты как оконные (без GROUP BY)

```sql
SELECT 
    OrderId,
    Amount,
    SUM(Amount)   OVER (PARTITION BY CustomerId) AS CustomerTotal,
    AVG(Amount)   OVER (PARTITION BY CustomerId) AS CustomerAvg,
    SUM(Amount)   OVER (ORDER BY CreatedAt)     AS RunningTotal
FROM Orders;
```

### LAG / LEAD — предыдущая и следующая строка

```sql
SELECT 
    CreatedAt,
    Amount,
    LAG(Amount)  OVER (ORDER BY CreatedAt) AS PrevAmount,
    LEAD(Amount) OVER (ORDER BY CreatedAt) AS NextAmount
FROM Orders;
```

### FIRST_VALUE / LAST_VALUE

```sql
SELECT 
    DepartmentId,
    Name,
    Salary,
    FIRST_VALUE(Name) OVER (PARTITION BY DepartmentId ORDER BY Salary DESC) AS TopEarner
FROM Employees;
```

### Рамка окна (ROWS / RANGE)

```sql
-- Скользящее среднее по последним 3 строкам
SELECT 
    Date,
    Value,
    AVG(Value) OVER (ORDER BY Date ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS MovingAvg3
FROM DailySales;
```

---

## 4. Быстрая сводка

| Тема | Ключевые слова |
|------|----------------|
| JOIN | INNER, LEFT, RIGHT, FULL OUTER, CROSS, ON |
| Подзапросы | (SELECT ...), IN, EXISTS, WITH ... AS (CTE) |
| Оконные | OVER, PARTITION BY, ORDER BY, ROW_NUMBER, RANK, DENSE_RANK, LAG, LEAD, SUM/AVG OVER |

На собеседовании часто просят: «напиши запрос с JOIN», «как взять топ-N в каждой группе», «чем RANK отличается от ROW_NUMBER», «что такое CTE». Этой шпаргалки достаточно для повторения.
