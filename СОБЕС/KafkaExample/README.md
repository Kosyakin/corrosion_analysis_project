# Межсервисное взаимодействие через Kafka — как устроено

## Где настраиваются очереди (топики)

В Kafka роль "очередей" играют **топики (topics)**. Они **не настраиваются в коде контроллера или сервисов** — только **используются по имени**.

- **Конфигурация (имена топиков и брокеров)** лежит в:
  - **`appsettings.json`** или **`appsettings.Kafka.json`** → секция **`Kafka`**;
  - класс **`KafkaOptions.cs`** — маппинг этой секции (BootstrapServers, TopicCommands, TopicEvents, ConsumerGroupId и т.д.).

- **Создание самих топиков** делается на стороне Kafka (админка, `kafka-topics.sh` или Admin API), например:
  ```bash
  kafka-topics --create --topic service.commands --partitions 3 --replication-factor 1 --bootstrap-server localhost:9092
  kafka-topics --create --topic service.events  --partitions 3 --replication-factor 1 --bootstrap-server localhost:9092
  ```

Итого: **очереди (топики) настраиваются** в конфиге приложения (имена + адрес брокера) и в самом Kafka (создание топиков с партициями и репликацией). В коде только чтение этих имён из `KafkaOptions`.

---

## Структура проекта

| Файл | Назначение |
|------|------------|
| **KafkaOptions.cs** | Настройки: адреса брокеров, имена топиков (очередей), Consumer Group, AutoOffsetReset. Секция конфига: `Kafka`. |
| **appsettings.Kafka.json** | Пример конфига с секцией `Kafka` (BootstrapServers, TopicCommands, TopicEvents, ConsumerGroupId). |
| **KafkaProducerService.cs** | Продюсер: отправка сообщений в топики. Использует имена топиков из `KafkaOptions`. |
| **KafkaConsumerService.cs** | Фоновый консьюмер: подписка на топик из `KafkaOptions.TopicEvents`, чтение сообщений из очереди, обработка. |
| **KafkaController.cs** | HTTP API: приём запросов и публикация команд в Kafka (межсервисный уровень). |
| **Messages/ServiceCommand.cs** | Модель сообщения (команды) для сериализации в JSON и передачи через топик. |
| **ServiceCollectionExtensions.cs** | Регистрация Kafka (Options, Producer, Consumer) в DI. |
| **Program.cs** | Подключение конфига Kafka и вызов `AddKafka()`. |

---

## Как это работает

1. **Запрос приходит по HTTP** в `KafkaController` (например `POST /api/kafka/commands`).
2. **Контроллер** формирует `ServiceCommand` и вызывает **`KafkaProducerService.PublishCommandAsync`**.
3. **Продюсер** сериализует команду в JSON и отправляет сообщение в топик **`TopicCommands`** (имя из `KafkaOptions` → по сути очередь `service.commands`).
4. **Другой сервис** (или этот же) подписан на этот топик через консьюмера. **`KafkaConsumerService`** в этом примере подписан на **`TopicEvents`** — для входящих событий; для команд от этого сервиса другой сервис будет подписан на `service.commands`.
5. **Консьюмер** в цикле читает сообщения из топика (очереди), десериализует в `ServiceCommand` и обрабатывает в `HandleCommandAsync`.

Поток данных: **HTTP → Controller → Producer → топик Kafka (очередь) → Consumer другого/этого сервиса → обработка**.

---

## Запуск

1. Поднять Kafka (например Docker: `docker run -d --name kafka -p 9092:9092 apache/kafka`).
2. Создать топики (см. команды выше).
3. Указать в `appsettings.Kafka.json` правильный `BootstrapServers`.
4. Запустить приложение и вызывать `POST /api/kafka/commands` с телом `{ "type": "DoSomething", "payload": "..." }`.

Пакет: **Confluent.Kafka** (уже в .csproj).
