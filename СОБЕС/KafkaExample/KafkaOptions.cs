namespace KafkaExample;

/// <summary>
/// Настройки подключения к Kafka и имена топиков.
/// В Kafka "очереди" — это топики (topics). Здесь задаются их имена и параметры брокера.
/// Секция в appsettings.json: "Kafka".
/// </summary>
public class KafkaOptions
{
    public const string SectionName = "Kafka";

    /// <summary>
    /// Адреса брокеров Kafka (например: "localhost:9092" или "broker1:9092,broker2:9092").
    /// Клиент подключается к любому из них и узнаёт остальные по метаданным.
    /// </summary>
    public string BootstrapServers { get; set; } = "localhost:9092";

    /// <summary>
    /// Имя топика для команд/событий от этого сервиса к другим.
    /// Топик = по сути "очередь" в Kafka (логическая очередь сообщений).
    /// </summary>
    public string TopicCommands { get; set; } = "service.commands";

    /// <summary>
    /// Имя топика для входящих событий от других сервисов (подписка).
    /// </summary>
    public string TopicEvents { get; set; } = "service.events";

    /// <summary>
    /// Идентификатор группы потребителей (Consumer Group).
    /// Все консьюмеры с одним GroupId делят партиции топика между собой (масштабирование).
    /// </summary>
    public string ConsumerGroupId { get; set; } = "my-service-consumer-group";

    /// <summary>
    /// Откуда читать при первом запуске: "earliest" — с начала, "latest" — только новые.
    /// </summary>
    public string AutoOffsetReset { get; set; } = "Earliest";
}
