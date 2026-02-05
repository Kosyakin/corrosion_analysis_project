namespace KafkaExample.Messages;

/// <summary>
/// Модель команды для межсервисной передачи через Kafka.
/// Сериализуется в JSON и кладётся в тело сообщения топика.
/// </summary>
public class ServiceCommand
{
    public string CommandId { get; set; } = Guid.NewGuid().ToString("N");
    public string Type { get; set; } = string.Empty;
    public string Payload { get; set; } = string.Empty;
    public DateTime CreatedAtUtc { get; set; } = DateTime.UtcNow;
}
