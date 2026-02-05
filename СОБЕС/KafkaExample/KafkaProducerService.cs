using System.Text.Json;
using Confluent.Kafka;
using KafkaExample.Messages;
using Microsoft.Extensions.Options;

namespace KafkaExample;

/// <summary>
/// Сервис отправки сообщений в топики Kafka (продюсер).
/// Используется контроллером и другими частями приложения для межсервисной коммуникации.
/// Очереди (топики) не настраиваются здесь — только отправка. Настройка топиков — в KafkaOptions.
/// </summary>
public class KafkaProducerService : IDisposable
{
    private readonly IProducer<string, string> _producer;
    private readonly KafkaOptions _options;
    private readonly ILogger<KafkaProducerService> _logger;
    private static readonly JsonSerializerOptions JsonOptions = new() { PropertyNamingPolicy = JsonNamingPolicy.CamelCase };

    public KafkaProducerService(IOptions<KafkaOptions> options, ILogger<KafkaProducerService> logger)
    {
        _options = options.Value;
        _logger = logger;

        var config = new ProducerConfig
        {
            BootstrapServers = _options.BootstrapServers,
            // Гарантия доставки: All = ждём подтверждения от всех реплик (надёжно для межсервисных команд).
            Acks = Acks.All,
            RetryBackoffMs = 100,
            MessageSendMaxRetries = 3,
            EnableIdempotence = true,
        };

        _producer = new ProducerBuilder<string, string>(config).Build();
    }

    /// <summary>
    /// Публикует команду в топик "очереди" service.commands.
    /// Ключ (key) используется для партиционирования: один и тот же key всегда попадёт в одну партицию (порядок сохраняется).
    /// </summary>
    public async Task PublishCommandAsync(ServiceCommand command, CancellationToken cancellationToken = default)
    {
        var topic = _options.TopicCommands;
        var key = command.CommandId;
        var value = JsonSerializer.Serialize(command, JsonOptions);

        var message = new Message<string, string>
        {
            Key = key,
            Value = value,
            Headers = new Headers
            {
                { "source", System.Text.Encoding.UTF8.GetBytes("kafka-example-service") },
                { "command-type", System.Text.Encoding.UTF8.GetBytes(command.Type) },
            },
        };

        try
        {
            var result = await _producer.ProduceAsync(topic, message, cancellationToken);
            _logger.LogInformation(
                "Сообщение отправлено в топик {Topic}, партиция {Partition}, offset {Offset}",
                result.Topic, result.Partition.Value, result.Offset.Value);
        }
        catch (ProduceException<string, string> ex)
        {
            _logger.LogError(ex, "Ошибка отправки в Kafka, топик {Topic}", topic);
            throw;
        }
    }

    /// <summary>
    /// Отправка сырого сообщения в указанный топик (удобно для разных типов событий).
    /// </summary>
    public async Task PublishAsync(string topic, string key, string value, CancellationToken cancellationToken = default)
    {
        await _producer.ProduceAsync(topic, new Message<string, string> { Key = key, Value = value }, cancellationToken);
    }

    public void Dispose() => _producer.Dispose();
}
