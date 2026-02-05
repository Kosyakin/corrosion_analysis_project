using System.Text.Json;
using Confluent.Kafka;
using KafkaExample.Messages;
using Microsoft.Extensions.Options;

namespace KafkaExample;

/// <summary>
/// Фоновый сервис — потребитель сообщений из топика Kafka (очереди).
/// Подключается к топику, указанному в KafkaOptions (TopicEvents), входит в Consumer Group
/// и в цикле читает сообщения. Партиции топика распределяются между экземплярами с одним GroupId.
/// Очереди (топики) не настраиваются в коде — только имена из KafkaOptions. Создание топиков — на стороне Kafka (kafka-topics или админ API).
/// </summary>
public class KafkaConsumerService : BackgroundService
{
    private readonly KafkaOptions _options;
    private readonly ILogger<KafkaConsumerService> _logger;
    private readonly IServiceProvider _serviceProvider;
    private static readonly JsonSerializerOptions JsonOptions = new() { PropertyNamingPolicy = JsonNamingPolicy.CamelCase };

    public KafkaConsumerService(
        IOptions<KafkaOptions> options,
        ILogger<KafkaConsumerService> logger,
        IServiceProvider serviceProvider)
    {
        _options = options.Value;
        _logger = logger;
        _serviceProvider = serviceProvider;
    }

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        var config = new ConsumerConfig
        {
            BootstrapServers = _options.BootstrapServers,
            GroupId = _options.ConsumerGroupId,
            AutoOffsetReset = _options.AutoOffsetReset == "Earliest"
                ? AutoOffsetReset.Earliest
                : AutoOffsetReset.Latest,
            EnableAutoCommit = true,
            EnableAutoOffsetStore = true,
        };

        using var consumer = new ConsumerBuilder<string, string>(config).Build();

        // Подписываемся на топик (очередь). Имя топика берётся из настроек — там и "настраиваются очереди".
        consumer.Subscribe(_options.TopicEvents);

        _logger.LogInformation(
            "Консьюмер запущен: топик {Topic}, группа {GroupId}",
            _options.TopicEvents, _options.ConsumerGroupId);

        try
        {
            while (!stoppingToken.IsCancellationRequested)
            {
                try
                {
                    // Блокирующее чтение следующего сообщения из топика (с таймаутом).
                    var cr = consumer.Consume(TimeSpan.FromSeconds(5));
                    if (cr == null) continue;

                    await ProcessMessageAsync(cr, stoppingToken);
                }
                catch (ConsumeException ex)
                {
                    _logger.LogWarning(ex, "Ошибка потребления сообщения Kafka");
                }
            }
        }
        finally
        {
            consumer.Close();
        }
    }

    private async Task ProcessMessageAsync(ConsumeResult<string, string> cr, CancellationToken ct)
    {
        _logger.LogInformation(
            "Получено сообщение: топик {Topic}, партиция {Partition}, offset {Offset}, key {Key}",
            cr.Topic, cr.Partition.Value, cr.Offset.Value, cr.Message.Key);

        try
        {
            var command = JsonSerializer.Deserialize<ServiceCommand>(cr.Message.Value, JsonOptions);
            if (command != null)
            {
                // Здесь — бизнес-обработка (вызов сервисов, сохранение в БД и т.д.).
                // Можно получить IMyApplicationService из _serviceProvider и вызвать его.
                await HandleCommandAsync(command, ct);
            }
        }
        catch (JsonException ex)
        {
            _logger.LogError(ex, "Не удалось десериализовать сообщение: {Value}", cr.Message.Value);
        }
    }

    private Task HandleCommandAsync(ServiceCommand command, CancellationToken ct)
    {
        _logger.LogInformation("Обработка команды {CommandId}, тип {Type}", command.CommandId, command.Type);
        return Task.CompletedTask;
    }
}
