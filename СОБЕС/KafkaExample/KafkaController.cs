using System.ComponentModel.DataAnnotations;
using KafkaExample.Messages;
using Microsoft.AspNetCore.Mvc;

namespace KafkaExample;

/// <summary>
/// Контроллер межсервисного взаимодействия через Kafka.
/// Принимает HTTP-запросы и публикует команды/события в топик Kafka — другие сервисы подписаны на этот топик и обрабатывают сообщения.
/// Очереди (топики) не настраиваются в контроллере: имена топиков берутся из KafkaOptions (appsettings → секция Kafka).
/// </summary>
[ApiController]
[Route("api/[controller]")]
public class KafkaController : ControllerBase
{
    private readonly KafkaProducerService _producer;
    private readonly ILogger<KafkaController> _logger;

    public KafkaController(KafkaProducerService producer, ILogger<KafkaController> logger)
    {
        _producer = producer;
        _logger = logger;
    }

    /// <summary>
    /// Отправка команды в топик Kafka (межсервисная очередь).
    /// Сервис получает запрос по HTTP и кладёт сообщение в топик; потребители других сервисов обрабатывают его асинхронно.
    /// </summary>
    /// <param name="request">Тело команды</param>
    /// <response code="202">Команда принята и отправлена в Kafka</response>
    [HttpPost("commands")]
    [ProducesResponseType(StatusCodes.Status202Accepted)]
    [ProducesResponseType(StatusCodes.Status400BadRequest)]
    public async Task<IActionResult> SendCommand([FromBody] SendCommandRequest request, CancellationToken cancellationToken)
    {
        if (string.IsNullOrWhiteSpace(request?.Type))
            return BadRequest("Укажите Type команды.");

        var command = new ServiceCommand
        {
            CommandId = request.CommandId ?? Guid.NewGuid().ToString("N"),
            Type = request.Type,
            Payload = request.Payload ?? string.Empty,
            CreatedAtUtc = DateTime.UtcNow,
        };

        await _producer.PublishCommandAsync(command, cancellationToken);

        _logger.LogInformation("Команда отправлена в Kafka: {CommandId}, {Type}", command.CommandId, command.Type);

        // 202 Accepted — запрос принят, обработка будет асинхронной (другим сервисом).
        return Accepted(new { commandId = command.CommandId, topic = "service.commands" });
    }

    /// <summary>
    /// Отправка произвольного сообщения в указанный топик (для тестов или универсального шлюза).
    /// </summary>
    [HttpPost("publish")]
    [ProducesResponseType(StatusCodes.Status202Accepted)]
    public async Task<IActionResult> Publish(
        [FromBody] PublishRequest request,
        CancellationToken cancellationToken)
    {
        if (string.IsNullOrWhiteSpace(request?.Topic))
            return BadRequest("Укажите Topic.");
        if (string.IsNullOrWhiteSpace(request.Key))
            return BadRequest("Укажите Key.");

        await _producer.PublishAsync(request.Topic, request.Key, request.Value ?? string.Empty, cancellationToken);

        return Accepted(new { topic = request.Topic, key = request.Key });
    }
}

/// <summary>Тело запроса на отправку команды в Kafka.</summary>
public class SendCommandRequest
{
    public string? CommandId { get; set; }
    [Required] public string? Type { get; set; }
    public string? Payload { get; set; }
}

/// <summary>Тело запроса на публикацию в произвольный топик.</summary>
public class PublishRequest
{
    [Required] public string? Topic { get; set; }
    [Required] public string? Key { get; set; }
    public string? Value { get; set; }
}
