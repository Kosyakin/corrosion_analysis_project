using asp_mvc_app.Models;
using Microsoft.AspNetCore.Mvc;
using Npgsql;

namespace asp_mvc_app.Controllers;

[ApiController]
[Route("api/[controller]")]
[Produces("application/json")]
public class AirSamplesController : ControllerBase
{
    private readonly IConfiguration _config;
    private readonly ILogger<AirSamplesController> _logger;

    public AirSamplesController(IConfiguration config, ILogger<AirSamplesController> logger)
    {
        _config = config;
        _logger = logger;
    }

    /// <summary>
    /// Получить последние показания для графиков (по умолчанию 500 записей).
    /// </summary>
    [HttpGet]
    [ProducesResponseType(typeof(List<AirSampleChartItem>), StatusCodes.Status200OK)]
    public async Task<ActionResult<List<AirSampleChartItem>>> Get([FromQuery] int limit = 500, [FromQuery] string? device_id = null, CancellationToken ct = default)
    {
        if (limit <= 0 || limit > 5000) limit = 500;
        var connectionString = _config.GetConnectionString("DefaultConnection");
        if (string.IsNullOrEmpty(connectionString))
            return StatusCode(500, new List<AirSampleChartItem>());

        try
        {
            await using var conn = new NpgsqlConnection(connectionString);
            await conn.OpenAsync(ct);
            var sql = @"
                SELECT measured_at, co2_ppm, temperature_c, humidity_rh
                FROM public.air_samples
                WHERE (@device_id IS NULL OR device_id = @device_id)
                ORDER BY measured_at DESC
                LIMIT @limit";
            await using var cmd = new NpgsqlCommand(sql, conn);
            cmd.Parameters.AddWithValue("limit", limit);
            cmd.Parameters.AddWithValue("device_id", (object?)device_id ?? DBNull.Value);

            var list = new List<AirSampleChartItem>();
            await using var reader = await cmd.ExecuteReaderAsync(ct);
            while (await reader.ReadAsync(ct))
            {
                list.Add(new AirSampleChartItem
                {
                    MeasuredAt = reader.GetDateTime(0),
                    Co2Ppm = reader.GetInt32(1),
                    TemperatureC = reader.GetDecimal(2),
                    HumidityRh = reader.GetDecimal(3)
                });
            }
            list.Reverse();
            return Ok(list);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Ошибка при чтении данных для графиков");
            return StatusCode(500, new List<AirSampleChartItem>());
        }
    }

    /// <summary>
    /// Принять и сохранить показания датчика в БД.
    /// </summary>
    /// <param name="dto">Данные с датчика: device_id, measured_at, co2_ppm, temperature_c, humidity_rh, fw_version.</param>
    [HttpPost]
    [ProducesResponseType(StatusCodes.Status201Created)]
    [ProducesResponseType(StatusCodes.Status400BadRequest)]
    [ProducesResponseType(StatusCodes.Status500InternalServerError)]
    public async Task<ActionResult<long>> Post([FromBody] AirSampleDto dto, CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(dto.DeviceId))
        {
            return BadRequest(new { error = "device_id обязателен" });
        }
        if (dto.Co2Ppm < 0 || dto.Co2Ppm > 5000)
        {
            return BadRequest(new { error = "co2_ppm должен быть от 0 до 5000" });
        }
        if (dto.TemperatureC < -40 || dto.TemperatureC > 85)
        {
            return BadRequest(new { error = "temperature_c должен быть от -40 до 85" });
        }
        if (dto.HumidityRh < 0 || dto.HumidityRh > 100)
        {
            return BadRequest(new { error = "humidity_rh должен быть от 0 до 100" });
        }

        var connectionString = _config.GetConnectionString("DefaultConnection");
        if (string.IsNullOrEmpty(connectionString))
        {
            _logger.LogError("ConnectionStrings:DefaultConnection не задан");
            return StatusCode(500, new { error = "Не настроено подключение к БД" });
        }

        try
        {
            await using var conn = new NpgsqlConnection(connectionString);
            await conn.OpenAsync(ct);

            const string sql = @"
                INSERT INTO public.air_samples (device_id, measured_at, co2_ppm, temperature_c, humidity_rh, fw_version)
                VALUES (@device_id, @measured_at, @co2_ppm, @temperature_c, @humidity_rh, @fw_version)
                RETURNING id;";

            await using var cmd = new NpgsqlCommand(sql, conn);
            cmd.Parameters.AddWithValue("device_id", dto.DeviceId);
            cmd.Parameters.AddWithValue("measured_at", DateTime.SpecifyKind(dto.MeasuredAt, DateTimeKind.Utc));
            cmd.Parameters.AddWithValue("co2_ppm", dto.Co2Ppm);
            cmd.Parameters.AddWithValue("temperature_c", dto.TemperatureC);
            cmd.Parameters.AddWithValue("humidity_rh", dto.HumidityRh);
            cmd.Parameters.AddWithValue("fw_version", (object?)dto.FwVersion ?? DBNull.Value);

            var id = (long)(await cmd.ExecuteScalarAsync(ct))!;
            _logger.LogInformation("Сохранена запись air_samples id={Id}, device={Device}", id, dto.DeviceId);
            return CreatedAtAction(nameof(Post), new { id }, id);
        }
        catch (PostgresException ex) when (ex.SqlState == "23514")
        {
            _logger.LogWarning("Нарушение CHECK при вставке: {Msg}", ex.Message);
            return BadRequest(new { error = "Данные не прошли проверку диапазонов в БД" });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Ошибка при записи в БД");
            return StatusCode(500, new { error = "Ошибка при сохранении данных" });
        }
    }
}
