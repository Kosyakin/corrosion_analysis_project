using System.Text.Json.Serialization;

namespace asp_mvc_app.Models;

/// <summary>
/// Модель запроса на запись показаний датчика SCD40/SCD41.
/// Принимает JSON в snake_case от ESP32.
/// </summary>
public class AirSampleDto
{
    [JsonPropertyName("device_id")]
    public string DeviceId { get; set; } = "";

    [JsonPropertyName("measured_at")]
    public DateTime MeasuredAt { get; set; }

    [JsonPropertyName("co2_ppm")]
    public int Co2Ppm { get; set; }

    [JsonPropertyName("temperature_c")]
    public decimal TemperatureC { get; set; }

    [JsonPropertyName("humidity_rh")]
    public decimal HumidityRh { get; set; }

    [JsonPropertyName("fw_version")]
    public string? FwVersion { get; set; }
}
