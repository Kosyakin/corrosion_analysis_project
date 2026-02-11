using System.Text.Json.Serialization;

namespace asp_mvc_app.Models;

/// <summary>
/// Элемент для графика: одна запись по времени.
/// </summary>
public class AirSampleChartItem
{
    [JsonPropertyName("measured_at")]
    public DateTime MeasuredAt { get; set; }

    [JsonPropertyName("co2_ppm")]
    public int Co2Ppm { get; set; }

    [JsonPropertyName("temperature_c")]
    public decimal TemperatureC { get; set; }

    [JsonPropertyName("humidity_rh")]
    public decimal HumidityRh { get; set; }
}
