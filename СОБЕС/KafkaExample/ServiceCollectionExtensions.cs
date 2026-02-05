using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;

namespace KafkaExample;

/// <summary>
/// Регистрация Kafka-сервисов в DI. Очереди (топики) настраиваются через конфиг — см. KafkaOptions и appsettings.
/// </summary>
public static class ServiceCollectionExtensions
{
    public static IServiceCollection AddKafka(this IServiceCollection services, IConfiguration configuration)
    {
        services.Configure<KafkaOptions>(configuration.GetSection(KafkaOptions.SectionName));
        services.AddSingleton<KafkaProducerService>();
        services.AddHostedService<KafkaConsumerService>();
        return services;
    }
}
