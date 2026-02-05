using KafkaExample;

var builder = WebApplication.CreateBuilder(args);

builder.Configuration.AddJsonFile("appsettings.Kafka.json", optional: true);

// Регистрация Kafka: опции читаются из секции "Kafka" (там имена топиков — очередей).
builder.Services.AddKafka(builder.Configuration);
builder.Services.AddControllers();

var app = builder.Build();
app.MapControllers();
app.Run();
