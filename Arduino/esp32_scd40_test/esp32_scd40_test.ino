/*
 * ESP32 + SCD40/41 — чтение датчика и отправка в API раз в минуту.
 * Настройте Wi-Fi и URL API ниже.
 * I2C: SDA = 23, SCL = 22.
 * Serial: 115200.
 */

#include <Arduino.h>
#include <Wire.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <time.h>
#include <SensirionI2cScd4x.h>

// ----- Настройки: измените под свою сеть и сервер -----
#define WIFI_SSID      "Puk-puk"
#define WIFI_PASSWORD  "Kvv12345"
#define API_BASE_URL   "http://192.168.31.73:5640"   // Без слэша в конце
#define DEVICE_ID      "esp32-scd40-01"
#define FW_VERSION     "1.0"

// Пины I2C для ESP32
#define SDA_PIN 23
#define SCL_PIN 22

// Интервал отправки в API (мс)
#define SEND_INTERVAL_MS  60000

#ifdef NO_ERROR
#undef NO_ERROR
#endif
#define NO_ERROR 0

SensirionI2cScd4x sensor;
static char errorMessage[64];
static int16_t error;

// Последние прочитанные значения (отправляются в API)
static uint16_t lastCo2 = 0;
static float lastTemperature = 0.0f;
static float lastHumidity = 0.0f;
static bool lastDataValid = false;
static unsigned long lastSendTime = 0;

static void printError(const char* label) {
  Serial.print(label);
  Serial.print(": ");
  errorToString(error, errorMessage, sizeof(errorMessage));
  Serial.println(errorMessage);
}

// Получить текущее время UTC в формате ISO 8601 (буфер не менее 25 символов)
static bool getIsoTimeUtc(char* buf, size_t bufSize) {
  struct tm timeinfo;
  if (!getLocalTime(&timeinfo, 1000))
    return false;
  snprintf(buf, bufSize, "%04d-%02d-%02dT%02d:%02d:%02dZ",
           timeinfo.tm_year + 1900, timeinfo.tm_mon + 1, timeinfo.tm_mday,
           timeinfo.tm_hour, timeinfo.tm_min, timeinfo.tm_sec);
  return true;
}

static void connectWiFi() {
  if (WiFi.status() == WL_CONNECTED)
    return;
  Serial.print("Wi-Fi ");
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 30) {
    delay(500);
    Serial.print(".");
    attempts++;
  }
  Serial.println();
  if (WiFi.status() == WL_CONNECTED) {
    Serial.print("OK, IP: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("Wi-Fi не подключён");
  }
}

static void syncTime() {
  if (time(nullptr) > 1600000000)  // уже синхронизировано (после 2020)
    return;
  configTime(0, 0, "pool.ntp.org");
  Serial.print("NTP ");
  for (int i = 0; i < 10; i++) {
    delay(500);
    if (time(nullptr) > 1600000000) {
      Serial.println(" OK");
      return;
    }
    Serial.print(".");
  }
  Serial.println(" таймаут");
}

static void sendToApi() {
  if (!lastDataValid) {
    Serial.println("Нет данных для отправки");
    return;
  }
  connectWiFi();
  if (WiFi.status() != WL_CONNECTED)
    return;

  char isoTime[32];
  if (!getIsoTimeUtc(isoTime, sizeof(isoTime))) {
    strcpy(isoTime, "1970-01-01T00:00:00Z");
  }

  char tStr[12], hStr[12];
  snprintf(tStr, sizeof(tStr), "%.2f", lastTemperature);
  snprintf(hStr, sizeof(hStr), "%.2f", lastHumidity);

  char body[320];
  snprintf(body, sizeof(body),
           "{\"device_id\":\"%s\",\"measured_at\":\"%s\",\"co2_ppm\":%u,\"temperature_c\":%s,\"humidity_rh\":%s,\"fw_version\":\"%s\"}",
           DEVICE_ID, isoTime, lastCo2, tStr, hStr, FW_VERSION);

  HTTPClient http;
  String url = String(API_BASE_URL) + "/api/AirSamples";
  http.begin(url);
  http.addHeader("Content-Type", "application/json");
  int code = http.POST(body);
  if (code > 0) {
    if (code >= 200 && code < 300) {
      Serial.print("API OK ");
      Serial.println(code);
    } else {
      Serial.print("API ");
      Serial.print(code);
      Serial.print(" ");
      Serial.println(http.getString());
    }
  } else {
    Serial.print("API err ");
    Serial.println(http.errorToString(code));
  }
  http.end();
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("\n=== ESP32 SCD40 -> API ===");

  Wire.begin(SDA_PIN, SCL_PIN);
  sensor.begin(Wire, SCD41_I2C_ADDR_62);
  delay(30);

  error = sensor.wakeUp();
  if (error != NO_ERROR) printError("wakeUp");
  error = sensor.stopPeriodicMeasurement();
  if (error != NO_ERROR) printError("stopPeriodicMeasurement");
  error = sensor.reinit();
  if (error != NO_ERROR) printError("reinit");
  error = sensor.startPeriodicMeasurement();
  if (error != NO_ERROR) {
    printError("startPeriodicMeasurement");
    return;
  }
  Serial.println("SCD40 OK");

  connectWiFi();
  syncTime();

  lastSendTime = millis();
  Serial.println("Ожидание первого измерения (~5 с)...");
}

void loop() {
  bool dataReady = false;
  uint16_t co2 = 0;
  float temperature = 0.0f;
  float humidity = 0.0f;

  error = sensor.getDataReadyStatus(dataReady);
  if (error != NO_ERROR) {
    printError("getDataReadyStatus");
    delay(1000);
    return;
  }

  if (!dataReady) {
    if (lastDataValid && (millis() - lastSendTime >= SEND_INTERVAL_MS)) {
      sendToApi();
      lastSendTime = millis();
    }
    delay(1000);
    return;
  }

  error = sensor.readMeasurement(co2, temperature, humidity);
  if (error != NO_ERROR) {
    printError("readMeasurement");
    delay(1000);
    return;
  }

  lastCo2 = co2;
  lastTemperature = temperature;
  lastHumidity = humidity;
  lastDataValid = (co2 > 0);

  Serial.print("CO2: ");
  Serial.print(co2);
  Serial.print(" ppm, T: ");
  Serial.print(temperature, 2);
  Serial.print(" C, RH: ");
  Serial.print(humidity, 2);
  Serial.println(" %");

  if (lastDataValid && (millis() - lastSendTime >= SEND_INTERVAL_MS)) {
    sendToApi();
    lastSendTime = millis();
  }

  delay(1000);
}
