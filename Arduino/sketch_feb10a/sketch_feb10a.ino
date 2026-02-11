/*
 * SCD40/SCD41 — тестовая программа для ESP32.
 * Подключение I2C: SDA = GPIO 21, SCL = GPIO 22 (стандартные пины ESP32).
 * Можно использовать любые GPIO пины — измените SDA_PIN и SCL_PIN ниже.
 * Вывод: CO2 (ppm), температура (°C), влажность (%).
 * Serial: 115200.
 */

#include <Arduino.h>
#include <Wire.h>
#include <SensirionI2cScd4x.h>

// Пины I2C для ESP32 (можно изменить на любые GPIO)
#define SDA_PIN 23
#define SCL_PIN 22

#ifdef NO_ERROR
#undef NO_ERROR
#endif
#define NO_ERROR 0

SensirionI2cScd4x sensor;
static char errorMessage[64];
static int16_t error;

static void printError(const char* label) {
  Serial.print(label);
  Serial.print(": ");
  errorToString(error, errorMessage, sizeof(errorMessage));
  Serial.println(errorMessage);
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("\n=== ESP32 SCD40 Test ===");

  // Инициализация I2C с указанием пинов (ESP32 поддерживает это)
  Wire.begin(SDA_PIN, SCL_PIN);
  sensor.begin(Wire, SCD41_I2C_ADDR_62);
  delay(30);

  error = sensor.wakeUp();
  if (error != NO_ERROR) {
    printError("wakeUp");
  }
  error = sensor.stopPeriodicMeasurement();
  if (error != NO_ERROR) {
    printError("stopPeriodicMeasurement");
  }
  error = sensor.reinit();
  if (error != NO_ERROR) {
    printError("reinit");
  }

  error = sensor.startPeriodicMeasurement();
  if (error != NO_ERROR) {
    printError("startPeriodicMeasurement");
    return;
  }

  Serial.println("SCD40 ready. Waiting for first measurement (~5 s)...");
  Serial.print("I2C pins: SDA=");
  Serial.print(SDA_PIN);
  Serial.print(", SCL=");
  Serial.println(SCL_PIN);
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
    delay(1000);
    return;
  }

  error = sensor.readMeasurement(co2, temperature, humidity);
  if (error != NO_ERROR) {
    printError("readMeasurement");
    delay(1000);
    return;
  }

  Serial.print("CO2: ");
  Serial.print(co2);
  Serial.print(" ppm, T: ");
  Serial.print(temperature, 2);
  Serial.print(" C, RH: ");
  Serial.print(humidity, 2);
  Serial.println(" %");

  delay(1000);
}
