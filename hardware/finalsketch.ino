#include <WiFi.h>
#include <HTTPClient.h>
#include <Adafruit_BMP085.h>

Adafruit_BMP085 bmp;

const char* ssid = "Shynar-Kausar";
const char* password = "bektai201409";

const char* serverName = "http://192.168.100.237:8000/sensors";
unsigned long lastTime = 0;
unsigned long timerDelay = 5000;

float temperature = 0;
float pressure = 0;

void setup() {
  Serial.begin(115200);

  WiFi.begin(ssid, password);
  Serial.println("Connecting");
  while(WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("");
  Serial.print("Connected to WiFi network with IP Address: ");
  Serial.println(WiFi.localIP());

  if (!bmp.begin()) {
	Serial.println("Could not find a valid BMP085/BMP180 sensor, check wiring!");
	while (1) {}
  }
  
  Serial.println("Timer set to 5 seconds (timerDelay variable), it will take 5 seconds before publishing the first reading.");
}



void read_sensors() {
  temperature = bmp.readTemperature();
  pressure = bmp.readSealevelPressure();
}

void send_data() {
  if ((millis() - lastTime) > timerDelay) {
    if(WiFi.status()== WL_CONNECTED){
      WiFiClient client;
      HTTPClient http;
      http.begin(client, serverName);
      http.addHeader("Content-Type", "application/json");
      String json = "{";
      json += "\"Temperature\":" + String(temperature) + ",";
      json += "\"Pressure\":" + String(pressure);
      json += "}";          
      int httpResponseCode = http.POST(json);
      Serial.print("HTTP Response code: ");
      Serial.println(httpResponseCode);
      http.end();
    }
    else {
      Serial.println("WiFi Disconnected");
    }
    lastTime = millis();
  }
}
void loop() {
  read_sensors();
  send_data();
}