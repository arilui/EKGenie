// Basic EKG data sender for Arduino
// Assumes you have an EKG sensor connected to analog pin A0

const int ekgPin = A0;

void setup() {
  Serial.begin(115200);  // Match this baud rate with your Python script
}

void loop() {
  int sensorValue = analogRead(ekgPin);
  
  // Convert to voltage (assuming 5V reference)
  float voltage = sensorValue * (5.0 / 1023.0);
  
  // Send the value over serial
  Serial.println(voltage, 4);  // 4 decimal places
  
  // Adjust delay based on your desired sampling rate
  delay(10);  // ~100Hz sampling rate
}