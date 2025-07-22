#include <Servo.h>
Servo myservo; 
int pos = 0;   
void setup() {
  myservo.attach(9); //glass diode, black side. Other end connected to A0.
  Serial.begin(9600);
}
void loop() {
  for (pos = 0; pos <= 180; pos += 1) {
    myservo.write(pos); 
    Serial.println(analogRead(A0));
    delay(15);
  }
  for (pos = 180; pos >= 0; pos -= 1) {
    myservo.write(pos);      
    Serial.println(analogRead(A0));
    delay(15);
  }
}

