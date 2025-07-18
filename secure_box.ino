/*
* Arduino Secure Random Number Generator - V3
*
* This sketch implements a robust startup handshake. It waits for a serial
* connection to be fully established before seeding the random number
* generator. This prevents using the same seed after each reset.
*
* It then waits for an access token 'R' to provide a new random number.
*/

void setup() {
  // 1. Initialize serial communication
  Serial.begin(9600);

  // 2. THIS IS THE KEY FIX: Wait until the serial port is connected.
  // This gives the floating analog pin time to build up random noise.
  while (!Serial); 
  
  // 3. Add a small extra delay for stability.
  delay(50);

  // 4. Now, seed the random number generator. The seed will be much more random.
  randomSeed(analogRead(A0));
  
  // 5. Send a "ready" signal to the Python script to confirm setup is complete.
  Serial.println("ARDUINO_READY");
}

void loop() {
  // Check if there is data available to read.
  if (Serial.available() > 0) {
    // Read the incoming byte.
    char accessToken = Serial.read();

    // Check if the received token is our access key ('R').
    if (accessToken == 'R') {
      // NOTE: We do NOT re-seed here. We just pull the next number
      // from the already-seeded random sequence.
      long newRandomNumber = random(1, 100);
      
      delay(5000);

      // Send the fresh random number back.
      Serial.println(newRandomNumber);
    }
  }
}
