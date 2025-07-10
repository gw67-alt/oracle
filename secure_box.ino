/*
 * Arduino Random Number Generator
 * 
 * This sketch generates a random number when the Arduino boots up
 * and repeatedly prints that number to the serial monitor.
 */

// The random number we'll generate at boot
long randomNumber;

void setup() {
  // Initialize serial communication at 9600 baud
  Serial.begin(9600);
  
  // Wait for serial connection to be established
  while (!Serial) {
    ; // Wait for serial port to connect
  }
  
  // Seed the random number generator
  // Using analogRead on an unconnected pin creates a fairly random seed
  randomSeed(analogRead(A0));
  
  // Generate our random number (between 1 and 1000)
  randomNumber = random(1, 9999);

  Serial.println(randomNumber);
 
  
  // Small delay before starting the loop
  delay(100);
}

void loop() {
  // Print the random number
  Serial.println(randomNumber);
  
  // Wait for a second before printing again
  delay(100);
}
