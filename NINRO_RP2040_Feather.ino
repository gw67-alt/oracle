// Advanced Curvature Analysis Oscilloscope with Matrix Storage
// Enhanced for curvature-RMS correlation analysis

#define NUM_CHANNELS 6
#define SAMPLE_RATE_MS 20
#define HISTORY_SIZE 50        // Increased for better matrix analysis
#define MATRIX_DEPTH 100       // Matrix history depth
#define DISPLAY_HEIGHT 12
#define DISPLAY_WIDTH 80
#define SMOOTHING_WINDOW 7
#define STATE_SPACE 999999

int analogPins[NUM_CHANNELS] = {A0, A1, A2, A3};

// Enhanced matrix structures for curvature-RMS correlation
struct CurvatureRMSMatrix {
  float curvature_history[NUM_CHANNELS][MATRIX_DEPTH];
  float rms_history[NUM_CHANNELS][MATRIX_DEPTH];
  float correlation_matrix[NUM_CHANNELS][MATRIX_DEPTH];
  float peak_matrix[NUM_CHANNELS][MATRIX_DEPTH];
  float slope_matrix[NUM_CHANNELS][MATRIX_DEPTH];
  unsigned long timestamps[MATRIX_DEPTH];
  int matrix_index = 0;
  int filled_entries = 0;
  bool matrix_full = false;
};

// Signal processing buffers
float channelHistory[NUM_CHANNELS][HISTORY_SIZE];
float smoothedData[NUM_CHANNELS][HISTORY_SIZE];
float curvatureData[NUM_CHANNELS][HISTORY_SIZE];
float instantRMS[NUM_CHANNELS];
float instantCurvature[NUM_CHANNELS];

// Matrix storage
CurvatureRMSMatrix signalMatrix;

// Analysis variables
float signFlips[HISTORY_SIZE];
int flipCount = 0;
int historyIndex = 0;
unsigned long lastSample = 0;
int selectedChannel = 0;
unsigned long analysisCount = 0;

void setup() {
  Serial.begin(115200);
  
  initializeMatrices();
  
  Serial.println("CURVATURE-RMS MATRIX OSCILLOSCOPE");
  Serial.println("=================================");
  Serial.println("Commands:");
  Serial.println("  0-5: Select channel");
  Serial.println("  m: Display matrix stats");
  Serial.println("  c: Curvature analysis"); 
  Serial.println("  r: Reset matrices");
  Serial.println("  x: Export matrix data");
  Serial.println("  t: Real-time correlation");
  Serial.println("=================================");
}

void loop() {
  unsigned long currentTime = millis();
  
  handleSerialCommands();
  
  if (currentTime - lastSample >= SAMPLE_RATE_MS) {
    // Enhanced sampling with immediate processing
    sampleAllChannelsEnhanced();
    
    // Apply smoothing to all channels
    applySmoothingFilterAll();
    
    // Calculate curvature for all channels
    performMultiChannelCurvatureAnalysis();
    
    // Calculate RMS for all channels
    calculateInstantaneousRMS();
    
    // Store in matrix
    updateCurvatureRMSMatrix();
    
    // Perform correlation analysis
    calculateCorrelationMetrics();
    
    // Display real-time data
    displayMatrixOscilloscope();
    
    // Detect patterns
    detectCurvatureRMSPatterns();
    
    historyIndex = (historyIndex + 1) % HISTORY_SIZE;
    analysisCount++;
    lastSample = currentTime;
  }
}

void initializeMatrices() {
  // Clear all matrix data
  for (int ch = 0; ch < NUM_CHANNELS; ch++) {
    for (int i = 0; i < HISTORY_SIZE; i++) {
      channelHistory[ch][i] = 0.0;
      smoothedData[ch][i] = 0.0;
      curvatureData[ch][i] = 0.0;
    }
    instantRMS[ch] = 0.0;
    instantCurvature[ch] = 0.0;
  }
  
  // Clear matrix storage
  memset(&signalMatrix, 0, sizeof(CurvatureRMSMatrix));
  
  Serial.println("■ Matrices initialized");
}

void sampleAllChannelsEnhanced() {
  for (int ch = 0; ch < NUM_CHANNELS; ch++) {
    int rawValue = analogRead(analogPins[ch]);
    float voltage = (rawValue / 1023.0) * 5.0;
    
    // Enhanced noise filtering with adaptive threshold
    if (historyIndex > 0) {
      float prevVoltage = channelHistory[ch][(historyIndex - 1 + HISTORY_SIZE) % HISTORY_SIZE];
      float diff = abs(voltage - prevVoltage);
      
      // Adaptive filtering based on signal characteristics
      float noiseThreshold = max(0.01, instantRMS[ch] * 0.1);
      if (diff > noiseThreshold) {
        voltage = (voltage + prevVoltage) / 2.0;
      }
    }
    
    channelHistory[ch][historyIndex] = voltage;
  }
}

void applySmoothingFilterAll() {
  for (int ch = 0; ch < NUM_CHANNELS; ch++) {
    float sum = 0;
    int count = 0;
    
    for (int i = 0; i < min(SMOOTHING_WINDOW, HISTORY_SIZE); i++) {
      int idx = (historyIndex - i + HISTORY_SIZE) % HISTORY_SIZE;
      sum += channelHistory[ch][idx];
      count++;
    }
    
    if (count > 0) {
      smoothedData[ch][historyIndex] = sum / count;
    }
  }
}

void performMultiChannelCurvatureAnalysis() {
  for (int ch = 0; ch < NUM_CHANNELS; ch++) {
    if (historyIndex >= 2) {
      int curr = historyIndex;
      int prev = (historyIndex - 1 + HISTORY_SIZE) % HISTORY_SIZE;
      int prev2 = (historyIndex - 2 + HISTORY_SIZE) % HISTORY_SIZE;
      
      float currentVal = smoothedData[ch][curr];
      float prevVal = smoothedData[ch][prev];
      float prev2Val = smoothedData[ch][prev2];
      
      // Enhanced curvature calculation
      float firstDeriv = currentVal - prevVal;
      float secondDeriv = currentVal - 2 * prevVal + prev2Val;
      
      // Store both local curvature and smoothed curvature
      curvatureData[ch][historyIndex] = abs(secondDeriv) * 100;
      
      // Calculate instantaneous curvature (average over recent samples)
      float curvSum = 0;
      int curvCount = 0;
      for (int i = 0; i < min(5, HISTORY_SIZE); i++) {
        int idx = (historyIndex - i + HISTORY_SIZE) % HISTORY_SIZE;
        curvSum += curvatureData[ch][idx];
        curvCount++;
      }
      instantCurvature[ch] = curvSum / curvCount;
    }
  }
}

void calculateInstantaneousRMS() {
  for (int ch = 0; ch < NUM_CHANNELS; ch++) {
    float sumSquares = 0;
    int rmsWindow = min(10, HISTORY_SIZE); // 10-sample RMS window
    
    for (int i = 0; i < rmsWindow; i++) {
      int idx = (historyIndex - i + HISTORY_SIZE) % HISTORY_SIZE;
      float value = smoothedData[ch][idx];
      sumSquares += value * value;
    }
    
    instantRMS[ch] = sqrt(sumSquares / rmsWindow);
  }
}

void updateCurvatureRMSMatrix() {
  int idx = signalMatrix.matrix_index;
  
  // Store current values in matrix
  for (int ch = 0; ch < NUM_CHANNELS; ch++) {
    signalMatrix.curvature_history[ch][idx] = instantCurvature[ch];
    signalMatrix.rms_history[ch][idx] = instantRMS[ch];
    signalMatrix.peak_matrix[ch][idx] = getPeakValue(ch);
    
    // Calculate slope (rate of change)
    if (signalMatrix.filled_entries > 0) {
      int prevIdx = (idx - 1 + MATRIX_DEPTH) % MATRIX_DEPTH;
      signalMatrix.slope_matrix[ch][idx] = 
        (instantRMS[ch] - signalMatrix.rms_history[ch][prevIdx]) / SAMPLE_RATE_MS * 1000;
    }
  }
  
  signalMatrix.timestamps[idx] = millis();
  signalMatrix.matrix_index = (idx + 1) % MATRIX_DEPTH;
  
  if (!signalMatrix.matrix_full) {
    signalMatrix.filled_entries++;
    if (signalMatrix.filled_entries >= MATRIX_DEPTH) {
      signalMatrix.matrix_full = true;
    }
  }
}

void calculateCorrelationMetrics() {
  int entries = signalMatrix.matrix_full ? MATRIX_DEPTH : signalMatrix.filled_entries;
  if (entries < 10) return;
  
  // Calculate curvature-RMS correlation for each channel
  for (int ch = 0; ch < NUM_CHANNELS; ch++) {
    float curvMean = 0, rmsMean = 0;
    
    // Calculate means
    for (int i = 0; i < entries; i++) {
      curvMean += signalMatrix.curvature_history[ch][i];
      rmsMean += signalMatrix.rms_history[ch][i];
    }
    curvMean /= entries;
    rmsMean /= entries;
    
    // Calculate correlation coefficient
    float numerator = 0, curvVar = 0, rmsVar = 0;
    for (int i = 0; i < entries; i++) {
      float curvDiff = signalMatrix.curvature_history[ch][i] - curvMean;
      float rmsDiff = signalMatrix.rms_history[ch][i] - rmsMean;
      
      numerator += curvDiff * rmsDiff;
      curvVar += curvDiff * curvDiff;
      rmsVar += rmsDiff * rmsDiff;
    }
    
    float correlation = 0;
    if (curvVar > 0 && rmsVar > 0) {
      correlation = numerator / sqrt(curvVar * rmsVar);
    }
    
    // Store correlation in matrix
    int currentIdx = (signalMatrix.matrix_index - 1 + MATRIX_DEPTH) % MATRIX_DEPTH;
    signalMatrix.correlation_matrix[ch][currentIdx] = correlation;
  }
}

void displayMatrixOscilloscope() {
  // Compact real-time display
  Serial.print("CH"); Serial.print(selectedChannel); Serial.print(": ");
  Serial.print("RMS="); Serial.print(instantRMS[selectedChannel], 3); Serial.print("V ");
  Serial.print("CURV="); Serial.print(instantCurvature[selectedChannel], 2); Serial.print(" ");
  
  // Get latest correlation
  int lastIdx = (signalMatrix.matrix_index - 1 + MATRIX_DEPTH) % MATRIX_DEPTH;
  if (signalMatrix.filled_entries > 10) {
    Serial.print("CORR="); Serial.print(signalMatrix.correlation_matrix[selectedChannel][lastIdx], 3);
  }
  Serial.println();
}

void detectCurvatureRMSPatterns() {
  // Advanced pattern detection using matrix data
  if (signalMatrix.filled_entries < 20) return;
  
  int entries = min(20, signalMatrix.filled_entries);
  bool patternDetected = false;
  
  for (int ch = 0; ch < NUM_CHANNELS; ch++) {
    // Check for high correlation between curvature and RMS
    int recentIdx = (signalMatrix.matrix_index - 1 + MATRIX_DEPTH) % MATRIX_DEPTH;
    float correlation = signalMatrix.correlation_matrix[ch][recentIdx];
    
    // Pattern: High correlation with significant RMS change
    float rmsChange = abs(signalMatrix.slope_matrix[ch][recentIdx]);
    
    if (abs(correlation) > 0.7 && rmsChange > 0.1 && instantRMS[ch] > 0.1) {
      Serial.print("■ PATTERN CH"); Serial.print(ch); 
      Serial.print(": Strong Curv-RMS correlation ("); 
      Serial.print(correlation, 3); Serial.print("), Rate=");
      Serial.print(rmsChange, 3); Serial.println("V/s");
      patternDetected = true;
    }
  }
  
  if (patternDetected) {
    analysisCount += 10; // Boost analysis counter for pattern events
  }
}

void handleSerialCommands() {
  if (Serial.available() > 0) {
    char cmd = Serial.read();
    
    if (cmd >= '0' && cmd <= '5') {
      selectedChannel = cmd - '0';
      Serial.print("■ SELECTED CHANNEL A");
      Serial.println(selectedChannel);
    } else if (cmd == 'm') {
      displayMatrixStatistics();
    } else if (cmd == 'c') {
      displayCurvatureStats();
    } else if (cmd == 'r') {
      resetAnalysis();
    } else if (cmd == 'x') {
      exportMatrixData();
    } else if (cmd == 't') {
      displayRealTimeCorrelation();
    }
  }
}

void displayMatrixStatistics() {
  Serial.println("\n█████ CURVATURE-RMS MATRIX ANALYSIS █████");
  
  int entries = signalMatrix.matrix_full ? MATRIX_DEPTH : signalMatrix.filled_entries;
  Serial.print("Matrix Entries: "); Serial.println(entries);
  
  for (int ch = 0; ch < NUM_CHANNELS; ch++) {
    if (instantRMS[ch] < 0.01) continue; // Skip inactive channels
    
    Serial.print("CH"); Serial.print(ch); Serial.println(":");
    
    // Calculate statistics
    float avgRMS = 0, avgCurv = 0, avgCorr = 0;
    float maxRMS = 0, maxCurv = 0;
    
    for (int i = 0; i < entries; i++) {
      avgRMS += signalMatrix.rms_history[ch][i];
      avgCurv += signalMatrix.curvature_history[ch][i];
      if (i >= 10) avgCorr += signalMatrix.correlation_matrix[ch][i];
      
      if (signalMatrix.rms_history[ch][i] > maxRMS) maxRMS = signalMatrix.rms_history[ch][i];
      if (signalMatrix.curvature_history[ch][i] > maxCurv) maxCurv = signalMatrix.curvature_history[ch][i];
    }
    
    avgRMS /= entries;
    avgCurv /= entries;
    if (entries > 10) avgCorr /= (entries - 10);
    
    Serial.print("  Avg RMS: "); Serial.print(avgRMS, 3); Serial.println("V");
    Serial.print("  Max RMS: "); Serial.print(maxRMS, 3); Serial.println("V");
    Serial.print("  Avg Curvature: "); Serial.println(avgCurv, 2);
    Serial.print("  Max Curvature: "); Serial.println(maxCurv, 2);
    Serial.print("  Avg Correlation: "); Serial.println(avgCorr, 3);
  }
  
  Serial.println("██████████████████████████████████████████");
}

void exportMatrixData() {
  Serial.println("\n=== MATRIX DATA EXPORT ===");
  Serial.println("Timestamp,Channel,RMS,Curvature,Correlation,Peak,Slope");
  
  int entries = signalMatrix.matrix_full ? MATRIX_DEPTH : signalMatrix.filled_entries;
  
  for (int i = 0; i < entries; i++) {
    for (int ch = 0; ch < NUM_CHANNELS; ch++) {
      if (signalMatrix.rms_history[ch][i] < 0.01) continue;
      
      Serial.print(signalMatrix.timestamps[i]); Serial.print(",");
      Serial.print(ch); Serial.print(",");
      Serial.print(signalMatrix.rms_history[ch][i], 4); Serial.print(",");
      Serial.print(signalMatrix.curvature_history[ch][i], 4); Serial.print(",");
      Serial.print(signalMatrix.correlation_matrix[ch][i], 4); Serial.print(",");
      Serial.print(signalMatrix.peak_matrix[ch][i], 4); Serial.print(",");
      Serial.print(signalMatrix.slope_matrix[ch][i], 4);
      Serial.println();
    }
  }
}

void displayRealTimeCorrelation() {
  Serial.println("\n=== REAL-TIME CORRELATION MATRIX ===");
  
  for (int ch = 0; ch < NUM_CHANNELS; ch++) {
    if (instantRMS[ch] < 0.01) continue;
    
    int lastIdx = (signalMatrix.matrix_index - 1 + MATRIX_DEPTH) % MATRIX_DEPTH;
    float correlation = signalMatrix.correlation_matrix[ch][lastIdx];
    
    Serial.print("A"); Serial.print(ch); Serial.print(": ");
    
    // Visual correlation indicator
    if (correlation > 0.5) Serial.print("██████");
    else if (correlation > 0.3) Serial.print("████  ");
    else if (correlation > 0) Serial.print("██    ");
    else if (correlation > -0.3) Serial.print("      ");
    else if (correlation > -0.5) Serial.print("  ██  ");
    else Serial.print("██████");
    
    Serial.print(" ("); Serial.print(correlation, 3); Serial.println(")");
  }
}

// Utility functions from original code
float getCurrentValue(int channel) {
  int currentIndex = (historyIndex - 1 + HISTORY_SIZE) % HISTORY_SIZE;
  return channelHistory[channel][currentIndex];
}

float getPeakValue(int channel) {
  float peak = 0;
  for (int i = 0; i < HISTORY_SIZE; i++) {
    if (abs(channelHistory[channel][i]) > peak) {
      peak = abs(channelHistory[channel][i]);
    }
  }
  return peak;
}

void resetAnalysis() {
  initializeMatrices();
  analysisCount = 0;
  flipCount = 0;
  Serial.println("■ ALL MATRICES RESET");
}

void displayCurvatureStats() {
  Serial.println("█████ CURVATURE ANALYSIS REPORT █████");
  Serial.print("Total Samples: "); Serial.println(analysisCount);
  Serial.print("Matrix Entries: "); 
  Serial.println(signalMatrix.matrix_full ? MATRIX_DEPTH : signalMatrix.filled_entries);
  
  for (int ch = 0; ch < NUM_CHANNELS; ch++) {
    if (instantRMS[ch] > 0.01) {
      Serial.print("CH"); Serial.print(ch); Serial.print(" - RMS:");
      Serial.print(instantRMS[ch], 3); Serial.print("V Curv:");
      Serial.println(instantCurvature[ch], 2);
    }
  }
  Serial.println("██████████████████████████████████████");
}
