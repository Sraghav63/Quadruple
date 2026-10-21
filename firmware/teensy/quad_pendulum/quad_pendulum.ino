// Teensy 4.1 / Teensyduino. Generate qp_gains.h with scripts/export_firmware.py.
#include <Arduino.h>
#include <IntervalTimer.h>
#include <SPI.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "qp_gains.h"

static_assert(QP_NY == 5, "This sketch reads four relative joint encoders");
static_assert(QP_DT > 0.0f, "Invalid controller period");
static_assert(QP_COMPUTE_DELAY_STEPS >= 0, "Invalid command delay");

constexpr uint8_t kCS[4] = {10, 9, 8, 7};
constexpr uint8_t kCalibrateButton = 2;  // active low, INPUT_PULLUP
constexpr int kEncoderSign[4] = {1, 1, 1, 1};  // verify on each physical joint
constexpr float kRadiansPerCount = 6.283185307179586f / 16384.0f;
constexpr float kArmAngle = 3.0f * DEG_TO_RAD;
constexpr float kTripAngle = 25.0f * DEG_TO_RAD;
constexpr unsigned kHoldTicks = (500000UL + QP_CONTROL_US - 1) / QP_CONTROL_US;
constexpr unsigned kTelemetryTicks = (10000UL + QP_CONTROL_US - 1) / QP_CONTROL_US;

IntervalTimer controlTimer;
SPISettings encoderSPI(1000000, MSBFIRST, SPI_MODE1);

// HARDWARE STUB: read the motor/linear encoder, convert to metres from rail
// centre, check timestamp and drive faults. Use an interrupt-safe cached CAN
// reply or quadrature counter; never wait for a CAN reply inside the timer ISR.
bool readCartPositionMeters(float &x) {
  x = NAN;
  return false;  // fail closed until implemented; motor turns require pulley gearing
}

// HARDWARE STUB: enqueue a moteus CAN-FD velocity-mode frame, with motor
// revolutions/s = belt_m_s / (pulley_circumference_m / transmission_ratio).
// enabled=false MUST enqueue zero velocity AND a drive stop/disable command.
// Return false for queue overflow, stale feedback or drive fault. Bound execution
// time; configure a drive-side watchdog <=20 ms and an independent hard E-stop.
bool sendDriveVelocity(float belt_m_s, bool enabled) {
  (void)belt_m_s;
  (void)enabled;
  return false;  // implement using the chosen CAN-FD transceiver + moteus transport
}

enum Fault : uint8_t { OK, SENSOR, CART, LIMIT, DRIVE, OVERRUN, NUMERIC };
struct Telemetry {
  uint32_t ticks;
  float x, ref, theta[4], accel, velocity;
  uint32_t runtime_us;
  bool armed, calibrated;
  Fault fault;
};
volatile Telemetry telemetry = {};
volatile bool requestArm = false, requestDisarm = false, requestCalibration = false;
volatile float requestedReference = 0.0f;

// The following state belongs exclusively to the timer ISR.
float z[QP_NZ] = {};
float commandQueue[QP_COMPUTE_DELAY_STEPS + 1] = {};
uint16_t zeroCounts[4] = {};
bool calibrated = false, armed = false, armPending = false;
unsigned uprightTicks = 0;
float previousReference = 0.0f, velocitySetpoint = 0.0f;
uint32_t ticks = 0;
Fault fault = OK;

float clampValue(float x, float lower, float upper) {
  return fminf(upper, fmaxf(lower, x));
}

bool evenParity(uint16_t word) {
  word ^= word >> 8;
  word ^= word >> 4;
  word ^= word >> 2;
  word ^= word >> 1;
  return (word & 1u) == 0;
}

uint16_t spiFrame(uint8_t cs, uint16_t command) {
  digitalWrite(cs, LOW);
  uint16_t result = SPI.transfer16(command);
  digitalWrite(cs, HIGH);
  delayMicroseconds(1);  // AS5047P inter-frame CS-high time
  return result;
}

bool readEncoder(uint8_t cs, uint16_t &angle) {
  // Pipelined read of ANGLECOM (0x3fff): read bit + even parity = 0xffff.
  SPI.beginTransaction(encoderSPI);
  spiFrame(cs, 0xffff);
  uint16_t response = spiFrame(cs, 0x0000);  // NOP clocks out requested data
  bool valid = evenParity(response) && !(response & 0x4000);  // EF
  if (!valid) {
    spiFrame(cs, 0x4001);  // read ERRFL; clear sticky SPI errors for next sample
    spiFrame(cs, 0x0000);
  }
  SPI.endTransaction();
  angle = response & 0x3fff;
  return valid;
}

void stopController(Fault reason) {
  armed = false;
  armPending = false;
  uprightTicks = 0;
  velocitySetpoint = 0.0f;
  memset(z, 0, sizeof(z));
  memset(commandQueue, 0, sizeof(commandQueue));
  fault = reason;
  sendDriveVelocity(0.0f, false);
}

void resetObserver(float x, const float theta[4], float reference) {
  memset(z, 0, sizeof(z));
  z[0] = x - reference;
  for (unsigned i = 0; i < 4; ++i) z[3 + i] = theta[i];
  memset(commandQueue, 0, sizeof(commandQueue));
  previousReference = reference;
  velocitySetpoint = 0.0f;
}

float controllerStep(float x, const float theta[4], float reference) {
  // Match LQRController.step, including shifting the reference coordinate.
  z[0] -= reference - previousReference;
  previousReference = reference;
  float y[QP_NY] = {x - reference, theta[0], theta[1], theta[2], theta[3]};
  float innovation[QP_NY] = {};
  for (unsigned i = 0; i < QP_NY; ++i) {
    innovation[i] = y[i];
    for (unsigned j = 0; j < QP_NZ; ++j)
      innovation[i] -= QP_C[i * QP_NZ + j] * z[j];
  }
  for (unsigned i = 0; i < QP_NZ; ++i)
    for (unsigned j = 0; j < QP_NY; ++j)
      z[i] += QP_L[i * QP_NY + j] * innovation[j];
  float u = 0.0f;
  for (unsigned i = 0; i < QP_NZ; ++i) u -= QP_K[i] * z[i];
  if (!isfinite(u)) return NAN;
  u = clampValue(u, -QP_ACCEL_LIMIT, QP_ACCEL_LIMIT);
  float predicted[QP_NZ] = {};
  for (unsigned i = 0; i < QP_NZ; ++i) {
    predicted[i] = QP_B[i] * u;
    for (unsigned j = 0; j < QP_NZ; ++j)
      predicted[i] += QP_A[i * QP_NZ + j] * z[j];
    if (!isfinite(predicted[i])) return NAN;
  }
  memcpy(z, predicted, sizeof(z));
  return u;
}

void controlTick() {
  const uint32_t start = micros();
  ++ticks;
  uint16_t raw[4] = {};
  bool sensorOK = true;
  for (unsigned i = 0; i < 4; ++i) {
    bool valid = readEncoder(kCS[i], raw[i]);
    sensorOK = sensorOK && valid;
  }
  if (requestDisarm) {
    requestDisarm = false;
    requestArm = false;
    stopController(OK);
  }
  if (requestCalibration) {
    requestCalibration = false;
    requestArm = false;
    stopController(OK);
    calibrated = sensorOK;  // operator holds all links upright in the jig
    if (sensorOK) memcpy(zeroCounts, raw, sizeof(raw));
  }
  float theta[4] = {}, sum = 0.0f;
  bool nearUpright = true, withinLimits = true;
  for (unsigned i = 0; i < 4; ++i) {
    int count = (int(raw[i]) - int(zeroCounts[i]) + 8192 + 16384) % 16384 - 8192;
    sum += kEncoderSign[i] * count * kRadiansPerCount;
    theta[i] = sum;  // absolute tilt = cumulative relative encoder angles
    nearUpright = nearUpright && fabsf(sum) <= kArmAngle;
    withinLimits = withinLimits && fabsf(sum) <= kTripAngle;
  }
  float x = NAN;
  bool cartOK = readCartPositionMeters(x) && isfinite(x);
  const float reference = requestedReference;
  if (requestArm) {
    requestArm = false;
    if (!armed) { armPending = true; uprightTicks = 0; }
  }
  if (!sensorOK) stopController(SENSOR);
  else if (!cartOK) stopController(CART);
  else if (!withinLimits || fabsf(x) > QP_RAIL_LIMIT) stopController(LIMIT);
  else if (!calibrated) stopController(OK);
  else {
    if (armPending && !armed) {
      uprightTicks = nearUpright ? uprightTicks + 1 : 0;
      if (uprightTicks >= kHoldTicks) {
        resetObserver(x, theta, reference);
        armPending = false;
        armed = true;
        fault = OK;
      }
    }
  }
  float applied = 0.0f;
  if (armed) {
    const float u = controllerStep(x, theta, reference);
    if (!isfinite(u)) stopController(NUMERIC);
    else {
      // Same delay queue as Simulation, separate from the observer's in-flight states.
      commandQueue[QP_COMPUTE_DELAY_STEPS] = u;
      applied = commandQueue[0];
      for (unsigned i = 0; i < QP_COMPUTE_DELAY_STEPS; ++i)
        commandQueue[i] = commandQueue[i + 1];
      velocitySetpoint = clampValue(velocitySetpoint + applied * QP_DT,
                                    -QP_VELOCITY_LIMIT, QP_VELOCITY_LIMIT);
      if (!sendDriveVelocity(velocitySetpoint, true)) stopController(DRIVE);
    }
  } else sendDriveVelocity(0.0f, false);
  const uint32_t elapsed = micros() - start;
  if (elapsed >= QP_CONTROL_US) stopController(OVERRUN);
  if (!armed) applied = 0.0f;
  if (ticks % kTelemetryTicks == 0) {
    telemetry.ticks = ticks;
    telemetry.x = x;
    telemetry.ref = reference;
    for (unsigned i = 0; i < 4; ++i) telemetry.theta[i] = theta[i];
    telemetry.accel = applied;
    telemetry.velocity = velocitySetpoint;
    telemetry.runtime_us = elapsed;
    telemetry.armed = armed;
    telemetry.calibrated = calibrated;
    telemetry.fault = fault;
  }
}

void handleSerialLine(char *line) {
  if (!strcmp(line, "a")) requestArm = true;
  else if (!strcmp(line, "d")) requestDisarm = true;
  else if (!strcmp(line, "c")) requestCalibration = true;
  else if (line[0] == 'r' && line[1] == ' ') {
    char *end = nullptr;
    float value = strtof(line + 2, &end);
    if (end != line + 2 && *end == '\0' && isfinite(value))
      requestedReference = clampValue(value, -.8f * QP_RAIL_LIMIT, .8f * QP_RAIL_LIMIT);
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(kCalibrateButton, INPUT_PULLUP);
  for (unsigned i = 0; i < 4; ++i) {
    pinMode(kCS[i], OUTPUT);
    digitalWrite(kCS[i], HIGH);
  }
  SPI.begin();  // MOSI=11, MISO=12, SCK=13, separate CS=10/9/8/7
  sendDriveVelocity(0.0f, false);
  Serial.println("ticks,x_m,ref_m,t1_rad,t2_rad,t3_rad,t4_rad,u_mps2,vsp_mps,us,armed,cal,fault");
  if (!controlTimer.begin(controlTick, QP_CONTROL_US)) {
    sendDriveVelocity(0.0f, false);
    Serial.println("ERROR: IntervalTimer allocation failed");
  } else controlTimer.priority(128);
}

void loop() {
  static char line[48];
  static unsigned used = 0;
  static bool overflow = false;
  while (Serial.available()) {
    char c = char(Serial.read());
    if (c == '\r') continue;
    if (c == '\n') {
      line[used] = '\0';
      if (!overflow) handleSerialLine(line);
      used = 0;
      overflow = false;
    } else if (used < sizeof(line) - 1) line[used++] = c;
    else overflow = true;
  }
  static bool lastButton = HIGH;
  static uint32_t lastEdge = 0;
  bool button = digitalRead(kCalibrateButton);
  if (button != lastButton && millis() - lastEdge >= 50) {
    lastButton = button;
    lastEdge = millis();
    if (button == LOW) requestCalibration = true;
  }
  static uint32_t printedTick = 0;
  Telemetry sample;
  noInterrupts();
  sample.ticks = telemetry.ticks;
  sample.x = telemetry.x;
  sample.ref = telemetry.ref;
  for (unsigned i = 0; i < 4; ++i) sample.theta[i] = telemetry.theta[i];
  sample.accel = telemetry.accel;
  sample.velocity = telemetry.velocity;
  sample.runtime_us = telemetry.runtime_us;
  sample.armed = telemetry.armed;
  sample.calibrated = telemetry.calibrated;
  sample.fault = telemetry.fault;
  interrupts();
  if (sample.ticks != printedTick) {
    printedTick = sample.ticks;
    char buffer[240];
    int n = snprintf(buffer, sizeof(buffer), "%lu,%.5f,%.4f,%.5f,%.5f,%.5f,%.5f,%.3f,%.3f,%lu,%u,%u,%u\n",
                     (unsigned long)sample.ticks, sample.x, sample.ref,
                     sample.theta[0], sample.theta[1], sample.theta[2], sample.theta[3],
                     sample.accel, sample.velocity, (unsigned long)sample.runtime_us,
                     unsigned(sample.armed), unsigned(sample.calibrated), unsigned(sample.fault));
    if (n > 0 && n < int(sizeof(buffer)) && Serial.availableForWrite() >= n)
      Serial.write(reinterpret_cast<const uint8_t *>(buffer), size_t(n));
  }
}
