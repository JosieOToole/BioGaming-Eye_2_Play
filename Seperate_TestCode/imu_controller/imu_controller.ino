/********* 
  Uses head tilt for WASD and a head lift for jumping
  Quick forward tilt from neutral starts sprint until returning to neutral

  Forward jumps tap Ctrl for sprint before holding Space for 200 ms
  Keeps forward movement during a jump if W was already held

  Uses a Leonardo and BNO085 with the SparkFun BNO080 library

  m turns controls on/off, x stops, z zeros the head position
  p pauses printed readings, j turns jump detection on/off, h shows commands
********/
#include <Wire.h>
#include <SparkFun_BNO080_Arduino_Library.h>
#include <Keyboard.h>
#include <math.h>

const uint8_t BUTTON_PIN = 7;
const int FORWARD_SIGN = 1, SIDE_SIGN = 1;
const char MOVE_KYS[4] = {'w','s','a','d'};
const float START_ANGLE[4] = {8,8,4,6};
const float RELEASE_ANGLE[4] = {4,5,2.5f,4};
const int8_t NO_DIR = -1;
const unsigned long DATA_TIMEOUT_MS=300, START_CNTDOWN_MS=3000, DIR_WAIT_MS=120, PRINT_INTERVAL_MS=500;

// Prevents jump from going backwards
const unsigned long BACKWARD_WAIT_MS=400;

// Acceleration is in m/s^2. These are initial tuning values
const float JMP_START_ACCEL=1.5f, JMP_SLOW_ACCEL=0.6f, STILL_ACCEL_LIMIT=0.45f;
const float MIN_VERTICAL_FRAC=0.65f;
const unsigned long MIN_JMP_DETECT_MS=40, MAX_JMP_DETECT_MS=450;
const unsigned long JMP_COOLDOWN_MS=650, STILL_WAIT_MS=160, SPACE_HOLD_MS=200;

const int VERTICAL_SIGN=1;

// Keep W through the jump  and for 550 ms after the Space press
const unsigned long JMP_AST_MS=550;
const unsigned long SPRINT_JMP_TAP_MS=30;

// A quick tilt from neutral starts sprint; a slow tilt still walks.
const unsigned long FLICK_MAX_MS=220, FLICK_NEUTRAL_MS=400;
const unsigned long FLICK_JMP_BLOCK_MS=650;
// Sprint needs both a larger tilt and enough speed; ordinary movement stays at 8 degrees.
const float FLICK_START_ANGLE=18.0f, FLICK_MIN_SPEED=85.0f;
const float FLICK_SIDE_LIMIT=6.0f;
float flickNeutralTilt=0,flickStartTilt=0;
unsigned long flickLastNeutralTime=0;
bool flickSprintOn=false,flickReady=false,flickMoving=false,flickPending=false;
bool flickNeutralTiming=false;
unsigned long flickNeutralTime=0,flickStartTime=0;

bool sprintKyHeld=false;
unsigned long sprintJmpTapStartTime=0;
bool forwardJmpAst=false;
unsigned long jmpAstStartTime=0;
unsigned long jmpAstDuration=0;

// Releases Ctrl and marks the sprint key as no longer held
void releaseSprint() {
  Keyboard.release(KEY_LEFT_CTRL);
  sprintKyHeld=false;
}

// Releases Ctrl after the sprint tap time has passed
void sprintJmpTap() {
  if(sprintKyHeld && millis()-sprintJmpTapStartTime>=SPRINT_JMP_TAP_MS){
      releaseSprint();
  }
}

// Releases Ctrl and stops holding forward for the jump
void cancelJmpAst() {
  releaseSprint();
  forwardJmpAst=false;
}

BNO080 imu;
unsigned long orientReportCnt=0,gravReportCnt=0,accelReportCnt=0,packetErrorCnt=0;
float pitch=0,roll=0,zeroPitch=0,zeroRoll=0;
float gravX=0,gravY=0,gravZ=0;
float vertAccel=0;
bool orientReceived=false,gravReceived=false,accelReceived=false;
bool controlOn=false,startCntdown=false,printReadings=true,jmpOn=true;
bool spaceHeld=false,jmpTriggered=false;
bool rawButtonState=HIGH,stableButtonState=HIGH;
unsigned long lastOrientTime=0,lastGravTime=0,lastAccelTime=0;
unsigned long cntdownStartTime=0,dirStartTime=0,lastPrintTime=0,buttonChangeTime=0;
unsigned long jmpStartTime=0,lastJmpTime=0,spaceStartTime=0,stillStartTime=0;
bool stillTimer=false;

uint8_t jmpStage=0;
int8_t activeDir=-1,pendingDir=-1;

void resetFlickSprint();

// Keeps the angle between -180 and 180 degrees
float keep180(float angle) {
  while(angle>180){
     angle-=360;
  }
  while(angle< -180){
    angle+=360;
  }
  return angle;
}
// Checks that an orientation reading was received recently
bool orientRefresh() {
  return orientReceived && millis()-lastOrientTime<DATA_TIMEOUT_MS;
}

// Checks that gravity and acceleration readings were received recently
bool accelRefresh() {
  return gravReceived && accelReceived && millis()-lastGravTime<100 && millis()-lastAccelTime<100;
}

// Releases Space and marks it as no longer held
void releaseSpace() {
  if(spaceHeld){
      Keyboard.release(' ');
  }
  spaceHeld=false;
}

// Resets jump detection and waits for no movement again
void resetJmp() {
  jmpStage=0;
  stillTimer=false;
}

// Releases WASD and clears directions without releasing Space
void releaseMove() {
  for(uint8_t kyIdx=0;kyIdx<4;kyIdx++){
    Keyboard.release(MOVE_KYS[kyIdx]);
  }
  activeDir=pendingDir=NO_DIR;
}

// Turns controls off, releases all keys, and resets jump detection
void stopControl() {
  resetFlickSprint();
  cancelJmpAst();
  controlOn=startCntdown=false;
  releaseMove();
  releaseSpace();
  resetJmp();
  Keyboard.releaseAll();
  digitalWrite(LED_BUILTIN,LOW);
}

// Releases movement and jump keys and zeros the current head position
void centerHere() {
  resetFlickSprint();
  cancelJmpAst();
  releaseMove();
  releaseSpace();
  resetJmp();
  zeroPitch=pitch;
  zeroRoll=roll;
}

// Turns controls off or starts the countdown before turning them on
void toggleControl() {
  if(controlOn||startCntdown) {
    stopControl();
    Serial.println(F("CONTROLS OFF"));
    return;
  }
  if(!orientRefresh()) {
    Serial.println(F("No orientation data; remains OFF"));
    return;
  }
  releaseMove();
  releaseSpace();
  resetJmp();
  startCntdown=true;
  cntdownStartTime=millis();
  Serial.println(F("3 seconds: focus game and hold neutral"));
}

// Prints the command options in the Serial Monitor
void help() {
  Serial.println(F("m toggle | x stop | z center | p print | j jump | h help"));
}

// Starts orientation, gravity, and movement acceleration readings every 20 ms
void startIMUReadings() {
  imu.enableGameRotationVector(20);
  imu.enableGravity(20);
  imu.enableLinearAccelerometer(20);
}

// Chooses the direction with the strongest tilt compared to its starting angle
int8_t chooseDir(float forwardTilt,float sideTilt) {
  float dirAngles[4]={forwardTilt,-forwardTilt,-sideTilt,sideTilt};
  int8_t bestDir=-1;
  float bestScore=0;
  for(uint8_t dirIdx=0;dirIdx<4;dirIdx++) {
    float angleThreshold=(activeDir==dirIdx)?RELEASE_ANGLE[dirIdx]:START_ANGLE[dirIdx];
    if(dirAngles[dirIdx]<=angleThreshold){
          continue;
    }
    float dirScore=dirAngles[dirIdx]/START_ANGLE[dirIdx];
    if(dirScore>bestScore){
      bestScore=dirScore;
      bestDir=dirIdx;
    }
  }
  return bestDir;
}

// Changes movement keys after the chosen direction stays consistent
void updateMove(int8_t chosenDir) {
  if(chosenDir<0){
    releaseMove();
    return;
  }
  if(chosenDir==activeDir){
    pendingDir=-1;
    return;
  }
  if(activeDir>=0) {
    Keyboard.release(MOVE_KYS[activeDir]);
    activeDir=-1;
  }
  if(chosenDir!=pendingDir) {
    pendingDir=chosenDir;
    dirStartTime=millis();
  }
  unsigned long dirWaitTime=(chosenDir==1)?BACKWARD_WAIT_MS:DIR_WAIT_MS;
  if(millis()-dirStartTime>=dirWaitTime) {
    activeDir=chosenDir;
    pendingDir=-1;
    Keyboard.press(MOVE_KYS[activeDir]);
  }
}

// Clears the flick gesture and sprint when stopping or recentering.
void resetFlickSprint() {
  flickSprintOn=false;
  flickReady=flickMoving=flickPending=flickNeutralTiming=false;
  releaseSprint();
}

// Checks for a quick forward tilt and ignores vertical jump movement.
void updateFlickSprint(float forwardTilt,float sideTilt) {
  unsigned long currentTime=millis();
  if(!controlOn || !orientRefresh() || !accelRefresh()) {
    resetFlickSprint();
    return;
  }
  bool jumpBlocked=jmpStage==2 || spaceHeld || forwardJmpAst ||
    (jmpTriggered && currentTime-lastJmpTime<FLICK_JMP_BLOCK_MS);
  if(jumpBlocked) {
    flickReady=flickMoving=flickPending=flickNeutralTiming=false;
    return;
  }
  // Use the normal movement dead zone so small backward drift stays neutral.
  bool nearNeutral=chooseDir(forwardTilt,sideTilt)==NO_DIR;
  if(nearNeutral) {
    flickNeutralTilt=forwardTilt;
    flickLastNeutralTime=lastOrientTime;
    if(flickSprintOn) {
      releaseSprint();
      if(Serial)Serial.println(F("FLICK SPRINT OFF: neutral"));
    }
    flickSprintOn=false;
    flickMoving=flickPending=false;
    if(!flickNeutralTiming) {
      flickNeutralTiming=true;
      flickNeutralTime=currentTime;
    }
    if(!flickReady && currentTime-flickNeutralTime>=FLICK_NEUTRAL_MS) {
      flickReady=true;
      if(Serial)Serial.println(F("FLICK READY"));
    }
    return;
  }
  flickNeutralTiming=false;
  // Changing to backward or sideways movement ends the sprint too.
  if(forwardTilt<=RELEASE_ANGLE[0] || chooseDir(forwardTilt,sideTilt)!=0) {
    if(flickSprintOn)releaseSprint();
    flickSprintOn=false;
  }
  if(flickSprintOn)return;
  if(flickReady && !flickMoving && !flickPending) {
    flickReady=false;
    if(forwardTilt>RELEASE_ANGLE[0] && fabs(sideTilt)<FLICK_SIDE_LIMIT) {
      flickMoving=true;
      flickStartTime=flickLastNeutralTime;
      flickStartTilt=flickNeutralTilt;
    }
  }
  if(flickMoving) {
    if(currentTime-flickStartTime>FLICK_MAX_MS ||
       forwardTilt<=RELEASE_ANGLE[0] || fabs(sideTilt)>=FLICK_SIDE_LIMIT) {
      flickMoving=false;
    } else if(forwardTilt>=FLICK_START_ANGLE && chooseDir(forwardTilt,sideTilt)==0 &&
              lastOrientTime-flickStartTime>0 &&
              (forwardTilt-flickStartTilt)*1000.0f/(lastOrientTime-flickStartTime)>=FLICK_MIN_SPEED) {
      if(Serial) {
        Serial.print(F("FLICK QUALIFIED: angle "));
        Serial.print(forwardTilt,1);
        Serial.print(F(" speed "));
        Serial.println((forwardTilt-flickStartTilt)*1000.0f/(lastOrientTime-flickStartTime));
      }
      flickMoving=false;
      flickPending=true;
    }
  }
  // Start as soon as W is active; returning to neutral cancels the pending flick.
  if(flickPending) {
    if(forwardTilt<=RELEASE_ANGLE[0] || chooseDir(forwardTilt,sideTilt)!=0) {
      flickPending=false;
    } else if(activeDir==0) {
      flickPending=false;
      flickSprintOn=true;
      if(Serial)Serial.println(F("FLICK SPRINT ON"));
    }
  }
}

// Holds sprint during forward movement, but never overlaps Ctrl and Space.
void holdFlickSprint() {
  if(flickSprintOn && activeDir==0 && !spaceHeld && jmpStage!=2 && !forwardJmpAst) {
    Keyboard.press(KEY_LEFT_CTRL);
  }
}

// Detects a head lift and presses Space as the upward movement slows
void detectJmp(float upAccel,float totalAccel) {
  unsigned long currentTime=millis();
  if(!controlOn||!jmpOn||!orientRefresh()||!accelRefresh()) {
    resetJmp();return;
  }
  if(jmpTriggered && currentTime-lastJmpTime<JMP_COOLDOWN_MS) {
    resetJmp();return;
  }
  if(jmpStage==0) {
    if(totalAccel<STILL_ACCEL_LIMIT) {
      if(!stillTimer){
        stillTimer=true;
        stillStartTime=currentTime;
      }
      if(currentTime-stillStartTime>=STILL_WAIT_MS){
        jmpStage=1;
        stillTimer=false;
      }
    } else {
      stillTimer=false;
    }
  } else if(jmpStage==1) {
    bool mostVerticalDir=totalAccel>0 && fabs(upAccel)/totalAccel>=MIN_VERTICAL_FRAC;
    if(upAccel>=JMP_START_ACCEL && mostVerticalDir) {
      jmpStage=2;
      jmpStartTime=currentTime;

      // Keeps forward movement only if W was held before the jump
      if(activeDir==0) {
        // Taps Ctrl when the jump starts while continuing to read the sensor and releases Ctrl before pressing Space
        Keyboard.press(KEY_LEFT_CTRL);
        if(Serial)Serial.println(F("JUMP SPRINT TAP"));
        sprintKyHeld=true;
        sprintJmpTapStartTime=currentTime;
        forwardJmpAst=true;
        jmpAstStartTime=currentTime;
        jmpAstDuration=MAX_JMP_DETECT_MS;
        pendingDir=NO_DIR;
      }
    } else if(totalAccel>=JMP_START_ACCEL) {
      resetJmp(); // Ignores sideways or downward bursts and waits for no movement again
    }
  } else {
    unsigned long jmpElapsedTime=currentTime-jmpStartTime;
    if(jmpElapsedTime>MAX_JMP_DETECT_MS){
      cancelJmpAst();
      resetJmp();
      return;
    }

    // Presses Space as the upward movement slows without requiring the head to return to neutral
    if(jmpElapsedTime>=MIN_JMP_DETECT_MS && upAccel<=-JMP_SLOW_ACCEL) {
      releaseSprint(); // Makes sure Ctrl is released before pressing Space
      Keyboard.press(' ');
      spaceHeld=true;spaceStartTime=currentTime;
      if(forwardJmpAst) {
        jmpAstStartTime=currentTime;
        jmpAstDuration=JMP_AST_MS;
      }
      lastJmpTime=currentTime;
      jmpTriggered=true;resetJmp();
      if(Serial){
        Serial.println(F("JUMP"));
      }
    }
  }
}
// Converts two sensor bytes into a signed and scaled measurement
float readSensorVal(const uint8_t *sensorData,uint8_t byteIdx,float scaleFactor) {
  uint16_t rawValue=(uint16_t)sensorData[byteIdx] | ((uint16_t)sensorData[byteIdx+1]<<8);
  int32_t signedValue=rawValue;
  if(rawValue & 0x8000){
    signedValue-=65536L;
  }
  return signedValue/scaleFactor;
}

void processIMUData() {
  // Reads all sensor records in the received packet and the library reads the packet but only processes its first record
  uint16_t packetLen=(uint16_t)imu.shtpHeader[0] |
    ((uint16_t)imu.shtpHeader[1]<<8);
  // Checks the packet header and size before reading its data
  if((packetLen & 0x8000) || packetLen<9 || imu.shtpHeader[2]!=3 || imu.shtpData[0]!=0xFB) {
    packetErrorCnt++;
    return;
  }
  uint16_t packetDataLen=packetLen-4;
  if(packetDataLen>sizeof(imu.shtpData)){
    packetErrorCnt++;
    return;
  }
  uint16_t byteIdx=5; // Skips the starting timestamp
  bool newAccelReceived=false;
  float latestAccelX=0,latestAccelY=0,latestAccelZ=0;
  while(byteIdx<packetDataLen) {
    uint8_t recordID=imu.shtpData[byteIdx];
    uint8_t recordLen=0;
    if(recordID==SENSOR_REPORTID_GAME_ROTATION_VECTOR){
      recordLen=12;
    }
    else if(recordID==SENSOR_REPORTID_GRAVITY || recordID==SENSOR_REPORTID_LINEAR_ACCELERATION){
      recordLen=10;
    }
    else if(recordID==0xFB){
      recordLen=5; // Another base timestamp
    }
    else if(recordID==0xFA){
      recordLen=2; // Timestamp rebase
    }
    else {
      packetErrorCnt++;
      break; // Stops if the record type is unknown or the record is incomplete
    } 
    if(byteIdx+recordLen>packetDataLen){
      packetErrorCnt++;
      break;
    }
    const uint8_t *recordData=&imu.shtpData[byteIdx];
    if(recordID==SENSOR_REPORTID_GAME_ROTATION_VECTOR) {
      float qi=readSensorVal(recordData,4,16384.0f);
      float qj=readSensorVal(recordData,6,16384.0f);
      float qk=readSensorVal(recordData,8,16384.0f);
      float qr=readSensorVal(recordData,10,16384.0f);
      float quatMag=sqrt(qi*qi+qj*qj+qk*qk+qr*qr);
      if(isfinite(quatMag)&&quatMag>.5f&&quatMag<1.5f) {
        qi/=quatMag;
        qj/=quatMag;
        qk/=quatMag;
        qr/=quatMag;
        float pitchSine=2*(qr*qj-qk*qi);
        if(pitchSine>1){
          pitchSine=1;
        }
        if(pitchSine< -1){
          pitchSine=-1;
        }
        pitch=asin(pitchSine)*57.29578f;
        roll=atan2(2*(qr*qi+qj*qk),1-2*(qi*qi+qj*qj))*57.29578f;
        orientReceived=true;
        lastOrientTime=millis();
        orientReportCnt++;
      }
    } else if(recordID==SENSOR_REPORTID_GRAVITY) {
      float gravXRead=readSensorVal(recordData,4,256.0f);
      float gravYRead=readSensorVal(recordData,6,256.0f);
      float gravZRead=readSensorVal(recordData,8,256.0f);
      float gravMag=sqrt(gravXRead*gravXRead+gravYRead*gravYRead+gravZRead*gravZRead);
      if(isfinite(gravMag)&&gravMag>7&&gravMag<12) {
        gravX=gravXRead/gravMag;
        gravY=gravYRead/gravMag;
        gravZ=gravZRead/gravMag;
        gravReceived=true;
        lastGravTime=millis();
        gravReportCnt++;
      }
    } else if(recordID==SENSOR_REPORTID_LINEAR_ACCELERATION) {
      latestAccelX=readSensorVal(recordData,4,256.0f);
      latestAccelY=readSensorVal(recordData,6,256.0f);
      latestAccelZ=readSensorVal(recordData,8,256.0f);
      newAccelReceived=true;
      accelReceived=true;
      lastAccelTime=millis();
      accelReportCnt++;
    }
    byteIdx+=recordLen;
  }
  // Uses the latest acceleration after updating gravity and head angles and uses the latest sample if the packet has multiple acceleration samples
  if(newAccelReceived && accelRefresh()) {
    vertAccel=VERTICAL_SIGN*(latestAccelX*gravX+latestAccelY*gravY+latestAccelZ*gravZ);
    float totalAccel=sqrt(latestAccelX*latestAccelX+latestAccelY*latestAccelY+latestAccelZ*latestAccelZ);
    // Let the jump detector recognize an upward lift before blocking a flick.
    // A negative acceleration spike alone does not count as a jump.
    detectJmp(vertAccel,totalAccel);
    if(jmpStage==2 || spaceHeld || forwardJmpAst) {
      flickReady=flickMoving=flickPending=flickNeutralTiming=false;
    }
  }
}

// Sets up the pins, USB keyboard, and IMU with controls off
void setup() {
  pinMode(BUTTON_PIN,INPUT_PULLUP);
  pinMode(LED_BUILTIN,OUTPUT);
  Serial.begin(115200);
  delay(3000);
  Keyboard.begin();
  stopControl();
  Wire.begin();
  Wire.setClock(100000);

  // Checks both possible IMU addresses
  if(!imu.begin(0x4A,Wire)&&!imu.begin(0x4B,Wire)) {
    // Keeps controls off if the IMU is not found
    while(true) {
      stopControl();
      if(Serial){
        Serial.println(F("BNO085 not found"));
      }
      delay(1000);
    }
  }
  Wire.setClock(100000);
  startIMUReadings();
  help();
}

// Runs commands, sensor readings, movement, and printed updates
void loop() {
  sprintJmpTap();

  // Releases Space when its hold time ends without waiting for sensor data
  if(spaceHeld && millis()-spaceStartTime>=SPACE_HOLD_MS){
    releaseSpace();
  }

  // Reads commands from the Serial Monitor
  while(Serial.available()) {
    char command=Serial.read();
    if(command=='m'){
      toggleControl();
    }
    else if(command=='x'){
      stopControl();
      Serial.println(F("CONTROLS OFF"));
    }
    else if(command=='z') {
      if(orientRefresh()){
        centerHere();
        Serial.println(F("Centered"));
      }
      else {
        stopControl();
        Serial.println(F("No fresh data; OFF"));
      }
    } else if(command=='p') {
      printReadings=!printReadings;
      Serial.println(printReadings?F("Readings resumed"):F("Readings paused"));
    } else if(command=='j') {
      jmpOn=!jmpOn;
      cancelJmpAst();
      releaseSpace();
      resetJmp();
      Serial.println(jmpOn?F("Jump ON"):F("Jump OFF"));
    } else if(command=='h'){
      help();
    }
  }
  bool buttonReading=digitalRead(BUTTON_PIN);
  if(buttonReading!=rawButtonState){
    rawButtonState=buttonReading;
    buttonChangeTime=millis();
  }

  // Checks that the button reading stays steady before accepting a press
  if(millis()-buttonChangeTime>=40 && stableButtonState!=rawButtonState) {
    stableButtonState=rawButtonState;
    if(stableButtonState==LOW){
      toggleControl();
    }
  }

  // Reads the next IMU packet
  uint16_t recordID=imu.getReadings();

  // Turns controls off if the IMU resets or orientation readings stop
  if(imu.hasReset()) {
    stopControl();
    orientReceived=gravReceived=accelReceived=false;
    startIMUReadings();
    if(Serial){
      Serial.println(F("IMU reset; OFF"));
    }
    return;
  }
  if(recordID==SENSOR_REPORTID_GAME_ROTATION_VECTOR || recordID==SENSOR_REPORTID_GRAVITY || recordID==SENSOR_REPORTID_LINEAR_ACCELERATION) {
    processIMUData();
  }
  if((controlOn||startCntdown)&&!orientRefresh()) {
    stopControl();
    if(Serial){
      Serial.println(F("Orientation timeout; OFF"));
    }
  }

  // Cancels jump assistance if acceleration readings stop
  if(!accelRefresh()){
    resetFlickSprint();
    cancelJmpAst();
    releaseSpace();
    resetJmp();
  }
  
  // Zeros the head position when the startup countdown ends
  if(startCntdown && millis()-cntdownStartTime>=START_CNTDOWN_MS && orientRefresh()) {
    centerHere();
    startCntdown=false;
    controlOn=true;
    if(Serial){
      Serial.println(F("CONTROLS ON; neutral captured"));
    }
  }

  digitalWrite(LED_BUILTIN,controlOn?HIGH:LOW);
  if(!orientRefresh()){
      return;
  }
  float forwardTilt=keep180(pitch-zeroPitch)*FORWARD_SIGN;
  float sideTilt=keep180(roll-zeroRoll)*SIDE_SIGN;
  if(forwardJmpAst && millis()-jmpAstStartTime>=jmpAstDuration) {
    cancelJmpAst();
  }

  if(controlOn) {
    if(forwardJmpAst) {
      // Keeps W held during jump assistance
      updateMove(0);
    } else {
      updateMove(chooseDir(forwardTilt,sideTilt));
    }
  }

  updateFlickSprint(forwardTilt,sideTilt);
  holdFlickSprint();

  // Prints sensor readings and control status
  if(printReadings && Serial && millis()-lastPrintTime>=PRINT_INTERVAL_MS && Serial.availableForWrite()>60) {
    lastPrintTime=millis();
    Serial.print(F("reports O/G/A "));
    Serial.print(orientReportCnt);
    Serial.print('/');Serial.print(gravReportCnt);
    Serial.print('/');Serial.println(accelReportCnt);
    Serial.print(F("packet errors "));
    Serial.println(packetErrorCnt);
    Serial.print(F("f "));
    Serial.print(forwardTilt,1);
    Serial.print(F(" side "));
    Serial.print(sideTilt,1);
    Serial.print(F(" up "));
    Serial.print(vertAccel,2);
    Serial.print(F(" stage "));
    Serial.print(jmpStage);
    Serial.print(forwardJmpAst?F(" assistW"):F(" normal"));
    Serial.print(flickSprintOn?F(" sprint"):F(" walk"));
    Serial.print(F(" key "));
    Serial.print(activeDir>=0?MOVE_KYS[activeDir]:'-');
    Serial.print(accelRefresh()?F(" accelOK"):F(" accelLOST"));
    Serial.println(controlOn?F(" ON"):(startCntdown?F(" ARMING"):F(" OFF")));
  }
}
