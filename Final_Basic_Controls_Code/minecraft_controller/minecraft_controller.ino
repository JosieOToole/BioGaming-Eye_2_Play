/********* 
  Uses head tilt for WASD and a head lift for jumping
  Quick forward tilt from neutral starts sprint until returning to neutral

  Forward jumps tap Ctrl for sprint before holding Space for 200 ms
  Keeps forward movement during a jump if W was already held

  Uses a Leonardo and BNO085 with the SparkFun BNO080 library

  Laptop window: n calibrates IMU neutral; c/l/r/u/d calibrate the eyes
  g starts after a 3-second countdown; x or Escape pauses; q quits
  Eye mouse is allowed only when the IMU is settled in neutral
********/
#include <Wire.h>
#include <SparkFun_BNO080_Arduino_Library.h>
#include <Keyboard.h>
#include <Mouse.h>
#include <stdlib.h>
#include <math.h>

const uint8_t BUTTON_PIN = 7;
const int FORWARD_SIGN = -1, SIDE_SIGN = -1;
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
const float FLICK_START_ANGLE=18.0f, FLICK_MIN_SPEED=90.0f;
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

// Setup stays separate from play so starting never changes the saved neutral.
const unsigned long HOST_TIMEOUT_MS=500, GAZE_TIMEOUT_MS=250;
const unsigned long NEUTRAL_SETTLE_MS=250, STATE_INTERVAL_MS=100, MOUSE_INTERVAL_MS=20;
const int MOUSE_STEP_X=3, MOUSE_STEP_Y=3;
const float GAZE_STILL_ACCEL=0.8f;
bool imuCalibrated=false,imuCapture=false,eyeCalReady=false,hostSeen=false;
bool neutralTiming=false,neutralReady=false,eyeGate=false;
unsigned long hostTime=0,imuCaptureTime=0,neutralTime=0,stateTime=0,mouseTime=0;
unsigned long gazeTime=0,calStillTime=0;
bool calStillTiming=false;
float totalMoveAccel=0,calPreviousPitch=0,calPreviousRoll=0;
uint16_t gazeEpoch=0;
char gazeDir='S';
char commandLine[32];
uint8_t commandLength=0;
bool commandOverflow=false;

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
bool hostFresh();
void clearGaze();

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
  imuCapture=false;
  clearGaze();
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
  clearGaze();
  resetFlickSprint();
  cancelJmpAst();
  releaseMove();
  releaseSpace();
  resetJmp();
  zeroPitch=pitch;
  zeroRoll=roll;
}

// Starts play only after both calibrations, without changing the saved neutral.
void toggleControl() {
  if(controlOn||startCntdown) {
    stopControl();
    Serial.println(F("CONTROLS OFF"));
    return;
  }
  if(!hostFresh() || !imuCalibrated || !eyeCalReady || !neutralReady || imuCapture) {
    if(!hostFresh())Serial.println(F("START BLOCKED: laptop heartbeat missing"));
    else if(imuCapture)Serial.println(F("START BLOCKED: IMU calibration in progress"));
    else if(!imuCalibrated)Serial.println(F("START BLOCKED: press n to calibrate IMU"));
    else if(!eyeCalReady)Serial.println(F("START BLOCKED: eye calibration incomplete"));
    else Serial.println(F("START BLOCKED: head not settled in neutral"));
    return;
  }
  releaseMove();
  releaseSpace();
  resetJmp();
  resetFlickSprint();
  clearGaze();
  startCntdown=true;
  cntdownStartTime=millis();
  Serial.println(F("STARTING: 3 seconds to focus Minecraft"));
}

// Prints the command options in the Serial Monitor
void help() {
  Serial.println(F("Laptop: n IMU neutral | c/l/r/u/d eyes | g start | x pause | q quit"));
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
        if(imuCapture) {
          bool steady=fabs(keep180(pitch-calPreviousPitch))<0.5f &&
            fabs(keep180(roll-calPreviousRoll))<0.5f;
          if(!steady)calStillTiming=false;
          else if(!calStillTiming) {
            calStillTiming=true;
            calStillTime=millis();
          }
          calPreviousPitch=pitch;
          calPreviousRoll=roll;
        }
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
    totalMoveAccel=totalAccel;
    // Let the jump detector recognize an upward lift before blocking a flick.
    // A negative acceleration spike alone does not count as a jump.
    detectJmp(vertAccel,totalAccel);
    if(jmpStage==2 || spaceHeld || forwardJmpAst) {
      flickReady=flickMoving=flickPending=flickNeutralTiming=false;
    }
  }
}

// A missing laptop heartbeat stops every game output.
bool hostFresh() {
  return hostSeen && millis()-hostTime<HOST_TIMEOUT_MS;
}

void clearGaze() {
  gazeDir='S';
  gazeTime=0;
}

// Capture IMU neutral with every game output off.
void beginIMUCalibration() {
  stopControl();
  imuCalibrated=false;
  // IMU re-zeroing does not invalidate saved eye calibration.
  neutralReady=neutralTiming=false;
  imuCapture=true;
  imuCaptureTime=millis();
  calStillTiming=false;
  calPreviousPitch=pitch;
  calPreviousRoll=roll;
  Serial.println(F("IMU CALIBRATION: hold your normal screen-facing position for 3 seconds"));
}

void updateSetup() {
  if(imuCapture && millis()-imuCaptureTime>=START_CNTDOWN_MS) {
    if(orientRefresh() && accelRefresh() && totalMoveAccel<GAZE_STILL_ACCEL &&
       calStillTiming && millis()-calStillTime>=500) {
      centerHere();
      imuCapture=false;
      imuCalibrated=true;
      neutralTiming=false;
      Serial.println(F("IMU CALIBRATED: saved eye calibration kept; press g if eyes already calibrated"));
    } else if(millis()-imuCaptureTime>10000) {
      stopControl();
      Serial.println(F("IMU CALIBRATION FAILED: hold still and press n again"));
    }
  }
  if(startCntdown && millis()-cntdownStartTime>=START_CNTDOWN_MS) {
    if(hostFresh() && imuCalibrated && eyeCalReady && neutralReady && orientRefresh() &&
       accelRefresh() && totalMoveAccel<GAZE_STILL_ACCEL &&
       chooseDir(keep180(pitch-zeroPitch)*FORWARD_SIGN,keep180(roll-zeroRoll)*SIDE_SIGN)==NO_DIR) {
      startCntdown=false;
      controlOn=true;
      Serial.println(F("CONTROLS ON: saved calibration kept"));
    } else {
      stopControl();
      Serial.println(F("START CANCELLED: return neutral and press g"));
    }
  }
}

// Lines keep eye commands separate from IMU commands and include a neutral-session number.
void handleCommand(const char *line) {
  if(line[0]=='H' && line[1]==',' && (line[2]=='0'||line[2]=='1') && line[3]=='\0') {
    hostSeen=true;
    hostTime=millis();
    eyeCalReady=line[2]=='1';
    if(!eyeCalReady && (controlOn||startCntdown)) {
      stopControl();
      Serial.println(F("CONTROLS OFF: eye calibration incomplete"));
    }
  } else if(line[0]=='E' && line[1]==',' && line[2] && line[3]==',' && line[4]) {
    char *end;
    unsigned long epoch=strtoul(line+4,&end,10);
    char direction=line[2];
    if(*end=='\0' && epoch==gazeEpoch && eyeGate && hostFresh() &&
       (direction=='L'||direction=='R'||direction=='U'||direction=='D'||direction=='C'||direction=='S')) {
      gazeDir=direction;
      gazeTime=millis();
    }
  } else if(line[0] && line[1]=='\0') {
    char command=line[0];
    if(command=='X'||command=='x') {
      stopControl();
      Serial.println(F("CONTROLS OFF: paused by user"));
    }
    else if(command=='N')beginIMUCalibration();
    else if(command=='A'||command=='m')toggleControl();
    else if(command=='p')printReadings=!printReadings;
    else if(command=='j') {
      jmpOn=!jmpOn;
      cancelJmpAst();
      releaseSpace();
      resetJmp();
    } else if(command=='h')help();
  }
}

void readCommands() {
  while(Serial.available()) {
    char incoming=Serial.read();
    if(incoming=='\r')continue;
    if(incoming=='\n') {
      if(!commandOverflow) {
        commandLine[commandLength]='\0';
        handleCommand(commandLine);
      }
      commandLength=0;
      commandOverflow=false;
    } else if(commandLength<sizeof(commandLine)-1 && !commandOverflow) {
      commandLine[commandLength++]=incoming;
    } else commandOverflow=true;
  }
}

// Eye processing resumes only after neutral has stayed settled for 250 ms.
void updateNeutral(float forwardTilt,float sideTilt) {
  bool quiet=imuCalibrated && !imuCapture && orientRefresh() && accelRefresh() &&
    chooseDir(forwardTilt,sideTilt)==NO_DIR && activeDir==NO_DIR && pendingDir==NO_DIR &&
    !forwardJmpAst && !flickSprintOn && !flickMoving && !flickPending &&
    jmpStage!=2 && !spaceHeld && totalMoveAccel<GAZE_STILL_ACCEL;
  if(!quiet) {
    neutralReady=false;
    neutralTiming=false;
  } else {
    if(!neutralTiming) {
      neutralTiming=true;
      neutralTime=millis();
    }
    neutralReady=millis()-neutralTime>=NEUTRAL_SETTLE_MS;
  }
  bool nextEyeGate=controlOn && eyeCalReady && hostFresh() && neutralReady;
  if(nextEyeGate!=eyeGate) {
    eyeGate=nextEyeGate;
    gazeEpoch++;
    clearGaze();
    stateTime=millis()-STATE_INTERVAL_MS;
  }
  if(!eyeGate)clearGaze();
}

void updateEyeMouse() {
  if(!eyeGate || !controlOn || !neutralReady || !hostFresh() ||
     !orientRefresh() || !accelRefresh() || millis()-gazeTime>=GAZE_TIMEOUT_MS) {
    clearGaze();
    return;
  }
  if(millis()-mouseTime<MOUSE_INTERVAL_MS)return;
  mouseTime=millis();
  int dx=0,dy=0;
  if(gazeDir=='L')dx=-MOUSE_STEP_X;
  else if(gazeDir=='R')dx=MOUSE_STEP_X;
  else if(gazeDir=='U')dy=-MOUSE_STEP_Y;
  else if(gazeDir=='D')dy=MOUSE_STEP_Y;
  if(dx||dy)Mouse.move(dx,dy,0);
}

void sendControllerState() {
  if(!Serial || millis()-stateTime<STATE_INTERVAL_MS || Serial.availableForWrite()<40)return;
  stateTime=millis();
  uint8_t mode=imuCapture?1:(startCntdown?2:(controlOn?3:0));
  Serial.print(F("STATE,"));Serial.print(mode);
  Serial.print(',');Serial.print(imuCalibrated?1:0);
  Serial.print(',');Serial.print(neutralReady?1:0);
  Serial.print(',');Serial.println(gazeEpoch);
}

// Sets up the pins, USB keyboard, and IMU with controls off
void setup() {
  pinMode(BUTTON_PIN,INPUT_PULLUP);
  pinMode(LED_BUILTIN,OUTPUT);
  Serial.begin(115200);
  delay(3000);
  Keyboard.begin();
  Mouse.begin();
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

  // The laptop owns calibration, start, and pause while it owns the USB port.
  readCommands();
  if(!hostFresh() && (controlOn||startCntdown||imuCapture)) {
    stopControl();
    Serial.println(F("HOST TIMEOUT: controls off"));
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
    imuCalibrated=false;
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
  
  updateSetup();
  digitalWrite(LED_BUILTIN,controlOn?HIGH:LOW);
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
  updateNeutral(forwardTilt,sideTilt);
  updateEyeMouse();
  sendControllerState();

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
