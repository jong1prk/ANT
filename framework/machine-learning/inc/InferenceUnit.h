/* Copyright (c) 2017 SKKU ESLAB, and contributors. All rights reserved.
 *
 * Contributor: Gyeonghwan Hong<redcarrottt@gmail.com>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __INFERENCE_UNIT_H__
#define __INFERENCE_UNIT_H__

#include <pthread.h>
#include <iostream>
#include <string>
#include <vector>
#include <map>

#include "InputReaderSet.h"
#include "InferenceRunner.h"
#include "OutputNotifier.h"
#include "MLTensor.h"
#include "MLDataUnit.h"

namespace InferenceUnitState {
  enum Value {
    Initialized = 1,
    Ready = 2,
    Running = 3,
    Destroyed = 4
  };
}

namespace InferenceUnitType {
  enum Value {
    DNN = 1,
    ANN = 2,
    SVM = 3,
    KNN = 4
  };
}

class InferenceUnitOutputListener {
  public:
    virtual void onInferenceUnitOutput(int iuid, std::string listenerUri,
        MLDataUnit* outputData) = 0;
};

class InferenceUnitStateListener {
  public:
    virtual void onChangedInferenceUnitState(int iuid,
        InferenceUnitState::Value newState) = 0;
};

class InferenceUnit {
  public:
    friend class ModelPackageLoader;

    // Inference Unit Thread
    static void* inferenceLoop(void*);

    // Commands
    bool unload();
    bool start();
    bool stop();
    bool setInput(std::string inputName, std::string sourceUri);
    bool startListeningOutput(std::string listenerUri);
    bool stopListeningOutput(std::string listenerUri);
    std::string getResourceUsage();

    // Getters
    int getIuid() { return this->mIuid; }
    std::string getName() { return this->mName; }
    InferenceUnitState::Value getState() { return this->mState; }
    std::string getModelPackagePath() { return this->mModelPackagePath; }
    InferenceUnitType::Value getType() { return this->mType; }
    MLDataUnitLayout* getInputLayout() { return this->mInputLayout; }
    MLDataUnitLayout* getOutputLayout() { return this->mOutputLayout; }
    MLDataUnit* getParameters() { return this->mParameters; }
    int getPid() { return this->mPid; }
    
    // Setters
    void setIuid(int iuid) { this->mIuid = iuid; }

    // State Listener
    void setStateListener(InferenceUnitStateListener* stateListener) {
      this->mStateListener = stateListener;
    }

    // Output Listener
    void setOutputListener(InferenceUnitOutputListener* outputListener) {
      this->mOutputListener = outputListener;
    }

    ~InferenceUnit() {
      if(this->mInputLayout != NULL)
        delete this->mInputLayout;
      if(this->mOutputLayout != NULL)
        delete this->mOutputLayout;
      if(this->mParameters != NULL)
        delete this->mParameters;
      if(this->mInputReaderSet != NULL)
        delete this->mInputReaderSet;
      if(this->mInferenceRunner != NULL)
        delete this->mInferenceRunner;
    }

  protected:
    InferenceUnit(std::string name,
        std::string modelPackagePath,
        InferenceRunner* inferenceRunner,
        MLDataUnitLayout* inputLayout,
        MLDataUnitLayout* outputLayout,
        MLDataUnit* parameters)
    : mName(name),
    mState(InferenceUnitState::Initialized),
    mModelPackagePath(modelPackagePath),
    mInputLayout(inputLayout),
    mOutputLayout(outputLayout),
    mParameters(parameters),
    mPid(-1),
    mInferenceRunner(inferenceRunner),
    mIsThreadRunning(false) {
      // Initialize InputReaderSet
      this->mInputReaderSet = new InputReaderSet();

      // Initialize InputMap
      std::map<std::string, MLTensorLayout>& inputTensorMap
        = this->mInputLayout->getMap();
      std::map<std::string, MLTensorLayout>::iterator itmIter;
      for(itmIter = inputTensorMap.begin();
          itmIter != inputTensorMap.end();
          itmIter++) {
        std::string tensorName(itmIter->first);
        this->mInputMap.insert(
            std::pair<std::string, std::string>(tensorName, ""));
      }
    }

    // Check input & output connections and update state
    //   - Set "Ready" if inputs & outputs are all connected.
    //   - Set "Initialized" if one of inputs is disconnected
    //       or no output is connected.
    void checkConnectionsAndUpdateState();
    bool checkConnections();

    // State change
    void setState(InferenceUnitState::Value newState);

    // Fields determined at Initializd state //
    // Inference unit ID (IUID)
    int mIuid;
    // Inference unit name
    std::string mName;
    // Inference unit's state
    InferenceUnitState::Value mState;
    // Inference unit's model package path (it should be absolute path)
    std::string mModelPackagePath;
    InferenceUnitType::Value mType;
    // Input data's data unit layout
    MLDataUnitLayout* mInputLayout;
    // Output data's data unit layout
    MLDataUnitLayout* mOutputLayout;
    // Inference unit's parameters
    MLDataUnit* mParameters;

    // Fields which of key is determined at Initialized state,
    //   and value is determined at Initialized & Ready state
    // Input map (key: String name, value: String sourceURI)
    std::map<std::string, std::string> mInputMap;
    // Output listener URI list
    std::vector<std::string> mOutputListenerURIs;

    // Fields determined at Running state //
    // process ID (initialized as -1 at Initialized state)
    int mPid;

    // State listener //
    InferenceUnitStateListener* mStateListener = NULL;

    // Output listener //
    InferenceUnitOutputListener* mOutputListener = NULL;

    // Input Reader Set //
    InputReaderSet* mInputReaderSet = NULL;

    // Inference Runner //
    InferenceRunner* mInferenceRunner = NULL;

    // Inference Unit Thread //
    bool mIsThreadRunning;
    pthread_t mInferenceUnitThread;

    pthread_mutex_t mThreadRunningMutex = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_t mInputMutex = PTHREAD_MUTEX_INITIALIZER;
    pthread_mutex_t mOutputMutex = PTHREAD_MUTEX_INITIALIZER;
};

#endif // !defined(__INFERENCE_UNIT_H__)
