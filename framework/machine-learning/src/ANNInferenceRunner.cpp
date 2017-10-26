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

#include "ANNInferenceRunner.h"
#include <iostream>
#include "fann.h"
#include <cstdio>

MLDataUnit* ANNInferenceRunner::run(MLDataUnit* inputData) {
  // TODO: implement it
  
  fann_type *calc_out;
  unsigned int i;
  struct fann *ann;
  struct fann_train_data *data;

  printf("Creating network.\n");

  ann = fann_create_from_file("./model/accelerometer.net");

  if(!ann)
  {
    printf("Error creating ann --- ABORTING,\n");
    return NULL;
  }

  fann_print_connections(ann);
  fann_print_parameters(ann);

  printf("Testting network.\n");

  data = fann_read_train_from_file("./data/accelerometer.data");

  for (i = 0; i < fann_length_train_data(data); i++)
  {
    fann_reset_MSE(ann);
    calc_out = fann_test(ann, data->input[i], data->output[i]);

    printf("Accelerometer test (%f, %f) -> %f, should be %d, difference=%d\n",
        data->input[i][0], data->input[i][1], calc_out[0], data->output[i][0],
        (float) fann_abs(calc_out[0] - data->output[i][0]));
  }
  fann_destroy_train(data);
  fann_destroy(ann);

  return NULL;

}

// Get resource usage of inference runner
std::string ANNInferenceRunner::getResourceUsage() {
  std::string data("");
  // TODO: implement it
  
  return data;
}
