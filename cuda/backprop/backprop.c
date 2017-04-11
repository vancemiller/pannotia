/*
 ******************************************************************
 * HISTORY
 * 15-Oct-94  Jeff Shufelt (js), Carnegie Mellon University
 *  Prepared for 15-681, Fall 1994.
 * Modified by Shuai Che
 ******************************************************************
 */
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "backprop.h"














void bpnn_feedforward(BPNN* net) {
  int in = net->input_n;
  int hid = net->hidden_n;
  int out = net->output_n;

  /*** Feed forward input activations. ***/
  bpnn_layerforward(net->input_units, net->hidden_units,
                    net->input_weights, in, hid);
  bpnn_layerforward(net->hidden_units, net->output_units,
                    net->hidden_weights, hid, out);
}

void bpnn_train(BPNN* net, float* eo, float* eh) {
  int in = net->input_n;
  int hid = net->hidden_n;
  int out = net->output_n;

  /*** Feed forward input activations. ***/
  bpnn_layerforward(net->input_units, net->hidden_units,
                    net->input_weights, in, hid);
  bpnn_layerforward(net->hidden_units, net->output_units,
                    net->hidden_weights, hid, out);

  /*** Compute error on output and hidden units. ***/
  float out_err = bpnn_output_error(net->output_delta, net->target, net->output_units, out);
  float hid_err = bpnn_hidden_error(net->hidden_delta, hid, net->output_delta, out,
                    net->hidden_weights, net->hidden_units);
  *eo = out_err;
  *eh = hid_err;

  /*** Adjust input and hidden weights. ***/
  bpnn_adjust_weights(net->output_delta, out, net->hidden_units, hid,
                      net->hidden_weights, net->hidden_prev_weights);
  bpnn_adjust_weights(net->hidden_delta, hid, net->input_units, in,
                      net->input_weights, net->input_prev_weights);
}

void bpnn_save(BPNN* net, char* filename) {
  FILE* pFile = fopen(filename, "w+");
  if (!pFile) {
    fprintf(stderr, "BPNN_SAVE: Cannot create %s. %s\n", filename, strerror(errno));
    exit(EXIT_FAILURE);
  }
  printf("Saving %dx%dx%d network to '%s'\n", net->input_n, net->hidden_n, net->output_n, filename);
  fprintf(pFile, "%d %d %d", net->input_n, net->hidden_n, net->output_n);
  for (int i = 0; i <= net->input_n; i++) {
    for (int j = 0; j <= net->hidden_n; j++) {
      fprintf(pFile, "%f ", net->input_weights[i][j]);
    }
  }
  for (int i = 0; i <= net->hidden_n; i++) {
    for (int j = 0; j <= net->output_n; j++) {
      fprintf(pFile, "%f ", net->hidden_weights[i][j]);
    }
  }
  fclose(pFile);
  return;
}


BPNN* bpnn_read(char* filename) {
  FILE* fp = fopen(filename, "r");
  if (!fp) {
    fprintf(stderr, "BPNN_READ: couldn't open %s. %s\n", filename, strerror(errno));
    exit(EXIT_FAILURE);
  }
  printf("Reading '%s'\n", filename);

  int input_n;
  int hidden_n;
  int output_n;
  int ret = fscanf(fp, "%d %d %d", &input_n, &hidden_n, &output_n);
  if (ret != 3) {
    fprintf(stderr, "Invalid file format. %s\n", strerror(errno));
    exit(EXIT_FAILURE);
  }

  BPNN* new = bpnn_internal_create(input_n, hidden_n, output_n);

  printf("'%s' contains a %dx%dx%d network\n", filename, input_n, hidden_n, output_n);
  printf("Reading input weights...");

  for (int i = 0; i <= input_n; i++) {
    for (int j = 0; j <= hidden_n; j++) {
      int ret = fscanf(fp, "%f ", &new->input_weights[i][j]);
      if (!ret) {
        fprintf(stderr, "Invalid file format. %s\n", strerror(errno));
        exit(EXIT_FAILURE);
      }
    }
  }
  printf("Done\nReading hidden weights...");
  for (int i = 0; i <= hidden_n; i++) {
    for (int j = 0; j <= output_n; j++) {
      int ret = fscanf(fp, "%f ", &new->hidden_weights[i][j]);
      if (!ret) {
        fprintf(stderr, "Invalid file format. %s\n", strerror(errno));
        exit(EXIT_FAILURE);
      }
    }
  }
  fclose(fp);
  printf("Done\n");
  bpnn_zero_weights(new->input_prev_weights, input_n, hidden_n);
  bpnn_zero_weights(new->hidden_prev_weights, hidden_n, output_n);
  return new;
}

