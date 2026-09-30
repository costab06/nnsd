#include <stdio.h>
#include <stdlib.h>


int main(int argc, char **argv) {

  int layers = 3;
  float **weights[8];
  int neurons[8];
  char *fileName = "/tmp/foo.test";


  neurons[0] = 2;
  neurons[1] = 5;
  neurons[2] = 1;
  
  // Allocate the weights


  int i; for (i = 0; i < layers -1; i++) {
    weights[i] = (float **)calloc(neurons[i],sizeof (float *));
    int j; for (j = 0; j < neurons[i]; j++) {
      weights[i][j] = (float *)calloc(neurons[i + 1],sizeof (float));
    }
  }
  
  
  // Free the weights
  
  for (i = 0; i < layers - 1; i++) {
    int j; for (j=0;j<neurons[i];j++)
	     if (weights[i][j] != NULL) {
	       free(weights[i][j]);
	     }
    free(weights[i]);
  }




  // open the file
  FILE * fileHdl = fopen(fileName,"wt");

  int retValue = fclose(fileHdl);




}
