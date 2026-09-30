#include <stdio.h>
#include <errno.h>
#include <stdlib.h>

#define SUCCESS 0
#define FOPENERROR 1
#define FREADERROR 2

#define MAX_LAYERS 8


int neurons[MAX_LAYERS];
int layers;	
float learningRate;
float momentum;
float trainingTolerance;
float trainingSuccessPercentage;
float testingTolerance;
int numberOfInputFileFacts;
float sumSquaredError;
int numberOfPasses;


float *vectors[MAX_LAYERS];
float **weights[MAX_LAYERS];

float **previousWeightChange[MAX_LAYERS];
float *deltas[MAX_LAYERS];

float *minInputArray;
float *maxInputArray;
float *minOutputArray;
float *maxOutputArray;


int main(int argc, char **argv) {
  
  if (argc != 2) {
    printf("usage: %s <infile>\n",argv[0]);
    exit(-1);
  }

  int ccode = PrintMatrixFile(argv[1]);
  printf("PrintMatrixFile returned: %d\n",ccode);
  
}


/*************************** PrintMatrixFile *****************************/
int PrintMatrixFile (char *filename) {
  
  int ccode = SUCCESS;
  
  char msg_buff[8192];
  
  char *mode="r";
  FILE *mfp;
  
  
  // open the mtxfile, read the header and allocate 
  mfp=fopen(filename, mode);
  if ( mfp == NULL) {
    
    sprintf (msg_buff, "file open error for %s, mode:%s, errno: %d, strerror(errno): %s\n",
	     filename, mode, errno, strerror(errno));
    printf (msg_buff);
    ccode = FOPENERROR;
    return ccode;
  }
  
  
  if (fread (&(learningRate), sizeof(float), 1,
	     mfp) != 1) {
    printf ("Cannot read learningRate\n");
    ccode = FREADERROR;
    return ccode;
  }
  
  if (fread (&(momentum), sizeof(float), 1,
	     mfp) != 1) {
    printf ("Cannot read Momentum\n");
    ccode = FREADERROR;
    return ccode;
  }
  
  if (fread (&(trainingTolerance), sizeof(float), 1,
	     mfp) != 1) {
    printf ("Cannot read trainingTolerance\n");
    ccode = FREADERROR;
    return ccode;
  }

  if (fread (&(trainingSuccessPercentage), sizeof(float), 1,
	     mfp) != 1) {
    printf ("Cannot read trainingSuccessPercentage\n");
    ccode = FREADERROR;
    return ccode;
  }

  if (fread (&(testingTolerance), sizeof(float), 1,
	     mfp) != 1) {
    printf ("Cannot read testingTolerance\n");
    ccode = FREADERROR;
    return ccode;
  }
  
  if (fread (&(layers), sizeof(int), 1,
	     mfp) != 1) {
    printf ("cannot read layers\n");
    ccode = FREADERROR;
    return ccode;
  }
  
  
  if (fread (&(neurons), sizeof(int)*MAX_LAYERS, 1,
	     mfp) != 1) {
    printf ("cannot read neurons\n");
    ccode = FREADERROR;
    return ccode;
  }
  
  
  sprintf (msg_buff,
	   "Read in Matrix File values: learningRate = %f\nmomentum = %f\ntrainingTolerance = %f\n trainingSuccessPercentage = %f\n testingTolerance = %f\n layers = %d\n neurons: %d %d %d %d %d %d %d %d\n\n",
	   learningRate, 
	   momentum,
	   trainingTolerance,
	   trainingSuccessPercentage,
	   testingTolerance,
	   layers,
	   neurons[0],
	   neurons[1],
	   neurons[2],
	   neurons[3],
	   neurons[4],
	   neurons[5],
	   neurons[6],
	   neurons[7]);
  printf(msg_buff);
  
  
  printf("Weights:\n");
  
  float *weightPtr = NULL;
  float weight;
  
  
  // read in the values from the matrix file into the weights  
  int i; for (i = 0; i < layers - 1; i++) {
    printf("Layer[%d]: \n",i);
    
    // Add one for the BIAS node
    int j; for (j = 0; j < (neurons[i]+1); j++) {
      printf("\t\tfrom layer[%d] neuron[%d]: \n",i,j);
      if (weightPtr != NULL)
	free(weightPtr);
      weightPtr=calloc(1,sizeof(float)*neurons[i+1]);
      if (fread (weightPtr,
		 sizeof (float),
		 (neurons[i+1]),
		 mfp) != (neurons[i+1])) {
	sprintf (msg_buff, "Error reading weights from matrix file: %s with mode %s: %d\n",
		 "matrixfile", mode, errno);
	printf(msg_buff);
      } else {
	int k; for(k=0;k<neurons[i+1];k++) {
	  printf("\t\t\tto layer[%d] neuron %d]: %f\n",i+1,k,weightPtr[k]);
	}
      }
    }
  }

    // free up the buffer
    if (weightPtr != NULL)
      free(weightPtr);
    

    
    
  // read min/max arrays
  
  // min input
  float value;
  
  printf("MinInput: \n");
  for (i = 0; i < neurons[0]; i++) {
    if (fread(&value, sizeof (float), 1,mfp) != 1) {
      return FREADERROR;
    } else {
      printf(" %f ",value);
    }
  }
  printf("\n\n");
  
  
  // min output
  printf("MinOutput: \n");
  for (i = 0;i < neurons[layers - 1]; i++) {
    if (fread(&value, sizeof (float), 1,mfp) != 1) {
      return FREADERROR;
    } else {
      printf(" %f ",value);
    }
  }
  printf("\n\n");


  
  // max input
  printf("MaxInput: \n");
  for (i = 0; i < neurons[0]; i++) {
    if (fread(&value, sizeof (float), 1,mfp) != 1) {
      return FREADERROR;
    } else {
      printf(" %f ",value);
    }
  }
  printf("\n\n");
  

  
  // max output
  printf("MaxOutput: \n");
  for (i = 0;i < neurons[layers - 1]; i++) {
    if (fread(&value, sizeof (float), 1,mfp) != 1) {
      return FREADERROR;
    } else {
      printf(" %f ",value);
    }
  }
  printf("\n\n");

  
  
  // close the matrix file
  fclose(mfp);
  
  return ccode;
}
