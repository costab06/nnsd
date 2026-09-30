#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>

#define MAXLAYERS 8

int VerifyMatrixFile (char *);

int main(int argc, char **argv)
{
  if (argc < 1)
    printf("Usage: vermtxfile matrixfilename\n");
  else
    VerifyMatrixFile(argv[1]);
  return 0;
}

int VerifyMatrixFile (char *filename)

{
  typedef struct LayerInfoStructure
  {
    float learningRate;
    float momentum;
    float trainingTolerance; 
    long thresholds;
    long layers;
    long neuronsPerLayer[MAXLAYERS];
  } LayerInfoStruct;

  LayerInfoStruct mtxInfo;

  int i,j,k;
  long unsigned int weightCount = 0;
  int thresholds = 1;
  float temp;
  FILE *mtxFile=NULL;

  if ((access(filename,0)) == 0) /*matrix files doesn't exist*/
    {
      if ((mtxFile = fopen(filename, "rb")) != NULL)
	{
	  if (fread(&mtxInfo,sizeof(LayerInfoStruct),1,mtxFile) == 1)
	    {
	      printf("LearningRate is %f\n",mtxInfo.learningRate);
	      printf("Momentum is %f\n",mtxInfo.momentum);
              printf("TrainingTolerance is %f\n",mtxInfo.trainingTolerance);
	      printf("Thresholds is %d\n",mtxInfo.thresholds);
	      printf("Layers is %d\n",mtxInfo.layers);
	      for (i=0;i<MAXLAYERS;i++)
		printf("number of neurons in %i is %d\n",i,mtxInfo.neuronsPerLayer[i]);


	      printf("weights start at file position %ld\n",ftell(mtxFile));


	      for (i=0;i<mtxInfo.layers-1;i++)
		{
		  for (j=0;j<mtxInfo.neuronsPerLayer[i+1];j++)
		    {
		      for (k=0;k<mtxInfo.neuronsPerLayer[i]+thresholds;k++)
			{
			  if (fread(&temp,sizeof(float),1,mtxFile) == 1)
			    {
			      printf("[%d,%d,%d]=%f ",i,j,k,temp);
			      weightCount++;
			    }
			  else
			    printf("Error reading weights from matrix file %s\n", filename);
			}
		    }
		  printf("\n");
		  printf("next layer\n");
		  printf("\n");
		}


	      printf("Min Inputs = ");
	      for(i=0;i<mtxInfo.neuronsPerLayer[0];i++)
		{
		  if (fread(&temp,sizeof(float),1,mtxFile) == 1)
		    printf("%f ",temp);
		  else
		    {
		      printf("ERROR: cannot read Min Inputs\n");
		      exit(1);
		    }
		}
	      printf("\n");
	      
	      printf("Min Outputs = ");
	      for(i=0;i<mtxInfo.neuronsPerLayer[mtxInfo.layers-1];i++)
		{
		  if (fread(&temp,sizeof(float),1,mtxFile) == 1)
		    printf("%f ",temp);
		  else
		    {
		      printf("ERROR: cannot read Min Outputs\n");
		      exit(1);
		    }
		}
	      printf("\n");
	      
	      printf("Max Inputs = ");
	      for(i=0;i<mtxInfo.neuronsPerLayer[0];i++)
		{
		  if (fread(&temp,sizeof(float),1,mtxFile) == 1)
		    printf("%f ",temp);
		  else
		    {
		      printf("ERROR: cannot read Max Inputs\n");
		      exit(1);
		    }
		}
	      printf("\n");
	      
	      printf("Max Outputs = ");
	      for(i=0;i<mtxInfo.neuronsPerLayer[mtxInfo.layers-1];i++)
		{
		  if (fread(&temp,sizeof(float),1,mtxFile) == 1)
		    printf("%f ",temp);
		  else
		    {
		      printf("ERROR: cannot read Max Outputs\n");
		      exit(1);
		    }
		}
	      printf("\n");
	    }
	}
      else
	{
	  printf("Error opening matrix file %s\n",filename);
	  exit(1);
	}
    }
  else
    {
      printf("Matrix file %s does not exist\n",filename);
      exit(1);
    }


  printf("WeightCount is %lu\n",weightCount);
  printf("File size should be %lu\n",
    (unsigned long)((weightCount+MAXLAYERS+5)*4)+
    (mtxInfo.neuronsPerLayer[0]*2*sizeof(float))+
    (mtxInfo.neuronsPerLayer[mtxInfo.layers-1]*2*sizeof(float)));



  if (mtxFile != NULL)
    fclose(mtxFile);
  
  return (0);
  
}



