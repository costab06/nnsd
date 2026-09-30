#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define SUCCESS 0
#define FAILURE 1
#define MAX_LAYERS 8


int writeWeights(FILE *mfp,long layers,long *neurons,long thresholds)
{
int i,j,k;
float temp=.5;

for (i=0;i<layers-1;i++)
  {
    for (j=0;j<neurons[i] + thresholds;j++)
      {
	for (k=0;k<neurons[i+1];k++)
	  {
	    if (fwrite(&temp,sizeof(float),1,mfp) != 1)
	      {
		printf("ERROR writing weights\n");
		return FAILURE;
	      }
	  }
      }
  }
  return SUCCESS;
}





int main (int argc,char *argv[])
{
int i;
char *trnfile;
long layers;
long *neurons;
float learn;
float momentum;
float trainingTolerance;
long thresholds;
FILE *fp;
  
char *mtxfile;




  if (argc != 5)
    {
      printf("usage: %s <train file> <input neurons> <output neurons> <layers> \n",argv[0]);
      exit(1);
    }
  


trnfile=(char *)strdup(argv[1]);
layers=atoi(argv[4]);
learn=.1;
momentum=.1;
trainingTolerance=0.1;
thresholds=1;
mtxfile=(char *)strdup(trnfile);
strcpy(&(mtxfile[strlen(mtxfile)-3]),"mtx");

printf("trnfile = %s, mtxfile = %s\n",trnfile,mtxfile);

if ((fp = fopen(mtxfile,"w")) == NULL)
  {
    printf("ERROR: cannot open %s\n",mtxfile);
    exit(1);
  }

neurons=(long *)malloc(sizeof(long)*MAX_LAYERS);
// input layer
neurons[0]=atoi(argv[2]);
// output layer
neurons[layers-1]=atoi(argv[3]);
// middle layers
for (i=1;i<layers-1;i++)
  neurons[i]=(long)(((neurons[0]+neurons[layers-1])/2)+1); 
// layers output to MAX_LAYERS
for (i=layers;i<MAX_LAYERS;i++)
  neurons[i]=0;

fwrite(&learn,sizeof(learn),1,fp);
fwrite(&momentum,sizeof(momentum),1,fp);
fwrite(&trainingTolerance,sizeof(trainingTolerance),1,fp);
fwrite(&thresholds,sizeof(thresholds),1,fp);
fwrite(&layers,sizeof(layers),1,fp);

for (i=0;i<MAX_LAYERS;i++)
  fwrite(&(neurons[i]),sizeof(long),1,fp);


writeWeights(fp,layers,neurons,thresholds);

fclose(fp);
return 0;
}


















