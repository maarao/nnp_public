#include <stdio.h>
#include <string.h>
#include <time.h>
#include "config.h"
#include "loader.h"
#include "nnp.h"

#define TRAIN 1
#define PREDICT 2

int parseCmd(int argc,char** argv){
	if (argc!=2) return 0;
	if (strcmp(argv[1],"train")==0)return TRAIN;
	if (strcmp(argv[1],"predict")==0) return PREDICT;
	return 0;
}

int usage(){
	printf("Usage: nnp [train|predict]\n\tNote: predict requires a previously trained model in the directory named model.bin\n");
	return 0;
}

void train(){
	load_dataset();
	MODEL model;
	time_t tme=time(NULL);
	train_model(&model);
	tme=time(NULL)-tme;
	save_model(&model);
	printf("Trained in %ld seconds\n",tme);
}
void predict_test(){
	load_dataset();
	MODEL model;
	load_model(&model);
	for (int i=0;i<NUM_TEST;i++){
		predict(test_data[i],&model); 
	}
}

int main(int argc,char** argv){
	switch (parseCmd(argc,argv)){
		case TRAIN:{
			train();
	      		break;
		}
		case PREDICT:{
			predict_test();
			break;
		}
		default:{
		        return usage();
		}
	}
}
