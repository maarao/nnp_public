#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "config.h"
#include "loader.h"

// Global arrays
float train_data[NUM_TRAIN][SIZE];
float train_label[NUM_TRAIN][CLASSES];   // one-hot
float test_data[NUM_TEST][SIZE];
float test_label[NUM_TEST][CLASSES];     // one-hot

// Helper: read big-endian 32-bit int
int read_int(FILE *f) {
    unsigned char b[4];
    fread(b,1,4,f);
    return (b[0]<<24)|(b[1]<<16)|(b[2]<<8)|b[3];
}

void load_data(const char *filename, float data[][SIZE], int num) {
    FILE *f=fopen(filename,"rb");
    if(!f){ perror("open data file"); exit(1); }
    read_int(f); //magic number
    int n=read_int(f);
    int rows=read_int(f);
    int cols=read_int(f);
    if(n<num && num!=0) num=n;
    unsigned char *buf=(unsigned char*)malloc(rows*cols);
    for(int i=0;i<num;i++){
        fread(buf,1,rows*cols,f);
        for(int j=0;j<rows*cols;j++)
            data[i][j]=buf[j]/255.0f;  // normalize to [0,1]
    }
    free(buf);
    fclose(f);
}

void load_labels(const char *filename, float labels[][CLASSES], int num) {
    FILE *f=fopen(filename,"rb");
    if(!f){ perror("open label file"); exit(1); }
    read_int(f); //magic number
    int n=read_int(f);
    if(n<num && num!=0) num=n;
    unsigned char *buf=(unsigned char*)malloc(n);
    fread(buf,1,n,f);
    for(int i=0;i<num;i++){
        for(int k=0;k<CLASSES;k++) labels[i][k]=0.0f;
        labels[i][buf[i]]=1.0f;   // one-hot encode
    }
    free(buf);
    fclose(f);
}

void load_dataset() {
    load_data(TRAIN_DATA, train_data, NUM_TRAIN);
    load_labels(TRAIN_LABELS, train_label, NUM_TRAIN);
    load_data(TEST_DATA, test_data, NUM_TEST);
    load_labels(TEST_LABELS, test_label, NUM_TEST);
}

