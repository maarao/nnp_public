#ifndef CONFIG_H
#define CONFIG_H

#define TRAIN_DATA "train-images-idx3-ubyte"
#define TRAIN_LABELS "train-labels-idx1-ubyte"
#define TEST_DATA  "t10k-images-idx3-ubyte"
#define TEST_LABELS  "t10k-labels-idx1-ubyte"

#define NUM_TRAIN 60000
#define NUM_TEST  10000
#define ROWS 28
#define COLS 28
#define SIZE (ROWS*COLS)
#define CLASSES 10
#define H1    256     // hidden layer 1
#define H2    128     // hidden layer 2
#define EPOCHS 5
#define BATCH  64
#define LR     0.01f

#endif
