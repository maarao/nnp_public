#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <cuda.h>
#include "config.h"
#include "loader.h"
#include "nnp.h"
#include "kernels.h"


// Activation functions
float relu(float x) { return x > 0 ? x : 0; }
float drelu(float y) { return y > 0 ? 1 : 0; }

void softmax(float *z, float *out, int len) {
    float max = z[0];
    for (int i=1;i<len;i++) if (z[i]>max) max=z[i];
    float sum=0;
    for (int i=0;i<len;i++){ out[i]=expf(z[i]-max); sum+=out[i]; }
    for (int i=0;i<len;i++) out[i]/=sum;
}

// Initialize weights
void init_weights(float *w, int size) {
    for (int i=0;i<size;i++)
        w[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.1f;
}

void train_model(MODEL* model){
    init_weights(model->W1, SIZE*H1); init_weights(model->b1, H1);
    init_weights(model->W2, H1*H2); init_weights(model->b2, H2);
    init_weights(model->W3, H2*CLASSES); init_weights(model->b3, CLASSES);

    for (int epoch=0; epoch<EPOCHS; epoch++) {
        float loss=0;
        for (int n=0; n<NUM_TRAIN; n++) {
            // ---------- Forward ----------
            float h1[H1], h1a[H1];
            for (int j=0;j<H1;j++){
                h1[j]=model->b1[j];
                for (int i=0;i<SIZE;i++) h1[j]+=train_data[n][i]*model->W1[i*H1+j];
                h1a[j]=relu(h1[j]);
            }
            float h2[H2], h2a[H2];
            for (int j=0;j<H2;j++){
                h2[j]=model->b2[j];
                for (int i=0;i<H1;i++) h2[j]+=h1a[i]*model->W2[i*H2+j];
                h2a[j]=relu(h2[j]);
            }
            float out[CLASSES], outa[CLASSES];
            for (int k=0;k<CLASSES;k++){
                out[k]=model->b3[k];
                for (int j=0;j<H2;j++) out[k]+=h2a[j]*model->W3[j*CLASSES+k];
            }
            softmax(out,outa,CLASSES);

            // ---------- Loss ----------
            for (int k=0;k<CLASSES;k++)
                loss -= train_label[n][k]*logf(outa[k]+1e-8f);

            // ---------- Backprop ----------
            float delta3[CLASSES];
            for (int k=0;k<CLASSES;k++)
                delta3[k] = train_label[n][k]-outa[k];

            float delta2[H2];
            for (int j=0;j<H2;j++){
                float err=0;
                for (int k=0;k<CLASSES;k++) err+=delta3[k]*model->W3[j*CLASSES+k];
                delta2[j]=err*drelu(h2a[j]);
            }

            float delta1[H1];
            for (int j=0;j<H1;j++){
                float err=0;
                for (int k=0;k<H2;k++) err+=delta2[k]*model->W2[j*H2+k];
                delta1[j]=err*drelu(h1a[j]);
            }

            // ---------- Update ----------
            for (int j=0;j<H2;j++)
                for (int k=0;k<CLASSES;k++)
                    model->W3[j*CLASSES+k]+=LR*delta3[k]*h2a[j];
            for (int k=0;k<CLASSES;k++) model->b3[k]+=LR*delta3[k];

            for (int j=0;j<H1;j++)
                for (int k=0;k<H2;k++)
                    model->W2[j*H2+k]+=LR*delta2[k]*h1a[j];
            for (int k=0;k<H2;k++) model->b2[k]+=LR*delta2[k];

            for (int i=0;i<SIZE;i++)
                for (int j=0;j<H1;j++)
                    model->W1[i*H1+j]+=LR*delta1[j]*train_data[n][i];
            for (int j=0;j<H1;j++) model->b1[j]+=LR*delta1[j];
        }
        printf("Epoch %d, Loss=%.4f\n", epoch, loss/NUM_TRAIN);
    }
}
void save_model(MODEL* model){
	FILE *f = fopen("model.bin", "wb");
	fwrite(model->W1, sizeof(float), SIZE*H1, f);
	fwrite(model->b1, sizeof(float), H1, f);
	fwrite(model->W2, sizeof(float), H1*H2, f);
	fwrite(model->b2, sizeof(float), H2, f);
	fwrite(model->W3, sizeof(float), H2*CLASSES, f);
	fwrite(model->b3, sizeof(float), CLASSES,f);
	fclose(f);
}
void load_model(MODEL* model){
	FILE *f = fopen("model.bin", "rb");
	fread(model->W1, sizeof(float), SIZE*H1, f);
	fread(model->b1, sizeof(float), H1, f);
	fread(model->W2, sizeof(float), H1*H2, f);
	fread(model->b2, sizeof(float), H2, f);
	fread(model->W3, sizeof(float), H2*CLASSES, f);
	fread(model->b3, sizeof(float), CLASSES, f);
	fclose(f);
}

void predict(float *x, MODEL* model){
    float h1[H1], h1a[H1], h2[H2], h2a[H2], out[CLASSES], outa[CLASSES];

    // forward pass
    for (int j=0;j<H1;j++){ h1[j]=model->b1[j]; for(int i=0;i<SIZE;i++) h1[j]+=x[i]*model->W1[i*H1+j]; h1a[j]=relu(h1[j]); }
    for (int j=0;j<H2;j++){ h2[j]=model->b2[j]; for(int i=0;i<H1;i++) h2[j]+=h1a[i]*model->W2[i*H2+j]; h2a[j]=relu(h2[j]); }
    for (int k=0;k<CLASSES;k++){ out[k]=model->b3[k]; for(int j=0;j<H2;j++) out[k]+=h2a[j]*model->W3[j*CLASSES+k]; }
    softmax(out,outa,CLASSES);

    // print predicted class
    int pred=0; float max=outa[0];
    for(int k=1;k<CLASSES;k++) if(outa[k]>max){ max=outa[k]; pred=k; }
    printf("Predicted digit: %d (confidence %.2f)\n", pred, max);
}


