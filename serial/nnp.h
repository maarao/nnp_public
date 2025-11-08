typedef struct tagMODEL{
    float W1[SIZE*H1];
    float b1[H1];
    float W2[H1*H2];
    float b2[H2];
    float W3[H2*CLASSES];
    float b3[CLASSES];
} MODEL;

// Activation functions
float relu(float x);
float drelu(float y);

void softmax(float *z, float *out, int len);
void init_weights(float *w, int size);
void train_model(MODEL* model);
void save_model(MODEL* model);
void load_model(MODEL* model);
void predict(float *x, MODEL* model);
