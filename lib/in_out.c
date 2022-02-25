#include "../head/head_&_structures.h"

// PROCEDURA CHE COPIA LOCALMENTE LA STRUTTURA DEI CAMPI
// QUESTA VIENE POI SALVATA IN BINARIO
void writeFields(SystemParam_t *Par, Field_t *Fields){
  int iSite, k, mu;
  char buffer[64];
  FILE *fptr;

  typedef struct{                           // DEFINISCO UNA STRUTTURA DI CAMPO COPIA CON DIMENSIONI ASSEGNATE
    double complex scalar[Par->V][N];
    double gauge[Par->V][N];
  } Field_cpy_t;

  Field_cpy_t Fields_cpy;

  for(iSite=0;iSite<(Par->V);iSite++){      // COPIO LA CONFIGURAZIONE DI CAMPO
    for(k=0;k<N;k++){
      Fields_cpy.scalar[iSite][k] = Fields->scalar[iSite][k];
    }
    for(mu=0;mu<D;mu++){
      Fields_cpy.gauge[iSite][mu] = Fields->gauge[iSite][mu];
    }
  }

  snprintf(buffer, sizeof(char)*64, "./bin/config.bin");
  fptr = fopen(buffer, "wb");
  if (fptr == NULL) {
    perror("Errore in apertura");
    exit(1);
  }
  fwrite(&Fields_cpy, sizeof(Field_cpy_t), 1, fptr);
  fclose(fptr);
}

// PROCEDURA CHE LEGGE DA FILE LA CONFIGURAZIONE DEI CAMPI
// PRECEDENTEMENTE SALVATA E LA METTE NELLA CONFIGURAZIONE
// CORRENTE
void readFields(SystemParam_t *Par, Field_t *Fields){
  int iSite, k, mu, j;
  char buffer[64];
  FILE *fptr;

  typedef struct{                           // DEFINISCO UNA STRUTTURA DI CAMPO COPIA CON DIMENSIONI ASSEGNATE
    double complex scalar[Par->V][N];
    double gauge[Par->V][N];
  } Field_cpy_t;

  Field_cpy_t Fields_cpy;

  snprintf(buffer, sizeof(char)*64, "./bin/config.bin");
  fptr = fopen(buffer, "rb");
  if (fptr == NULL) {
    perror("Errore in apertura");
    exit(1);
  }

  j = fread(&Fields_cpy, sizeof(Field_cpy_t), 1, fptr);
  if (j!=1){
    printf("ERRORE NELLA LETTURA DA FILE\n");
    exit(EXIT_FAILURE);
  }
  fclose(fptr);

  for(iSite=0;iSite<(Par->V);iSite++){                // LEGGO DA FILE LA CONFIGURAZIONE
    for(k=0;k<N;k++){
      Fields->scalar[iSite][k] = Fields_cpy.scalar[iSite][k];
    }
    for(mu=0;mu<D;mu++){
      Fields->gauge[iSite][mu] = Fields_cpy.gauge[iSite][mu];
    }
  }
}

// PROCEDURA CHE COPIA LOCALMENTE LA STRUTTURA DEI
// PARAMETRI DI ACCETTANZA
void writeEps(SystemParam_t *Par){
  char buffer[64];
  FILE *fptr;

  typedef struct{               // DEFINISCO PER COMODITA UNA STRUTTURA COPIA DEI PARAMETRI DI ACCETTANZA
    double eps1;
    double eps2;
  } eps_cpy_t;

  eps_cpy_t eps_cpy;

  snprintf(buffer, sizeof(char)*64, "./bin/eps.bin");
  fptr = fopen(buffer, "wb");
  if (fptr == NULL) {
    perror("Errore in apertura");
    exit(1);
  }

  eps_cpy.eps1 = Par->eps1;     // COPIO SU FILE I PARAMETRI DI ACCETTANZA
  eps_cpy.eps2 = Par->eps2;

  fwrite(&eps_cpy, sizeof(eps_cpy), 1, fptr);
  fclose(fptr);
}

// PROCEDURA CHE LEGGE DA FILE BINARIO I PARAMETRI DI ACCETTANZA PRECEDENTI
void readEps(SystemParam_t *Par){
  int flag;
  char buffer[64];
  FILE *fptr;

  typedef struct{                 // DEFINISCO PER COMODITA UNA STRUTTURA COPIA DEI PARAMETRI DI ACCETTANZA
    double eps1;
    double eps2;
  } eps_cpy_t;

  eps_cpy_t eps_cpy;

  snprintf(buffer, sizeof(char)*64, "./bin/eps.bin");
  fptr = fopen(buffer, "rb");
  if (fptr == NULL) {
    perror("Errore in apertura");
    exit(1);
  }

  flag = fread(&eps_cpy, sizeof(eps_cpy), 1, fptr);
  if(flag != 1){
    printf("ERRORE NELLA LETTURA DEI FILE\n");
    exit(EXIT_FAILURE);
  }
  fclose(fptr);

  Par->eps1 = eps_cpy.eps1;       // ASSEGNO I VALORI LETTI DA FILE
  Par->eps2 = eps_cpy.eps2;
}

// PROCEDURA CHE SCRIVE LE OSSERVABILI SU FILE DOPO OGNI MISURA
void writeObs(FILE *fptr, Obs_t *Obs){
  fprintf(fptr, "%.13lf\t%.13lf\t%.13lf\t%.13lf\t%.13lf\t%.13lf\n", Obs->spin_ene_density, Obs->gauge_ene_density, Obs->ene_density, Obs->susc, Obs->G_pm, Obs->mu2);
}
