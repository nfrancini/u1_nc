#include "../head/head_&_structures.h"

// PROCEDURA CHE COPIA LOCALMENTE LA STRUTTURA DEI CAMPI
// QUESTA VIENE POI SALVATA IN BINARIO
void writeFields(SystemParam_t *Par, Field_t *Fields){
  int iSite, k, mu;
  FILE *fptr;

  typedef struct{                           // DEFINISCO UNA STRUTTURA DI CAMPO COPIA CON DIMENSIONI ASSEGNATE
    double complex scalar[Par->V][N];
    double gauge[Par->V][D];
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

  fptr = fopen(Par->conf_file, "wb");
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
  FILE *fptr;

  typedef struct{                           // DEFINISCO UNA STRUTTURA DI CAMPO COPIA CON DIMENSIONI ASSEGNATE
    double complex scalar[Par->V][N];
    double gauge[Par->V][D];
  } Field_cpy_t;

  Field_cpy_t Fields_cpy;

  fptr = fopen(Par->conf_file, "rb");
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
  FILE *fptr;

  typedef struct{               // DEFINISCO PER COMODITA UNA STRUTTURA COPIA DEI PARAMETRI DI ACCETTANZA
    double eps1;
    double eps2;
  } eps_cpy_t;

  eps_cpy_t eps_cpy;

  fptr = fopen(Par->eps_file, "wb");
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
  FILE *fptr;

  typedef struct{                 // DEFINISCO PER COMODITA UNA STRUTTURA COPIA DEI PARAMETRI DI ACCETTANZA
    double eps1;
    double eps2;
  } eps_cpy_t;

  eps_cpy_t eps_cpy;

  fptr = fopen(Par->eps_file, "rb");
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

// PROCEDURA CHE SCRIVE IL FILE DI LOGS
void writeLogs(SystemParam_t *Par){
  FILE *fptr;
  fptr = fopen(Par->log_file, "w");
  if(fptr == NULL){
    perror("Errore in apertura");
    exit(EXIT_FAILURE);
  }

  fprintf(fptr, "+-------------------------------+\n");
  fprintf(fptr, "| Simulation details for u1_nc  |\n");
  fprintf(fptr, "+-------------------------------+\n");

  #ifdef DEBUG
  fprintf(fptr, "DEBUG attivo\n\n");
  #endif

  fprintf(fptr, "NUMERO DI FLAVOUR %d\n", N);
  fprintf(fptr, "DIMENSIONE SISTEMA %d\n", D);
  fprintf(fptr, "LATO DEL RETICOLO %d\n\n", Par->L);

  fprintf(fptr, "J %.5lf\n", Par->J);
  fprintf(fptr, "K %.5lf\n\n", Par->K);

  fprintf(fptr, "eps1 %.5lf\n", Par->eps1);
  fprintf(fptr, "eps2 %.5lf\n\n", Par->eps2);

  fprintf(fptr, "iTerm %d\n", Par->iTerm);
  fprintf(fptr, "iDec %d\n", Par->iDec);
  fprintf(fptr, "iMis %d\n", Par->iMis);
  fprintf(fptr, "iOverr %d\n\n", Par->iOverr);

  fprintf(fptr, "iStart %d\n", Par->iStart);
  fprintf(fptr, "iBackup %d\n\n", Par->iBackup);

  #ifdef DEBUG
  fprintf(fptr, "1.0e-12<ERRORE<1.0E-11 = %lf\n", err1/((Par->iOverr)*D*(Par->V)*((Par->iDec))));
  fprintf(fptr, "ERRORE>1.0E-11 = %lf\n\n", err2/((Par->iOverr)*D*(Par->V)*((Par->iDec))));
  #endif

  fprintf(fptr, "ACCETTANZA UPDATE DI GAUGE %lf\n", acc1/((Par->iMis)*(Par->iDec)*(D)*(Par->V)));
  fprintf(fptr, "ACCETTANZA UPDATE SCALARE %lf\n\n", acc2/((Par->iMis)*(Par->iDec)*(Par->V)));

  fprintf(fptr, "PROGRAMMA ESEGUITO CORRETTAMENTE\n");

  fclose(fptr);
}
