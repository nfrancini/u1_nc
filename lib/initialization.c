#include "../head/head_&_structures.h"

// PROCEDURA GEOMETRY PER MEMORIZZARE I PRIMI VICINI;
void geometry(SystemParam_t *Par){
  int i, mu;

  // SFRUTTO L'ORDINAMENTO LESSICOGRAFICO i_x = x1 + x2*L + x3*L^2
  // IN QUESTO MODO x1 = i_x % L
  // x2 = (i_x / L) % L
  // x3 = (i_x / L^2) % L

  // IMPLEMENTAZIONE NPP
  for(i=0;i<(Par->V);i++){
    for(mu=0;mu<D;mu++){
      if((i / (int)pow(Par->L,mu)) % (Par->L) == ((Par->L)-1)){
        npp[i][mu] = i - (int)pow(Par->L,mu)*((Par->L)-1);
      }
      else npp[i][mu] = i + (int)pow(Par->L,mu);
    }
  }

  // IMPLEMENTAZIONE NMM
  for(i=0;i<(Par->V);i++){
    for(mu=0;mu<D;mu++){
      if((i / (int)pow((Par->L),mu)) % (Par->L) == 0){
        nmm[i][mu] = i + (int)pow((Par->L),mu)*((Par->L)-1);
      }
      else nmm[i][mu] = i - (int)pow((Par->L),mu);
    }
  }
}

// PROCEDURA CHE LEGGE DA FILE I PARAMETRI DI SISTEMA
void read_from_input_Param(SystemParam_t *Par){
  int flag;
  char buffer[64];
  FILE *fInput;

  snprintf(buffer, sizeof(char)*64, "./input/input.txt");
  fInput = fopen(buffer, "r");
  if (fInput == NULL) {
    perror("Errore in apertura in read_from_input_Param");
    exit(1);
  }
  flag = fscanf(fInput, "%d\n", &(Par->L));
  if (flag == EOF) {
    perror("Errore di lettura");
    exit(1);
  }
  flag = fscanf(fInput, "%lf\n", &(Par->J));
  if (flag == EOF) {
    perror("Errore di lettura");
    exit(1);
  }
  flag = fscanf(fInput, "%lf\n", &(Par->K));
  if (flag == EOF) {
    perror("Errore di lettura");
    exit(1);
  }
  flag = fscanf(fInput, "%lf\n", &(Par->eps1));
  if (flag == EOF) {
    perror("Errore di lettura");
    exit(1);
  }
  flag = fscanf(fInput, "%lf\n", &(Par->eps2));
  if (flag == EOF) {
    perror("Errore di lettura");
    exit(1);
  }
  flag = fscanf(fInput, "%d\n", &(Par->iTerm));
  if (flag == EOF) {
    perror("Errore di lettura");
    exit(1);
  }
  flag = fscanf(fInput, "%d\n", &(Par->iDec));
  if (flag == EOF) {
    perror("Errore di lettura");
    exit(1);
  }
  flag = fscanf(fInput, "%d\n", &(Par->iMis));
  if (flag == EOF) {
    perror("Errore di lettura");
    exit(1);
  }
  flag = fscanf(fInput, "%d\n", &(Par->iOverr));
  if (flag == EOF) {
    perror("Errore di lettura");
    exit(1);
  }
  fclose(fInput);

  Par->V = (int)pow(Par->L, D);    // CALCOLO QUA IL VOLUME
}

// PROCEDURA DI ALLOCAZIONE DINAMICA DELLA MEMORIA
void allocation(SystemParam_t *Par, Field_t *Fields){
  int i;

  Fields->scalar = (double complex **)malloc((Par->V)*sizeof(double complex *));
  for(i=0;i<(Par->V);i++){
    Fields->scalar[i] = (double complex *)malloc((N)*sizeof(double complex));
  }

  Fields->gauge = (double **)malloc((Par->V)*sizeof(double *));
  for(i=0;i<(Par->V);i++){
    Fields->gauge[i] = (double *)malloc((D)*sizeof(double));
  }

  npp = (int **)malloc((Par->V)*sizeof(int *));
  for(i=0;i<(Par->V);i++){
    npp[i] = (int *)malloc((D)*sizeof(int));
  }

  nmm = (int **)malloc((Par->V)*sizeof(int *));
  for(i=0;i<(Par->V);i++){
    nmm[i] = (int *)malloc((D)*sizeof(int));
  }
}

// PROCEDURA PER INIZIALIZZARE LA CONFIGURAZIONE DEI CAMPI
// PER IL MOMENTO VENGONO INIZIALIZZATI IN UNA CONFIGURAZIONE ORDINATA
// CON IL CAMPO SCALARE A COMPONENTI IDENTICHE E CAMPO DI GAUGE A ZERO
void initializeFields(SystemParam_t *Par, Field_t *Fields){
  int i, j, mu;

  #ifdef RESUME               // SE È DEFINITO RESUME ALLORA LEGGO I CAMPI DA FILE, SENZA INIZIALIZZARLI
  readFields(Par, Fields);
  return;
  #endif

  for(i=0;i<(Par->V);i++){
    for(j=0;j<N;j++){
      Fields->scalar[i][j] = 1/sqrt(N) + I*0;
    }
    for(mu=0;mu<D;mu++){
      Fields->gauge[i][mu] = 0.0;
    }
  }
}

// PROCEDURA DI SUPPORTO PER METTERE A ZERO TUTTE LE OSSERVABILI INIZIALMENTE
// NON SO SE È UTILE MA PER SICUREZZA LO FACCIO
void initializeObs(Obs_t *Obs){
  Obs->spin_ene_density = 0.0;
  Obs->gauge_ene_density = 0.0;
  Obs->ene_density = 0.0;
  Obs->susc = 0.0;
  Obs->G_pm = 0.0;
  Obs->mu2 = 0.0;
}

// PROCEDURA PER INIZIALIZZARE I PARAMETRI DI SISTEMA E LE CONFIGURAZIONI
void initializeSystem(SystemParam_t *Par, Field_t *Fields, Obs_t *Obs){

  read_from_input_Param(Par);       // LEGGO DA input.txt I PARAMETRI DEL SISTEMA

  #ifdef RESUME                     // SE RESUME È DEFINITO ALLORA AGGIUSTO LE EPS
  readEps(Par);                     // DELLE ACCETTANZE CON QUELLE SALVATE
  #endif

  allocation(Par, Fields);          // ALLOCO DINAMICAMENTE LA MEMORIA

  geometry(Par);                    // MEMORIZZO I PRIMI VICINI

  initializeFields(Par, Fields);    // INIZIALIZZO I CAMPI

  initializeObs(Obs);               // INIZIALIZZO LE OSSERVABILI

  #ifdef DEBUG
  resetErr();   // RESETTO LE VARIABILI DI ERROR HANDLING
  #endif
}

// PROCEDURA DI DEALLOCAZIONE DELLA MEMORIA DINAMICA PRECEDENTEMENTE ALLOCATA
void deallocation(Field_t *Fields){
  free(Fields->scalar);
  free(Fields->gauge);
  free(npp);
  free(nmm);
}
