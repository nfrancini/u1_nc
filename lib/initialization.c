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

// PROCEDURA PER ELIMINARE GLI SPAZI ED I COMMENTI DAL FILE DI INPUT
void remove_white_line_and_comments(FILE *input){
  int temp_i;

  temp_i=getc(input);
  if(temp_i=='\n' || temp_i==' ' || temp_i=='\043') // SCAN PER SPAZI VUOTI E COMMENTI
    {
    ungetc(temp_i, input);

    temp_i=getc(input);
    if(temp_i=='\n' || temp_i==' ') // LINEE VUOTE
      {
      do
       {
       temp_i=getc(input);
       }
      while(temp_i=='\n' || temp_i==' ');
      }
    ungetc(temp_i, input);

    temp_i=getc(input);
    if(temp_i=='\043')  // COMMENTI, 043 È IL CODICE ASCII PER #
      {
      do
       {
       temp_i=getc(input);
       }
      while(temp_i!='\n');
      }
    else
      {
      ungetc(temp_i, input);
      }

    remove_white_line_and_comments(input);
    }
  else
    {
    ungetc(temp_i, input);
    }
  }

// PROCEDURA CHE LEGGE DA FILE I PARAMETRI DI SISTEMA
void read_from_input_Param(SystemParam_t *Par, char const *finput){
  int flag, temp_i;
  int end=1;
  FILE *fInput;
  char str[100];

  fInput = fopen(finput, "r");
  if (fInput == NULL) {
    perror("Errore in apertura in read_from_input_Param");
    exit(EXIT_FAILURE);
  }
  else{
    while(end==1){
      remove_white_line_and_comments(fInput);

      flag = fscanf(fInput, "%s", str);
      if (flag != 1) {
        perror("Errore di lettura della prima stringa");
        exit(EXIT_FAILURE);
      }

      if(strncmp(str, "size", 4) == 0){
        flag = fscanf(fInput, "%d", &(Par->L));
        if (flag != 1){
          perror("Errore di lettura nella taglia");
          exit(EXIT_FAILURE);
        }
      }
      else if(strncmp(str, "j", 1) == 0){
        flag = fscanf(fInput, "%lf", &(Par->J));
        if (flag != 1){
          perror("Errore di lettura");
          exit(EXIT_FAILURE);
        }
      }
      else if(strncmp(str, "k", 1) == 0){
        flag = fscanf(fInput, "%lf", &(Par->K));
        if (flag != 1){
          perror("Errore di lettura");
          exit(EXIT_FAILURE);
        }
      }
      else if(strncmp(str, "eps_gauge", 9) == 0){
        flag = fscanf(fInput, "%lf", &(Par->eps1));
        if (flag != 1){
          perror("Errore di lettura");
          exit(EXIT_FAILURE);
        }
      }
      else if(strncmp(str, "eps_scalar", 10) == 0){
        flag = fscanf(fInput, "%lf", &(Par->eps2));
        if (flag != 1){
          perror("Errore di lettura");
          exit(EXIT_FAILURE);
        }
      }
      else if(strncmp(str, "iTerm", 5) == 0){
        flag = fscanf(fInput, "%d", &(Par->iTerm));
        if (flag != 1){
          perror("Errore di lettura");
          exit(EXIT_FAILURE);
        }
      }
      else if(strncmp(str, "iDec", 4) == 0){
        flag = fscanf(fInput, "%d", &(Par->iDec));
        if (flag != 1){
          perror("Errore di lettura");
          exit(EXIT_FAILURE);
        }
      }
      else if(strncmp(str, "iMis", 4) == 0){
        flag = fscanf(fInput, "%d", &(Par->iMis));
        if (flag != 1){
          perror("Errore di lettura");
          exit(EXIT_FAILURE);
        }
      }
      else if(strncmp(str, "iOverr", 6) == 0){
        flag = fscanf(fInput, "%d", &(Par->iOverr));
        if (flag != 1){
          perror("Errore di lettura");
          exit(EXIT_FAILURE);
        }
      }
      else if(strncmp(str, "iStart", 6) == 0){
        flag = fscanf(fInput, "%d", &(Par->iStart));
        if (flag != 1){
          perror("Errore di lettura");
          exit(EXIT_FAILURE);
        }
      }
      else if(strncmp(str, "iBackup", 7) == 0){
        flag = fscanf(fInput, "%d", &(Par->iBackup));
        if (flag != 1){
          perror("Errore di lettura");
          exit(EXIT_FAILURE);
        }
      }
      else if(strncmp(str, "conf_file", 9) == 0){
        flag = fscanf(fInput, "%s", Par->conf_file);
        if (flag != 1){
          perror("Errore di lettura");
          exit(EXIT_FAILURE);
        }
      }
      else if(strncmp(str, "eps_file", 8) == 0){
        flag = fscanf(fInput, "%s", Par->eps_file);
        if (flag != 1){
          perror("Errore di lettura");
          exit(EXIT_FAILURE);
        }
      }
      else if(strncmp(str, "data_file", 9) == 0){
        flag = fscanf(fInput, "%s", Par->data_file);
        if (flag != 1){
          perror("Errore di lettura");
          exit(EXIT_FAILURE);
        }
      }
      else if(strncmp(str, "log_file", 8) == 0){
        flag = fscanf(fInput, "%s", Par->log_file);
        if (flag != 1){
          perror("Errore di lettura");
          exit(EXIT_FAILURE);
        }
      }
      else if(strncmp(str, "random_seed", 11) == 0){
        flag = fscanf(fInput, "%u", &(Par->dSFMT_seed));
        if (flag != 1){
          perror("Errore di lettura");
          exit(EXIT_FAILURE);
        }
      }
      else{
        fprintf(stderr, "Error: unrecognized option %s in the file %s (%s, %d)\n", str, finput, __FILE__, __LINE__);
        exit(EXIT_FAILURE);
      }

      remove_white_line_and_comments(fInput);

      // CONTROLLO SE LA LINEA LETTA È L'ULTIMA
      temp_i = getc(fInput);
      if(temp_i == EOF){
        end=0;
      }
      else{
        ungetc(temp_i, fInput);
      }
    }

    fclose(fInput);
    Par->V = (int)pow(Par->L, D);    // CALCOLO QUA IL VOLUME
  }
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

  if((Par->iStart) == 1){             // COPIO L'ULTIMA CONFIGURAZIONE
    readFields(Par, Fields);
  }
  else if((Par->iStart) == 0){        // INIZIALIZZO IN MODO ORDINATO
    for(i=0;i<(Par->V);i++){
      for(j=0;j<N;j++){
        Fields->scalar[i][j] = 1/sqrt(N) + I*0;
      }
      for(mu=0;mu<D;mu++){
        Fields->gauge[i][mu] = 0.0;
      }
    }
  }
  else{
    printf("Errore in iStart\n");
    exit(EXIT_FAILURE);
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
void initializeSystem(SystemParam_t *Par, Field_t *Fields, Obs_t *Obs, char const *finput){

  read_from_input_Param(Par, finput);     // LEGGO DA input.txt I PARAMETRI DEL SISTEMA

  if((Par->iStart) == 1){           // LEGGO I PARAMETRI DI ACCETTANZA SALVATI
    readEps(Par);
  }

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
