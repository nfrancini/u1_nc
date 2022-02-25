#include "../head/head_&_structures.h"

// DEFINISCO UNA FUNZIONE RNDM CHE RICHIAMA IL DSFMT.
// QUESTO È UTILE PER CAMBIARE GENERATORE SENZA DOVERE OGNI
// VOLTA CORREGGERE IL CODICE
double rndm(){    // TEST OK
  double x;

  x = dsfmt_genrand_close_open(&dsfmt);
  return x;
}

// FUNZIONE BC CHE CONTROLLA SE VIENE ATTRAVERSATO IL BORDO
// QUESTA FUNZIONE NON È ADATTA A VALUTARE PERCORSI A 2 O PIÙ LOOP
// IN QUANTO NON CAPISCE QUANTI GIRI SONO STATI FATTI.
// IN PRATICA LA UTILIZZO SOLO CON I PRIMI VICINI COME POSIZIONI INIZIALI
// E FINALI
bool_t bc(int initPos, int finPos, bool_t ctrl){
  bool_t flag;

  if(ctrl == FALSE){    // IMPLEMENTO BCM
    if(finPos > initPos){
      flag = TRUE;      // BORDO ATTRAVERSATO
    }
    else if(finPos < initPos){
      flag = FALSE;     // BORDO NON ATTRAVERSATO
    }
    else{
      printf("Errore nelle posizioni \n");
      exit(EXIT_FAILURE);
    }
    return flag;
  }
  else if(ctrl == TRUE){  // IMPLEMENTO BCP
    if(finPos < initPos){
      flag = TRUE;        // BORDO ATTRAVERSATO
    }
    else if(finPos > initPos){
      flag = FALSE;       // BORDO NON ATTRAVERSATO
    }
    else{
      printf("Errore nelle posizioni \n");
      exit(EXIT_FAILURE);
    }
    return flag;
  }
  else{
    printf("Errore nelle bc\n");
    exit(EXIT_FAILURE);
  }
}

// FUNZIONI AUSILIARIE PER MANIPOLARE I VETTORI DI SU(N)
void diff(SystemParam_t *Par, double complex *vec1, double complex *vec2, double complex *result){
  int i;

  for(i=0;i<(N);i++){
    result[i] = vec1[i] - vec2[i];
  }
}

// FUNZIONE CHE COPIA VEC1 IN VEC2
void copy(SystemParam_t *Par, double complex *vec1, double complex *vec2){
  int i;

  for(i=0;i<(N);i++){
    vec2[i] = vec1[i];
  }
}

// FUNZIONE CHE FA IL PRODOTTO BAR(VEC1)*VEC2
double complex product(SystemParam_t *Par, double complex *vec1, double complex *vec2){
  int i;
  double complex sum;

  sum = 0+0*I;
  for(i=0; i<(N); i++){
    sum = sum + conj(vec1[i])*vec2[i];
  }
  return sum;
}

// FUNZIONE CHE RESETTA LE VARIABILI DI ERROR HANDLING
// VIENE RICHIAMATA NELLE VARIE FUNZIONI SE DEBUG È DEFINITO
// È IMPORTANTE RESETTARE OGNI VOLTA LE VARIABILI DI ERROR HANDLING ALTRIMENTI
// IN PRESENZA DI UN ERRORE NON SI RIESCE A CAPIRE DA DOVE ARRIVI
void resetErr(){
  errno = 0;
  feclearexcept(FE_ALL_EXCEPT);
}

// FUNZIONE CHE CONTROLLA IL RANGE DELLE ACCETTANZE
// RICHIAMATA NELLA TERMALIZZAZIONE, IN MODO
// DA IMPOSTARE AUTOMATICAMENTE I PARAMETRI EPS1 ED EPS2
// CHE REGOLANO LE ACCETTANZE
// IN QUESTO CASO HO PRESO UN LIMITE INFERIORE DI 0.30 E SUPERIORE DI 0.37
void ctrl_acceptance(double ac, bool_t *ctrl_1, bool_t *ctrl_2){
  double a_min, a_max;
  a_min = 0.30;
  a_max = 0.37;
  if((ac>=a_min)&&(ac<=a_max)){
    *ctrl_1 = TRUE;
    *ctrl_2 = FALSE;
  }
  else if(ac<a_min){
    *ctrl_1 = FALSE;
    *ctrl_2 = FALSE;
  }
  else if(ac>a_max){
    *ctrl_1 = FALSE;
    *ctrl_2 = TRUE;
  }
}
