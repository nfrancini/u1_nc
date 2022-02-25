#include "../head/head_&_structures.h"

// FUNZIONE CHE RESTITUISCE IL CAMPO SCALARE VICINO
// CONSIDERANDO LE C*-BC
double complex nearest_scalar(Field_t *Fields, int site, int dir, int component, bool_t ctrl){
    if(ctrl == TRUE){   // SITE + DIR
      if(bc(site, npp[site][dir], TRUE) == FALSE){
        return Fields->scalar[npp[site][dir]][component];             // NON C'È ATTRAVERSAMENTO DEL BORDO
      }
      else return conj(Fields->scalar[npp[site][dir]][component]);    // BC
    }
    else if(ctrl == FALSE){   // SITE - DIR
      if(bc(site, nmm[site][dir], FALSE) == FALSE){
        return Fields->scalar[nmm[site][dir]][component];             // NON C'È ATTRAVERSAMENTO DEL BORDO
      }
      else return conj(Fields->scalar[nmm[site][dir]][component]);    // BC
    }
    else{
      printf("Problema nel nearest_scalar\n");
      exit(EXIT_FAILURE);
    }
}

// FUNZIONE CHE RESTITUISCE IL CAMPO DI GAUGE VICINO
// CONSIDERANDO LE C*BC
double nearest_gauge(Field_t *Fields, int site, int dir, int mu, bool_t ctrl){
  if(ctrl == TRUE){   // SITE + DIR, MU
    if(bc(site, npp[site][dir], TRUE) == FALSE){
      return Fields->gauge[npp[site][dir]][mu];                         // NON C'È ATTRAVERSAMENTO DEL BORDO
    }
    else return -Fields->gauge[npp[site][dir]][mu];                     // BC
  }
  else if(ctrl == FALSE){ // SITE-DIR, MU
    if(bc(site, nmm[site][dir], FALSE) == FALSE){
      return Fields->gauge[nmm[site][dir]][mu];                         // NON C'È ATTRAVERSAMENTO DEL BORDO
    }
    else return -Fields->gauge[nmm[site][dir]][mu];                     // BC
  }
  else{
    printf("Problema nel nearest_gauge\n");
    exit(EXIT_FAILURE);
  }
}

// FUNZIONE CHE RESTITUISCE A(SITE+MU-NU,NU)
// DA UTILIZZARE SOLO IN MEAN_GAUGE
double second_nearest_gauge(Field_t *Fields, int site, int nu, int mu){
  int x1, x2;
  bool_t flag1, flag2;

  x1 = npp[site][mu];
  x2 = nmm[x1][nu];

  flag1 = bc(site, x1, TRUE);
  flag2 = bc(x1, x2, FALSE);

  if(((flag1 == TRUE) && (flag2 == FALSE)) || ((flag1 == FALSE) && (flag2 == TRUE))){       // SOLO UN ATTRAVERSAMENTO NEI DUE STEP
    return -Fields->gauge[x2][nu];
  }
  else if(((flag1 == TRUE) && (flag2 == TRUE)) || ((flag1 == FALSE) && (flag2 == FALSE))){  // DUE ATTRAVERSAMENTI DEL BORDO
    return Fields->gauge[x2][nu];
  }
  else{
    printf("Errore in second_nearest_gauge\n");
    exit(EXIT_FAILURE);
  }
}

// PROCEDURA CHE CALCOLA IL CAMPO MEDIO PER L'UPDATE SCALARE
// IL CAMPO MEDIO È UN VETTORE AD N COMPONENTI
// HO IMPLEMENTATO LA FORMULA f = SUM_mu (e^(iA_x,mu)*z_(x+mu) + e^(-iA_(x-mu,mu))*z_(x-mu))
void mean_scalar(SystemParam_t *Par, Field_t *Fields, int site, double complex *mean){
  int mu, i;
  double complex sum, a, b;

  #ifdef DEBUG
  resetErr();
  #endif

  sum = 0+0*I;
  for(i=0;i<N;i++){
    for(mu=0;mu<D;mu++){
      a = cexp(I * (Fields->gauge[site][mu]));
      b = cexp(-I * nearest_gauge(Fields, site, mu, mu, FALSE));
      sum = sum + a*nearest_scalar(Fields, site, mu, i, TRUE) + b*nearest_scalar(Fields, site, mu, i, FALSE);
    }
    #ifdef DEBUG
    if(errno == EDOM){
      printf("Funzione mean_scalar, cexp\n");
      perror("    errno == EDOM");
    }
    if(errno == ERANGE){
      printf("Funzione mean_scalar, cexp\n");
      perror("    errno == ERANGE");
    }
    if(fetestexcept(FE_INVALID)){
      puts("    FE_INVALID was raised");
      exit(EXIT_FAILURE);
    }
    #endif
    mean[i] = sum;
    sum = 0+0*I;
  }
}

// FUNZIONE CHE CALCOLA IL CAMPO MEDIO PER L'UPDATE DEL CAMPO DI GAUGE
// PER IMPLEMENTARE QUESTA VEDERE FORMULA ALLEGATA SU IPAD
// SOSTANZIALMENTE HO SFRUTTATO LA SIMMETRIZZAZIONE DELLA SOMMA ED HO SVILUPPATO
// TENENDO FUORI SOLO I TERMINI SU SITO E DIREZIONE VOLUTI
// SOLO IN QUESTA FUNZIONE VIENE RICHIAMATA second_nearest_gauge
double mean_gauge(SystemParam_t *Par, Field_t *Fields, int site, int mu){
  int nu;
  double sum;

  sum=0;
  for(nu=0;nu<D;nu++){
    if(nu != mu){
      sum = sum + nearest_gauge(Fields, site, mu, nu, TRUE) - (Fields->gauge[site][nu]) - nearest_gauge(Fields, site, nu, mu, TRUE) + nearest_gauge(Fields, site, nu, nu, FALSE) -nearest_gauge(Fields, site, nu, mu, FALSE) - second_nearest_gauge(Fields, site, nu, mu);
    }
  }
  return sum;
}

// FUNZIONE CHE RINORMALIZZA IL MODULO DEL CAMPO SCALARE
// VA RICHIAMATA OGNI TANTO DURANTE GLI UPDATE
// PER RIAGGIUSTARE IL MODULO DEL CAMPO SCALARE CHE POTREBBE
// ASSUMERE VALORI DIVERSI DA UNO PER ERRORI NUMERICI
void renormalize(SystemParam_t *Par, Field_t *Fields){
  int iSite, j;
  double norm;

  #ifdef DEBUG
  resetErr();
  #endif

  for(iSite=0;iSite<(Par->V);iSite++){
    norm = sqrt(creal(product(Par, Fields->scalar[iSite], Fields->scalar[iSite])));   // CALCOLO LA NORMA
    #ifdef DEBUG
    if(errno == EDOM){
      printf("Funzione renormalize, radice quadrata\n");
      perror("    errno == EDOM");
    }
    if(errno == ERANGE){
      printf("Funzione renormalize, radice quadrata\n");
      perror("    errno == ERANGE");
    }
    if(fetestexcept(FE_INVALID)){
      puts("    FE_INVALID was raised");
      exit(EXIT_FAILURE);
    }
    #endif
    for(j=0; j<N;j++){
      Fields->scalar[iSite][j] = Fields->scalar[iSite][j] / norm;                     // DIVIDO PER LA NORMA, ALMENO TORNA 1
    }
  }
}
