#include "../head/head_&_structures.h"

// FUNZIONE CHE CALCOLA IL CONTRIBUTO ENERGETICO DELLA PARTE SCALARE
double H_z_dens(SystemParam_t *Par, Field_t *Fields){
  int iSite, mu, k;
  double sum, ene;
  double complex prod;
  double complex near[N];

  #ifdef DEBUG
  resetErr();
  #endif

  sum=0;
  for(iSite=0;iSite<(Par->V);iSite++){
    for(mu = 0; mu<(D); mu++){
      for(k=0;k<(N);k++){
        near[k] = nearest_scalar(Fields, iSite, mu, k, TRUE);                   //  CALCOLO Z_(X+MU)
      }
      prod = product(Par, Fields->scalar[iSite], near);                         //  CALCOLO BAR(Z_MU)*Z_(X+MU)
      sum = sum + creal(prod * cexp(I * (Fields->gauge[iSite][mu])));           //  CALCOLO SUM RE(BAR(Z_MU)*Z_(X+MU)* e^(i*A_x,mu))
      #ifdef DEBUG
      if(errno == EDOM){
        printf("Funzione H_z, cexp\n");
        perror("    errno == EDOM");
      }
      if(errno == ERANGE){
        printf("Funzione H_z\n");
        perror("    errno == ERANGE");
        if(fetestexcept(FE_OVERFLOW)){
          puts("    FE_OVERFLOW was raised");
          exit(EXIT_FAILURE);
        }
        else if(fetestexcept(FE_UNDERFLOW)){
          puts("    FE_UNDERFLOW was raised");
          exit(EXIT_FAILURE);
        }
      }
      if(fetestexcept(FE_INVALID)){
        puts("    FE_INVALID was raised");
        exit(EXIT_FAILURE);
      }
      #endif
    }
  }
  ene = -2*(Par->J)*(N)*sum;                 // CALCOLO L'ENERGIA
  return ene/(Par->V);                            // RESTITUISCO LA DENSITÀ DI ENERGIA DOVUTA ALLA PARTE SCALARE
}

// FUNZIONE CHE CALCOLA IL CONTRIBUTO DEL CAMPO DI GAUGE
double H_g_dens(SystemParam_t *Par, Field_t *Fields){
  int iSite, mu, nu;
  double sum, ene;

  #ifdef DEBUG
  resetErr();
  #endif

  sum=0;
  for(iSite=0;iSite<(Par->V);iSite++){    // CALCOLO LA SOMMA PER IL CONTRIBUTO DI GAUGE
    for(mu=0;mu<(D);mu++){           // NOTO CHE SUM_(MU<NU) = 1/2 SUM(MU != NU) SE IL TENSORE SOMMATO È SIMMETRICO
      for(nu=0;nu<(D);nu++){
        if(nu != mu){
          sum = sum + pow(nearest_gauge(Fields, iSite, mu, nu, TRUE) -Fields->gauge[iSite][nu] - nearest_gauge(Fields, iSite, nu, mu, TRUE) + Fields->gauge[iSite][mu], 2);
        }
        #ifdef DEBUG
        if(errno == EDOM){
          printf("Funzione H_g, pow\n");
          perror("    errno == EDOM");
        }
        if(errno == ERANGE){
          printf("Funzione H_g, pow\n");
          perror("    errno == ERANGE");
          if(fetestexcept(FE_OVERFLOW)){
            puts("    FE_OVERFLOW was raised");
            exit(EXIT_FAILURE);
          }
          else if(fetestexcept(FE_UNDERFLOW)){
            puts("    FE_UNDERFLOW was raised");
            exit(EXIT_FAILURE);
          }
        }
        if(fetestexcept(FE_INVALID)){
          puts("    FE_INVALID was raised");
          exit(EXIT_FAILURE);
        }
        if(fetestexcept(FE_DIVBYZERO)){
          puts("    FE_DIVBYZERO was raised");
          exit(EXIT_FAILURE);
        }
        #endif
      }
    }
  }
  ene = (Par->K/2.0)*(sum/2.0);       // UN FATTORE 1/2 È DOVUTO ALLA NORMALIZZAZIONE DI K MENTRE L'ALTRO ALLA SIMMETRIZZAZIONE DELLA SOMMA
  return ene/(Par->V);              // RESTITUISCO LA DENSITÀ DI ENERGIA DEL CAMPO DI GAUGE
}

// FUNZIONE CHE RESTITUISCE LA DENSITÀ DI ENERGIA TOTALE
double H_dens(SystemParam_t *Par, Field_t *Fields){
  double ene_dens;
  ene_dens = H_z_dens(Par, Fields) + H_g_dens(Par, Fields);
  return ene_dens;
}

// PROCEDURA CHE CALCOLA SUSCETTIVITÀ E MU2 USANDO LA FORMULAZIONE MATRICIALE
void susc_mu2(SystemParam_t *Par, Field_t *Fields, double *susc, double *mu2){
  int a, b, iSite;
  double complex aux_matrix[N][N];
  double sum;

  sum = 0;
  // CREO LA MATRICE A=sum_x Q_x
  for(a=0;a<(N);a++){
    for(b=0;b<(N);b++){
      aux_matrix[a][b] = 0+0*I;
      for(iSite=0;iSite<(Par->V);iSite++){
        aux_matrix[a][b] = aux_matrix[a][b] + conj(Fields->scalar[iSite][a])*(Fields->scalar[iSite][b]);
        if(a==b){
          aux_matrix[a][b] = aux_matrix[a][b] - 1.0/(N);
        }
      }
    }
  }
  // CALCOLO LA SOMMA sum_a,b |A_a,b|^2
  for(a=0;a<(N);a++){
    for(b=0;b<(N);b++){
      sum = sum + pow(cabs(aux_matrix[a][b]), 2);
    }
  }

  *susc = sum/(Par->V);
  *mu2 = sum/pow(Par->V, 2);
}

// PROCEDURA CHE CALCOLA G_PM CON LA FORMULAZIONE MATRICIALE
void G_pm_matrix(SystemParam_t *Par, Field_t *Fields, double *G_pm){
  int a, b, iSite;
  double complex aux_matrix[N][N];
  double complex q_x;
  double sum;

  sum = 0;
  // CREO LA MATRICE A=sum_x Q_x
  for(a=0;a<(N);a++){
    for(b=0;b<(N);b++){
      aux_matrix[a][b] = 0+0*I;
      for(iSite=0;iSite<(Par->V);iSite++){
        q_x = conj(Fields->scalar[iSite][a])*(Fields->scalar[iSite][b]);
        if(a==b){
          q_x = q_x - 1.0/(N);
        }
        aux_matrix[a][b] = aux_matrix[a][b] + cexp(-I*2*PI*(iSite%(Par->L))/(Par->L))*q_x;
      }
    }
  }
  // CALCOLO LA SOMMA sum_a,b |A_a,b|^2
  for(a=0;a<(N);a++){
    for(b=0;b<(N);b++){
      sum = sum + pow(cabs(aux_matrix[a][b]), 2);
    }
  }
  *G_pm = sum/(Par->V);
}

// PROCEDURA CHE PRENDE IN INGRESSO LA STRUTTURA
// OSSERVABILI E FA LE VARIE MISURE
void measure(SystemParam_t *Par, Field_t *Fields, Obs_t *Obs){
  Obs->spin_ene_density = H_z_dens(Par, Fields);
  Obs->gauge_ene_density = H_g_dens(Par, Fields);
  Obs->ene_density = Obs->spin_ene_density + Obs->gauge_ene_density;
  // susc(Par, Fields, &(Obs->susc));
  // G_pm(Par, Fields, &(Obs->G_pm));
  // mu2(Par, Fields, &(Obs->mu2));
  susc_mu2(Par, Fields, &(Obs->susc), &(Obs->mu2));
  G_pm_matrix(Par, Fields, &(Obs->G_pm));
}
