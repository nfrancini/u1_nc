#include "../head/head_&_structures.h"

// FUNZIONE PER CREARE LO STATO DI PROVA DEL CAMPO SCALARE NEL SITO SITE
void scalar_trial(SystemParam_t *Par, Field_t *Fields, int site, double complex *trial){
  int i, j;
  double complex temp1, temp2;
  double a0, a1, a2, a3, norm, phi, theta, x;

  #ifdef DEBUG
  resetErr();
  #endif

  copy(Par, Fields->scalar[site], trial);       // INTANTO COPIO IN TRIAL IL VETTORE DI PARTENZA
                                                // POI CAMBIO SOLO LE DUE COMPONENTI SCELTE ALLA FINE

  i = floor((N)*rndm());    // SCELGO DUE COMPONENTI DA MESCOLARE IN MODO RANDOM
  j = floor((N)*rndm());
  while (j == i){                // PASSAGGIO NECESSARIO PER IL FUNZIONAMENTO (DEVO AVERE i != j)
    j = floor((N)*rndm());
  }

  a0 = 1-Par->eps2;      // EPS2->0 DEVE AVERE ACCETTANZA UNITARIA (MATRICE IDENTITÀ)
  a1 = 2*rndm()-1;       // COEFFICIENTI DI U = a0*id + i*a_i*sigma_i MATRICE UNITARIA
  a2 = 2*rndm()-1;       // SCELTI IN MODO RANDOM E POI RINORMALIZZATI A a0^2+a1^2+a2^2+a3^2=1
  a3 = 2*rndm()-1;       // IN MODO TALE DA RESTITUIRE EFFETTIVAMENTE UNA MATRICE UNITARIA
  norm = sqrt(pow(a1,2) + pow(a2,2) + pow(a3,2));     // NORMALIZZAZIONE
  x = sqrt(1-pow(a0,2));

  #ifdef DEBUG
  if(errno == EDOM){
    printf("Funzione scalar_trial, radice quadrata\n");
    perror("    errno == EDOM");
  }
  if(errno == ERANGE){
    printf("Funzione scalar_trial, radice quadrata\n");
    perror("    errno == ERANGE");
  }
  if(fetestexcept(FE_INVALID)){
    puts("    FE_INVALID was raised");
    exit(EXIT_FAILURE);
  }
  #endif

  a1 = (a1*x)/norm;             // NORMALIZZAZIONE
  a2 = (a2*x)/norm;
  a3 = (a3*x)/norm;

  phi = (Par->eps2)*(2*rndm() - 1);       // PRENDO DUE ANGOLI A CASO TRA -EPS2;EPS2
  theta = (Par->eps2)*(2*rndm() - 1);     // PER MESCOLARE ULTERIORMENTE I DUE VALORI. NON SO SE È UTILE

  temp1 = ((a0+I*a3)*(Fields->scalar[site][i]) + (I*a1+a2)*(Fields->scalar[site][j]));      // MOLTIPLICO IL VETTORE BIDIM
  temp2 = ((I*a1-a2)*(Fields->scalar[site][i]) + (a0-I*a3)*(Fields->scalar[site][j]));      // PER LA MATRICE U
  temp1 = cexp(I*phi)*temp1;              // AGGIUNGO IL MESCOLAMENTO DELLE FASI
  temp2 = cexp(I*theta)*temp2;

  #ifdef DEBUG
  if(errno == EDOM){
    printf("Funzione scalar_trial, cexp\n");
    perror("    errno == EDOM");
  }
  if(fetestexcept(FE_INVALID)){
    puts("    FE_INVALID was raised");
    exit(EXIT_FAILURE);
  }
  #endif

  trial[i] = temp1;           // CREO IL VETTORE DI PROVA CON LE DUE COMPONENTI i,j
  trial[j] = temp2;           // CAMBIATE DI VALORE
}

// UPDATE METROPOLIS PER IL CAMPO SCALARE
void update_metro_scalar(SystemParam_t *Par, Field_t *Fields){
  int iSite;
  double r, a;
  double complex prod;
  double complex f[N], trial[N], sub[N];

  #ifdef DEBUG
  resetErr();
  #endif

  r=0;
  for(iSite=0;iSite<(Par->V);iSite++){               // AGGIORNO SEQUENZIALMENTE SU TUTTI I SITI
    mean_scalar(Par, Fields, iSite, f);              // CALCOLO CAMPO MEDIO ATTORNO AL SITO SCELTO PER IL TEST DEL CAMPO SCALARE
    scalar_trial(Par, Fields, iSite, trial);         // CAMPO SCALARE DI PROVA
    diff(Par, trial, Fields->scalar[iSite], sub);    // OPERAZIONE INTERMEDIA PER IL CALCOLO DI r
    prod = product(Par, sub, f);                     // OPERAZIONE INTERMEDIA PER IL CALCOLO DI r

    #ifdef DEBUG
    double ene1, delta_loc;
    double ene2, delta_glob;
    delta_loc = -(2*(Par->J)*(N)*creal(prod));
    ene1 = H_dens(Par, Fields);
    #endif

    r = exp(2*(Par->J)*(N)*creal(prod));       // RAPPORTO DI PROBABILITÀ TRA CONFIG DI PROVA E QUELLA DI PARTENZA

    #ifdef DEBUG
    // if(errno == ERANGE){
    //   printf("Funzione update_metro\n");
    //   perror("    errno == ERANGE");
    // }
    if(fetestexcept(FE_OVERFLOW)){
      printf("    FE_OVERFLOW was raised\n");
      exit(EXIT_FAILURE);
    }
    // else if(fetestexcept(FE_UNDERFLOW)){
    //   printf("    FE_UNDERFLOW was raised\n");
    //   // exit(EXIT_FAILURE);
    // }
    else if(fetestexcept(FE_DIVBYZERO)){
      printf("    FE_DIVBYZERO was raised\n");
      // exit(EXIT_FAILURE);
    }
    #endif

    a = rndm();
    if(a < r){                                      // ACCETTO CON PROBABILITÀ r IL CAMBIO (SE r>1 ACCETTO SICURO)
      copy(Par, trial, Fields->scalar[iSite]);      // CAMBIO AVVENUTO
      acc2 = acc2 +1.0;                             // AGGIORNO IL NUMERO DI PASSI ACCETTATI

      #ifdef DEBUG
      ene2 = H_dens(Par, Fields);
      delta_glob = ene2 - ene1;
      if(fabs(delta_loc/Par->V - delta_glob)>1.0e-12){
        if((fabs(delta_loc/Par->V - delta_glob)>1.0e-12)&(fabs(delta_loc/Par->V - delta_glob)<1.0e-11)){
          err1 = err1+1;
        }
        else if((fabs(delta_loc/Par->V - delta_glob)>1.0e-11)){
          err2=err2+1;
        }
      }
      #endif
    }
  }
}

// UPDATE MICROCANONICO PER IL CAMPO SCALARE
// SOSTANZIALMENTE Z_x0 ----> b*f/|f|^2 - Z_x0
// DOVE b=2*RE(BAR(Z_x0)*f)
void update_micro(SystemParam_t *Par, Field_t *Fields){
  int iSite, j;
  double complex f[N], trial[N], prod1;
  double norm, b;

  #ifdef DEBUG
  resetErr();
  #endif

  for(iSite=0; iSite<(Par->V); iSite++){      // UPDATE MICROCANONICO SEQUENZIALE SU TUTTI I SITI
    #ifdef DEBUG
    double ene1;
    ene1 = H_dens(Par, Fields);
    #endif

    mean_scalar(Par, Fields, iSite, f);                             // CALCOLO f CAMPO MEDI0
    b = 2*creal(product(Par, Fields->scalar[iSite], f));            // CALCOLO b=2*RE(BAR(Z_x0)*f)
    norm = creal(product(Par, f,f));                                // CALCOLO |f|^2

    if(norm<1.0e-12){                                     // SE LA NORMA È TROPPO PICCOLA ALLORA SOSTITUISCO CON UN CAMPO RANDOM
      for(j=0;j<(N);j++){                            // SARÀ GIUSTO?
        trial[j] = 2*rndm()-1 + I*(2*rndm() - 1);
      }
      for(j=0;j<(N);j++){
        trial[j] = trial[j]/sqrt(creal(product(Par, trial, trial)));
      }
      copy(Par, trial, Fields->scalar[iSite]);
      return;
    }

    for(j=0;j<(N);j++){                                  // CREO IL CAMPO DA SOSTITUIRE
      trial[j] = f[j]*b/norm - (Fields->scalar[iSite][j]);
    }
    copy(Par, trial, Fields->scalar[iSite]);
    #ifdef DEBUG
    double ene2;
    ene2 = H_dens(Par, Fields);
    if(fabs(ene1-ene2)>1e-12){
      if((fabs(ene1-ene2)>1.0e-12)&(fabs(ene1-ene2)<1.0e-11)){
        err1 = err1+1;
        // printf("ERRORE NELL'UPDATE MICROCANONICO = %.13lf\n", fabs(ene1-ene2));
      }
      else if((fabs(ene1-ene2)>1.0e-11)){
        // printf("ERRORE NELL'UPDATE MICROCANONICO = %.13lf\n", fabs(ene1-ene2));
        err2=err2+1;
      }
    }
    #endif
  }
}

void update_metro_gauge(SystemParam_t *Par, Field_t *Fields){
  int iSite, mu, k;
  double a, r, f_g, trial;
  double complex prod;
  double complex near[N];

  #ifdef DEBUG
  resetErr();
  #endif

  r = 0;
  for(iSite=0; iSite<(Par->V); iSite++){                                  // AGGIORNO SEQUENZIALMENTE SUI SITI/DIREZIONI
    for(mu=0;mu<(D);mu++){
      f_g = mean_gauge(Par, Fields, iSite, mu);                           // CAMPO MEDIO PER L'UPDATE DEL LINK
      for(k=0;k<(N);k++){
        near[k] = nearest_scalar(Fields, iSite, mu, k, TRUE);             // CREO IL VETTORE CAMPO SCALARE VICINO(ISITE + MU)
      }
      prod = product(Par, Fields->scalar[iSite], near);                   // PRODOTTO CHE ENTRA IN r
      trial = Fields->gauge[iSite][mu] + (Par->eps1)*(2*rndm() - 1);      // CAMPO DI GAUGE DI PROVA


      #ifdef DEBUG
      double delta_glob, ene1, ene2, delta_loc;
      ene1 = H_dens(Par, Fields);
      delta_loc = -2*(Par->J)*(N)*creal(prod * (cexp(I*trial) - cexp(I*(Fields->gauge[iSite][mu])))) +2*((Par->K)/2.0)*(((D)-1)*(pow(trial,2) - pow(Fields->gauge[iSite][mu],2)) + f_g*(trial - Fields->gauge[iSite][mu]));
      #endif

      // RAPPORTO r DI PROBABILITÀ TRA CONFIGURAZIONE DI PROVA E CONFIG INIZIALE
      r = exp(2*(Par->J)*(N)*creal(prod * (cexp(I*trial) - cexp(I*(Fields->gauge[iSite][mu])))) -2*((Par->K)/2.0)*(((D)-1)*(pow(trial,2) - pow(Fields->gauge[iSite][mu],2)) + f_g*(trial - Fields->gauge[iSite][mu])));

      #ifdef DEBUG
      if(errno == EDOM){
        printf("Funzione update_metro_gauge, cexp\n");
        perror("    errno == EDOM");
      }
      if(errno == ERANGE){
        printf("Funzione update_metro_gauge, cexp\n");
        perror("    errno == ERANGE");
      }
      if(fetestexcept(FE_INVALID)){
        puts("    FE_INVALID was raised");
        exit(EXIT_FAILURE);
      }
      if(fetestexcept(FE_OVERFLOW)){
        puts("    FE_OVERFLOW was raised");
        exit(EXIT_FAILURE);
      }
      #endif

      a = rndm();
      if(a<r){                              // ACCETTO LA NUOVA CONFIGURAZIONE CON PROBABILITÀ r (SE r>1 ACCETTO SICURO)
        Fields->gauge[iSite][mu] = trial;
        acc1 = acc1 + 1.0;                  // AGGIORNO IL NUMERO DI PASSI ACCETTATI

        #ifdef DEBUG
        ene2 = H_dens(Par, Fields);
        delta_glob = ene2 - ene1;
        if((fabs(delta_loc/Par->V - delta_glob)>1.0e-12)&(fabs(delta_loc/Par->V - delta_glob)<1.0e-11)){
          // printf("ERRORE NELL'UPDATE DEL CAMPO DI GAUGE\n");
          // printf("delta_glob=%.13lf\n", delta_glob);
          // printf("delta_loc =%.13lf\n", delta_loc);
          // printf("|delta_loc-delta_glob| = %.13lf\n", fabs(delta_loc/Par->V -delta_glob));
          err1 = err1+1;
          // exit(EXIT_FAILURE);
        }
        else if((fabs(delta_loc - delta_glob)>1.0e-11)){
          // printf("|delta_loc-delta_glob| = %.13lf\n", fabs(delta_loc/Par->V -delta_glob));
          err2=err2+1;
        }
        #endif
      }
    }
  }
}

// PROCEDURA DI TERMALIZZAZIONE, DA CONTROLLARE CON MACRO !!!
void thermalization(SystemParam_t *Par, Field_t *Fields, int count){   // INSERIRE UNA PARTE DI GESTIONE AUTOMATICA DELLE ACCETTANZE
  int i, j;
  double a1, a2;                                         // PER QUESTA PARTE FARE IN MODO DI CONTROLLARE SE TERM O NO
  bool_t ctrl_1, ctrl_2, ctrl_3, ctrl_4;

  #ifdef RESUME       // SE È DEFINITO RESUME NON TERMALIZZO, RIPRENDO DA CONFIGURAZIONE SALVATA
  return;
  #endif

  if(count>=10){
    return;
  }

  for(i=0;i<(Par->iTerm); i++){       // PER UN NUMERO iTerm DI VOLTE RIPETO L'UPDATE SCALARE E DI GAUGE, CON 3 MICROCANONICI
    update_metro_scalar(Par, Fields);
    update_metro_gauge(Par, Fields);
    for(j=0;j<(Par->iOverr);j++){
      update_micro(Par, Fields);
    }

    if(i==((Par->iTerm)/2)){     // AGGIUSTO A MANO IL MODULO DEL CAMPO SCALARE CHE POTREBBE ESSERE CAMBIATO PER ERRORE NUMERICO
      renormalize(Par, Fields);
    }
  }

  a1 = acc1/((Par->iTerm)*(D)*(Par->V));     // ACCETTANZE NELLA PARTE DI TERMALIZZAZIONE
  a2 = acc2/((Par->iTerm)*(Par->V));
  ctrl_acceptance(a1, &ctrl_1, &ctrl_2);          // CONTROLLO LE ACCETTANZE
  ctrl_acceptance(a2, &ctrl_3, &ctrl_4);
  modify_eps(Par, Fields, ctrl_1, ctrl_2, ctrl_3, ctrl_4, count);    // SE LE ACCETTANZE NON PASSANO IL CHECK MODIFICO I PARAMETRI eps1, eps2
  #ifdef DEBUG
  err1=0;
  err2 =0;
  #endif
}

// PROCEDURA PER MODIFICA AUTOMATICA DEI PARAMETRI DI ACCETTANZA
void modify_eps(SystemParam_t *Par, Field_t *Fields, bool_t ctrl_1, bool_t ctrl_2, bool_t ctrl_3, bool_t ctrl_4, int count){
  if((ctrl_1==TRUE)&&(ctrl_3==TRUE)&&(ctrl_2==FALSE)&&(ctrl_4==FALSE)){         // LE ACCETTANZE SONO BUONE
    acc1=0;
    acc2=0;
    return;
  }
  else if((ctrl_1==TRUE)&&(ctrl_3==FALSE)&&(ctrl_2==FALSE)&&(ctrl_4==FALSE)){   // acc1 OK, acc2 TROPPO PICCOLA
    Par->eps2 = Par->eps2 - 0.05;
    acc1 = 0;
    acc2 = 0;
    thermalization(Par, Fields, count+1);
  }
  else if((ctrl_1==TRUE)&&(ctrl_3==FALSE)&&(ctrl_2==FALSE)&&(ctrl_4==TRUE)){    // acc1 OK, acc2 TROPPO GRANDE
    Par->eps2 = Par->eps2 + 0.05;
    acc1 = 0;
    acc2 = 0;
    thermalization(Par, Fields, count+1);
  }
  else if((ctrl_1==FALSE)&&(ctrl_3==TRUE)&&(ctrl_2==FALSE)&&(ctrl_4==FALSE)){   // acc1 TROPPO PICCOLA, acc2 OK
    Par->eps1 = Par->eps1 - 0.001;
    acc1 = 0;
    acc2 = 0;
    thermalization(Par, Fields, count+1);
  }
  else if((ctrl_1==FALSE)&&(ctrl_3==TRUE)&&(ctrl_2==TRUE)&&(ctrl_4==FALSE)){    // acc1 TROPPO GRANDE, acc2 OK
    Par->eps1 = Par->eps1 + 0.001;
    acc1 = 0;
    acc2 = 0;
    thermalization(Par, Fields, count+1);
  }
  else if((ctrl_1==FALSE)&&(ctrl_3==FALSE)&&(ctrl_2==FALSE)&&(ctrl_4==FALSE)){  // acc1 TROPPO PICCOLA, acc2 TROPPO PICCOLA
    Par->eps1 = Par->eps1 - 1/(Par->K);
    Par->eps2 = Par->eps2 - 0.05;
    acc1 = 0;
    acc2 = 0;
    thermalization(Par, Fields, count+1);
  }
  else if((ctrl_1==FALSE)&&(ctrl_3==FALSE)&&(ctrl_2==FALSE)&&(ctrl_4==TRUE)){   // acc1 TROPPO PICCOLA, acc2 TROPPO GRANDE
    Par->eps1 = Par->eps1 - 0.001;
    Par->eps2 = Par->eps2 + 0.05;
    acc1 = 0;
    acc2 = 0;
    thermalization(Par, Fields, count+1);
  }
  else if((ctrl_1==FALSE)&&(ctrl_3==FALSE)&&(ctrl_2==TRUE)&&(ctrl_4==FALSE)){   // acc1 TROPPO GRANDE, acc2 TROPPO PICCOLA
    Par->eps1 = Par->eps1 + 0.001;
    Par->eps2 = Par->eps2 - 0.05;
    acc1 = 0;
    acc2 = 0;
    thermalization(Par, Fields, count+1);
  }
  else if((ctrl_1==FALSE)&&(ctrl_3==FALSE)&&(ctrl_2==TRUE)&&(ctrl_4==TRUE)){    // acc1 TROPPO GRANDE, acc2 TROPPO GRANDE
    Par->eps1 = Par->eps1 + 0.001;
    Par->eps2 = Par->eps2 + 0.05;
    acc1 = 0;
    acc2 = 0;
    thermalization(Par, Fields, count+1);
  }
  else{
    printf("ERRORE NELLA TERMALIZZAZIONE\n");
    exit(EXIT_FAILURE);
  }
}

// PROCEDURA DI UPDATE DELLE CONFIGURAZIONI, PER IDEC VOLTE
void update_configurations(SystemParam_t *Par, Field_t *Fields){
  int i, j;

  for(i=0;i<(Par->iDec); i++){            // RIPETO PER IDEC VOLTE GLI UPDATE 1 SCALARE, 1 GAUGE, 3 MICRO
    update_metro_scalar(Par, Fields);
    update_metro_gauge(Par, Fields);
    for(j=0;j<(Par->iOverr);j++){
      update_micro(Par, Fields);
    }

    if((i%10)==0){     // AGGIUSTO A MANO IL MODULO DEL CAMPO SCALARE CHE POTREBBE ESSERE CAMBIATO PER ERRORE NUMERICO
      renormalize(Par, Fields);
    }
  }
}
