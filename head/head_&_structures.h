#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <time.h>
#include "dSFMT.h"
#include <math.h>
#include <complex.h>
#include <errno.h>
#include <fenv.h>
#include <sys/types.h>
#include <sys/stat.h>
#include "../config.h"

// MACRO
// #define DEBUG        // MACRO DI DEBUG
// #define RESUME       // MACRO DI RESUME, SE È DEFINITA ALLORA IL SISTEMA RIPARTE DALLA CONFIGURAZIONE SALVATA
#define PI 3.14159265358979   // PI GRECO
// #define N  2             // LE DEFINISCO DIRETTAMENTE DA TERMINALE TRAMITE CONFIGURE
// #define D  2             // LE DEFINISCO DIRETTAMENTE DA TERMINALE TRAMITE CONFIGURE

typedef struct {  // DEFINISCO IL TIPO SystemParam_t STRUTTURATO NEL SEGUENTE MODO
  int L;          // NUMERO DI SITI PER LATO
  int V;          // VOLUME DEL SISTEMA, CIOÈ NUMERO TOTALE DI SITI
  double J;       // ACCOPPIAMENTO zUz
  double K;       // ACCOPPIAMENTO DEL TERMINE DI PLACCHETTA
  double eps1;    // PARAMETRO CHE REGOLA L'ACCETTANZA PER LE VARIABILI DI LINK
  double eps2;    // PARAMETRO CHE REGOLA L'ACCETTANZA PER LE VARIABILI DI CAMPO SCALARI
  int iTerm;      // NUMERO DI UPDATE DI TERMALIZZAZIONE
  int iDec;       // NUMERO DI UPDATE PER DECORRELARE LA CATENA
  int iMis;       // NUMERO DI MISURAZIONI
  int iOverr;      // NUMERO DI PASSI DI OVERRELAXATION
} SystemParam_t;

typedef struct {              // DEFINISCO IL TIPO Field_t STRUTTURATO NEL SEGUENTO MODO
  double complex **scalar;    // CAMPO SCALARE COMPLESSO [V][N]
  double **gauge;             // CAMPO DI GAUGE REALE [V][D]
} Field_t;

typedef struct {              // DEFINISCO LA STRUTTURA DELLE OSSERVABILI
  double spin_ene_density;    // DENSITÀ DI ENERGIA DELLA PARTE DI H_Z, CAMPO SCALARE
  double gauge_ene_density;   // DENSITÀ DI ENERGIA DELLA PARTE DI GAUGE, H_G
  double ene_density;         // DENSITÀ DI ENERGIA DEL SISTEMA
  double susc;                // SUSCETTIVITÀ
  double G_pm;                // G_TILDE_PM CHE ENTRA NELLA COSTRUZIONE DELLA LUNGHEZZA DI CORRELAZIONE
  double mu2;                 // MU2 CHE ENTRA NELLA COSTRUZIONE DEL CUMULANTE DI BINDER
} Obs_t;

typedef enum {FALSE=0, TRUE=1} bool_t;      // DEFINISCO IL TIPO BOOLEANO, UTILE PER ALCUNE PROCEDURE

extern int **npp, **nmm;                    // VARIABILE ESTERNA CHE NEL MAIN È GLOBALE E CHE MEMORIZZA I PRIMI VICINI
extern dsfmt_t dsfmt;                       // VARIABILE DI SUPPORTO PER IL DSFMT
extern double acc1, acc2, err1, err2;       // VARIABILI PER L'ACCETTANZA E PER EVENTUALI ERRORI NUMERICI

// FUNZIONI E PROCEDURE DI UTILITY
double rndm();                                      // FUNZIONE CHE CHIAMA IL GENERATORE RANDOM E GENERA UN NUM RANDOM TRA 0 E 1
bool_t bc(int initPos, int finPos, bool_t ctrl);    // FUNZIONE CHE IMPLEMENTA LE BC,
                                                    // ATTENZIONE FUNZIONANO SOLO SE INITPOS E FINPOS SONO PRIMI VICINI
void diff(SystemParam_t *Par, double complex *vec1, double complex *vec2, double complex *result);  // PROCEDURA CHE IMPLEMENTA LA DIFFERENZA
                                                                                                    // TRA DUE VETTORI DI SU(N)
void copy(SystemParam_t *Par, double complex *vec1, double complex *vec2);                          // FUNZIONE CHE COPIA VEC1 IN VEC2
double complex product(SystemParam_t *Par, double complex *vec1, double complex *vec2);             // FUNZIONE CHE IMPLEMENTA IL PRODOTTO BAR(VEC1)*VEC2
void resetErr();                                                                                    // PROCEDURA CHE RESETTA LE VARIABILI DI ERROR HANDLING
void ctrl_acceptance(double ac, bool_t *ctrl_1, bool_t *ctrl_2);                                    // PROCEDURA DI CONTROLLO ACCETTANZE USATA NEL CALCOLO
                                                                                                    // AUTOMATICO DEI PARAMETRI DI ACCETTANZA
// FUNZIONI E PROCEDURE PER L'INIZIALIZZAZIONE E LA DEALLOCAZIONE
void geometry(SystemParam_t *Par);                                              // PROCEDURA CHE IMPLEMENTA NPP E NMM
void read_from_input_Param(SystemParam_t *Par);                                 // PROCEDURA PER LEGGERE I PARAMETRI DA input.txt
void allocation(SystemParam_t *Par, Field_t *Fields);                           // PROCEDURA DI ALLOCAZIONE DINAMICA PER LE STRUTTURE E NPP NMM
void initializeFields(SystemParam_t *Par, Field_t *Fields);                     // PROCEDURA DI INIZIALIZZAZIONE DEI CAMPI
void initializeObs(Obs_t *Obs);                                                 // PROCEDURA DI INIZIALIZZAZIONE A ZERO DELLE OSSERVABILI
void initializeSystem(SystemParam_t *Par, Field_t *Fields, Obs_t *Obs);         // PROCEDURA DI INIZIALIZZIONE DEL SISTEMA
void deallocation(Field_t *Fields);                                             // DEALLOCAZIONE DELLA MEMORIA

// FUNZIONI E PROCEDURE PER LA MANIPOLAZIONE DEI CAMPI, IN PARTICOLARE CAMPO MEDIO E PRIMI VICINI
double complex nearest_scalar(Field_t *Fields, int site, int dir, int component, bool_t ctrl);      // FUNZIONE CHE RESTITUISCE LA COMPONENTE component DEL CAMPO SCALARE NELLA DIREZIONE DIR VICINO AL SITO. SE TRUE ALLORA + ALTRIMENTI -
double nearest_gauge(Field_t *Fields, int site, int dir, int mu, bool_t ctrl);                      // STESSO FUNZIONAMENTO DI QUELLA SOPRA MA COL CAMPO DI GAUGE
double second_nearest_gauge(Field_t *Fields, int site, int nu, int mu);                             // FUNZIONE PER PER IL SECONDO VICINO, USARE SOLO IN DETERMINATI CONTESTI
void mean_scalar(SystemParam_t *Par, Field_t *Fields, int site, double complex *mean);              // CAMPO MEDIO PER L'UPDATE SCALARE
double mean_gauge(SystemParam_t *Par, Field_t *Fields, int site, int mu);                           // CAMPO MEDIO PER L'UPDATE DI GAUGE
void renormalize(SystemParam_t *Par, Field_t *Fields);                                              // RINORMALIZZAZIONE AD 1 DEL CAMPO SCALARE PER EVITARE ERRORI NUMERICI

// PROCEDURE DI UPDATE METROPOLIS E MICROCANONICO PER ENTRAMBI I CAMPI
void scalar_trial(SystemParam_t *Par, Field_t *Fields, int site, double complex *trial);                            // COSTRUZIONE DELLO STATO DI PROVA PER L'UPDATE SCALARE
void update_metro_scalar(SystemParam_t *Par, Field_t *Fields);                                                      // UPDATE METROPOLIS PER IL CAMPO SCALARE
void update_micro(SystemParam_t *Par, Field_t *Fields);                                                             // UPDATE MICROCANONICO PER IL CAMPO SCALARE
void update_metro_gauge(SystemParam_t *Par, Field_t *Fields);                                                       // UPDATE METROPOLIS PER IL CAMPO DI GAUGE
void thermalization(SystemParam_t *Par, Field_t *Fields, int count);                                                           // PROCEDURA DI TERMALIZZAZIONE
void modify_eps(SystemParam_t *Par, Field_t *Fields, bool_t ctrl_1, bool_t ctrl_2, bool_t ctrl_3, bool_t ctrl_4, int count);   // PROCEDURA DI MODIFICA DEI PARAMETRI DI ACCETTANZA, USATA NELLA TERMALIZZAZIONE
void update_configurations(SystemParam_t *Par, Field_t *Fields);                                                    // PROCEDURA CHE RICHIAMA INSIEME I VARI UPDATE

// FUNZIONI E PROCEDURE PER LE MISURAZIONI SUL RETICOLO
double H_z_dens(SystemParam_t *Par, Field_t *Fields);                 // FUNZIONE PER LA DENSITÀ DI ENERGIA DEL CAMPO SCALARE
double H_g_dens(SystemParam_t *Par, Field_t *Fields);                 // FUNZIONE PER LA DENSITÀ DEL CAMPO DI GAUGE
double H_dens(SystemParam_t *Par, Field_t *Fields);                   // FUNZIONE PER LA DENSITÀ ENERGETICA DEL SISTEMA
void measure(SystemParam_t *Par, Field_t *Fields, Obs_t *Obs);        // PROCEDURA CHE RICHIAMA LE VARIE MISURE DA FARE SUL RETICOLO
void susc_mu2(SystemParam_t *Par, Field_t *Fields, double *susc, double *mu2);   // PROCEDURA CHE CALCOLA SUSC E MU2
void G_pm_matrix(SystemParam_t *Par, Field_t *Fields, double *G_pm);   // PROCEDURA PER IL CALCOLO DI G_PM CON FORMULAIZONE MATRICIALE

// FUNZIONI E PROCEDURE CHE REGOLANO IL SALVATAGGIO DEL SISTEMA
void writeFields(SystemParam_t *Par, Field_t *Fields);                // PROCEDURA CHE SALVA LA CONFIGURAZIONE DI CAMPO SU FILE
void readFields(SystemParam_t *Par, Field_t *Fields);                 // PROCEDURA CHE LEGGE LA CONFIGURAZIONE SALVATA
void writeEps(SystemParam_t *Par);                                    // PROCEDURA CHE SALVA I PARAMETRI DI ACCETTANZA TROVATI DALLA TERMALIZZAZIONE
void readEps(SystemParam_t *Par);                                     // PROCEDURA CHE LEGGE I PARAMETRI DI ACCETTANZA PRECEDENTEMENTE SALVATI
void writeObs(FILE *fptr, Obs_t *Obs);                                // PROCEDURA CHE SALVA SU FILE LE OSSERVABILI MISURATE
