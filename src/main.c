// PROGRAMMA DI SIMULAZIONE NUMERICA PER
// UN MODELLO ABELIAN-HIGGS CON FORMULAZIONE
// NON COMPATTA DEL GRUPPO DI GAUGE U(1)

// INCLUDO HEADER DOVE HO DEFINIZIONI DELLE STRUTTURE, MACRO E PROTOTIPI DELLE FUNZIONI
#include "../head/head_&_structures.h"

// STRUTTURE E VARIABILI GLOBALI
dsfmt_t dsfmt;                          // VARIABILE GLOBALE PER IL GENERATORE DI NUMERI RANDOM
double acc1, acc2, err1=0, err2=0;      // VARIABILI GLOBALI PER AGGIORNARE I PASSI ACCETTATI ED EVENTUALI ERRORI NUMERICI
int **npp, **nmm;                       // ARRAY CHE MEMORIZZANO I PRIMI VICINI

struct stat st = {0};                   // NECESSARIA AL CONTROLLO DELLE CARTELLE

// MAIN DEL PROGRAMMA
int main(int argc, char const *argv[]){
  SystemParam_t Param;                  // STRUTTURA DEI PARAMETRI
  Field_t Config;                       // STRUTTURA DEI CAMPI
  Obs_t Oss;                            // STRUTTURA DELLE OSSERVABILI
  int seed;                             // SEME DEL GENERATORE RANDOM
  int i, count=0;                       // VARIABILE INTERA PER CICLO FOR DELLE MISURE
  char buffer[64];                      // BUFFER DI CARATTERI AUSILIARIO PER IL NOME DEL FILE DI USCITE DELLE MISURE
  FILE *fptr;                           // PUNTATORE A FILE PER SCRIVERE I DATI

  // INIZIALIZZO I PARAMETRI DI SISTEMA
  initializeSystem(&Param, &Config, &Oss, argv[1]);

  // INIZIALIZZO GEN RANDOM
  dsfmt_init_gen_rand(&dsfmt, Param.dSFMT_seed);    // INIZIALIZZO IL GENERATORE

  // TERMALIZZAZIONE
  thermalization(&Param, &Config, count);

  // SALVO CONFIGURAZIONE (DA CAPIRE OGNI QUANTO RICHIAMARLA)
  writeEps(&Param);
  writeFields(&Param, &Config);

  // ESEGUO DELLE MISURAZIONI DI ENERGIA OGNI CHIAMATA DI UP_CONF CHE RIPETE IDEC VOLTE L'UPDATE
  fptr = fopen(Param.data_file, "a");
  if (fptr == NULL) {
    perror("Errore in apertura per la scrittura dati");
    exit(1);
  }
  if (Param.iStart == 0){
    fprintf(fptr, "%d\t%d\t%d\t%lf\t%lf\n", Param.L, Param.V, N, Param.J, Param.K);           // SCRIVO I PARAMETRI SU FILE CHE POI RILEGGO NELL'ANALISI
    fprintf(fptr, "#en_sp_dens\ten_g_dens\tene_density\tsusceptib\tG_pm_tilde\tmu2\n\n");     // PRIMA LINEA SU FILE DELLE MISURE PER CAPIRE COSA SONO LE COLONNE DI DATI
  }
  
  for(i=0;i<(Param.iMis);i++){
    update_configurations(&Param, &Config);             // UPDATE DELLE CONFIGURAZIONI PER iDec VOLTE PRIMA DELLA MISURA
    measure(&Param, &Config, &Oss);                     // MISURE DELLE VARIE GRANDEZZE SU RETICOLO
    writeObs(fptr, &Oss);                               // SALVATAGGIO DELLE MISURE
    if((i % (Param.iBackup)) == 0){
      writeFields(&Param, &Config);                     // OGNI iBackup MISURE SALVO UNA CONFIGURAZIONE
    }
  }
  writeFields(&Param, &Config);                         // SALVO LA CONFIGURAZIONE PRIMA DI CHIUDERE

  // CHIUDO IL FILE DELLE NMISURE E DEALLOCO LA MEMORIA DINAMICA
  fclose(fptr);
  deallocation(&Config);

  // ALCUNI MESSAGGI DI USCITA TRA CUI ACCETTANZE E MESSAGGIO DI FINE PROGRAMMA
  writeLogs(&Param);

  // CONVIENE ANCHE INSERIRE IL NOME DEL FILE DI SALVATAGGIO DA INPUT

  return(EXIT_SUCCESS);
}
