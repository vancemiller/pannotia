//=====================================================================
//	MAIN FUNCTION
//=====================================================================
__device__ void kernel_ecc_2(float timeinst, float* d_initvalu, float* d_finavalu, int valu_offset,
    float* d_params) {

  //=====================================================================
  //	VARIABLES
  //=====================================================================

  // input parameters
  float cycleLength;

  // variable references				// GET VARIABLES FROM MEMORY AND SAVE LOCALLY !!!!!!!!!!!!!!!!!!
  int offset_1;
  int offset_2;
  int offset_3;
  int offset_4;
  int offset_5;
  int offset_6;
  int offset_7;
  int offset_8;
  int offset_9;
  int offset_10;
  int offset_11;
  int offset_12;
  int offset_13;
  int offset_14;
  int offset_15;
  int offset_16;
  int offset_17;
  int offset_18;
  int offset_19;
  int offset_20;
  int offset_21;
  int offset_22;
  int offset_23;
  int offset_24;
  int offset_25;
  int offset_26;
  int offset_27;
  int offset_28;
  int offset_29;
  int offset_30;
  int offset_31;
  int offset_32;
  int offset_33;
  int offset_34;
  int offset_35;
  int offset_36;
  int offset_37;
  int offset_38;
  int offset_39;
  int offset_40;
  int offset_41;
  int offset_42;
  int offset_43;
  int offset_44;
  int offset_45;
  int offset_46;

  // stored input array
  float d_initvalu_1;
  float d_initvalu_2;
  float d_initvalu_3;
  float d_initvalu_4;
  float d_initvalu_5;
  float d_initvalu_6;
  float d_initvalu_7;
  float d_initvalu_8;
  float d_initvalu_9;
  float d_initvalu_10;
  float d_initvalu_11;
  float d_initvalu_12;
  float d_initvalu_13;
  float d_initvalu_14;
  float d_initvalu_15;
  float d_initvalu_16;
  float d_initvalu_17;
  float d_initvalu_18;
  float d_initvalu_19;
  float d_initvalu_20;
  float d_initvalu_21;
  // float d_initvalu_22;
  float d_initvalu_23;
  float d_initvalu_24;
  float d_initvalu_25;
  float d_initvalu_26;
  float d_initvalu_27;
  float d_initvalu_28;
  float d_initvalu_29;
  float d_initvalu_30;
  float d_initvalu_31;
  float d_initvalu_32;
  float d_initvalu_33;
  float d_initvalu_34;
  float d_initvalu_35;
  float d_initvalu_36;
  float d_initvalu_37;
  float d_initvalu_38;
  float d_initvalu_39;
  float d_initvalu_40;
  // float d_initvalu_41;
  // float d_initvalu_42;
  // float d_initvalu_43;
  // float d_initvalu_44;
  // float d_initvalu_45;
  // float d_initvalu_46;

  // matlab constants undefined in c
  float pi;

  // Constants
  float R;																			// [J/kmol*K]
  float Frdy;																		// [C/mol]
  float Temp;																		// [K] 310
  float FoRT;																		//
  float Cmem;																		// [F] membrane capacitance
  float Qpow;

  // Cell geometry
  float cellLength;																	// cell length [um]
  float cellRadius;																	// cell radius [um]
  // float junctionLength;																// junc length [um]
  // float junctionRadius;																// junc radius [um]
  // float distSLcyto;																	// dist. SL to cytosol [um]
  // float distJuncSL;																	// dist. junc to SL [um]
  // float DcaJuncSL;																	// Dca junc to SL [cm^2/sec]
  // float DcaSLcyto;																	// Dca SL to cyto [cm^2/sec]
  // float DnaJuncSL;																	// Dna junc to SL [cm^2/sec]
  // float DnaSLcyto;																	// Dna SL to cyto [cm^2/sec]
  float Vcell;																		// [L]
  float Vmyo;
  float Vsr;
  float Vsl;
  float Vjunc;
  // float SAjunc;																		// [um^2]
  // float SAsl;																		// [um^2]
  float J_ca_juncsl;																	// [L/msec]
  float J_ca_slmyo;																	// [L/msec]
  float J_na_juncsl;																	// [L/msec]
  float J_na_slmyo;																	// [L/msec]

  // Fractional currents in compartments
  float Fjunc;
  float Fsl;
  float Fjunc_CaL;
  float Fsl_CaL;

  // Fixed ion concentrations
  float Cli;																			// Intracellular Cl  [mM]
  float Clo;																			// Extracellular Cl  [mM]
  float Ko;																			// Extracellular K   [mM]
  float Nao;																			// Extracellular Na  [mM]
  float Cao;																			// Extracellular Ca  [mM]
  float Mgi;																			// Intracellular Mg  [mM]

  // Nernst Potentials
  float ena_junc;																	// [mV]
  float ena_sl;																		// [mV]
  float ek;																			// [mV]
  float eca_junc;																	// [mV]
  float eca_sl;																		// [mV]
  float ecl;																			// [mV]

  // Na transport parameters
  float GNa;																			// [mS/uF]
  float GNaB;																		// [mS/uF]
  float IbarNaK;																		// [uA/uF]
  float KmNaip;																		// [mM]
  float KmKo;																		// [mM]
  // float Q10NaK;
  // float Q10KmNai;

  // K current parameters
  float pNaK;
  float GtoSlow;																		// [mS/uF]
  float GtoFast;																		// [mS/uF]
  float gkp;

  // Cl current parameters
  float GClCa;																		// [mS/uF]
  float GClB;																		// [mS/uF]
  float KdClCa;																		// [mM]																// [mM]

  // I_Ca parameters
  float pNa;																			// [cm/sec]
  float pCa;																			// [cm/sec]
  float pK;																			// [cm/sec]
  // float KmCa;																		// [mM]
  float Q10CaL;

  // Ca transport parameters
  float IbarNCX;																		// [uA/uF]
  float KmCai;																		// [mM]
  float KmCao;																		// [mM]
  float KmNai;																		// [mM]
  float KmNao;																		// [mM]
  float ksat;																			// [none]
  float nu;																			// [none]
  float Kdact;																		// [mM]
  float Q10NCX;																		// [none]
  float IbarSLCaP;																	// [uA/uF]
  float KmPCa;																		// [mM]
  float GCaB;																		// [uA/uF]
  float Q10SLCaP;																// [none]																	// [none]

  // SR flux parameters
  float Q10SRCaP;																	// [none]
  float Vmax_SRCaP;																	// [mM/msec] (mmol/L cytosol/msec)
  float Kmf;																			// [mM]
  float Kmr;																			// [mM]L cytosol
  float hillSRCaP;																	// [mM]
  float ks;																			// [1/ms]
  float koCa;																		// [mM^-2 1/ms]
  float kom;																			// [1/ms]
  float kiCa;																		// [1/mM/ms]
  float kim;																			// [1/ms]
  float ec50SR;																		// [mM]

  // Buffering parameters
  float Bmax_Naj;																	// [mM]
  float Bmax_Nasl;																	// [mM]
  float koff_na;																		// [1/ms]
  float kon_na;																		// [1/mM/ms]
  float Bmax_TnClow;																	// [mM], TnC low affinity
  float koff_tncl;																	// [1/ms]
  float kon_tncl;																	// [1/mM/ms]
  float Bmax_TnChigh;																// [mM], TnC high affinity
  float koff_tnchca;																	// [1/ms]
  float kon_tnchca;																	// [1/mM/ms]
  float koff_tnchmg;																	// [1/ms]
  float kon_tnchmg;																	// [1/mM/ms]
  // float Bmax_CaM;																	// [mM], CaM buffering
  // float koff_cam;																	// [1/ms]
  // float kon_cam;																		// [1/mM/ms]
  float Bmax_myosin;																	// [mM], Myosin buffering
  float koff_myoca;																	// [1/ms]
  float kon_myoca;																	// [1/mM/ms]
  float koff_myomg;																	// [1/ms]
  float kon_myomg;																	// [1/mM/ms]
  float Bmax_SR;																		// [mM]
  float koff_sr;																		// [1/ms]
  float kon_sr;																		// [1/mM/ms]
  float Bmax_SLlowsl;																// [mM], SL buffering
  float Bmax_SLlowj;																	// [mM]
  float koff_sll;																	// [1/ms]
  float kon_sll;																		// [1/mM/ms]
  float Bmax_SLhighsl;																// [mM]
  float Bmax_SLhighj;																// [mM]
  float koff_slh;																	// [1/ms]
  float kon_slh;																		// [1/mM/ms]
  float Bmax_Csqn;																	// 140e-3*Vmyo/Vsr; [mM]
  float koff_csqn;																	// [1/ms]
  float kon_csqn;																	// [1/mM/ms]

  // I_Na: Fast Na Current
  float am;
  float bm;
  float ah;
  float bh;
  float aj;
  float bj;
  float I_Na_junc;
  float I_Na_sl;
  // float I_Na;

  // I_nabk: Na Background Current
  float I_nabk_junc;
  float I_nabk_sl;
  // float I_nabk;

  // I_nak: Na/K Pump Current
  float sigma;
  float fnak;
  float I_nak_junc;
  float I_nak_sl;
  float I_nak;

  // I_kr: Rapidly Activating K Current
  float gkr;
  float xrss;
  float tauxr;
  float rkr;
  float I_kr;

  // I_ks: Slowly Activating K Current
  float pcaks_junc;
  float pcaks_sl;
  float gks_junc;
  float gks_sl;
  float eks;
  float xsss;
  float tauxs;
  float I_ks_junc;
  float I_ks_sl;
  float I_ks;

  // I_kp: Plateau K current
  float kp_kp;
  float I_kp_junc;
  float I_kp_sl;
  float I_kp;

  // I_to: Transient Outward K Current (slow and fast components)
  float xtoss;
  float ytoss;
  float rtoss;
  float tauxtos;
  float tauytos;
  float taurtos;
  float I_tos;

  //
  float tauxtof;
  float tauytof;
  float I_tof;
  float I_to;

  // I_ki: Time-Independent K Current
  float aki;
  float bki;
  float kiss;
  float I_ki;

  // I_ClCa: Ca-activated Cl Current, I_Clbk: background Cl Current
  float I_ClCa_junc;
  float I_ClCa_sl;
  float I_ClCa;
  float I_Clbk;

  // I_Ca: L-type Calcium Current
  float dss;
  float taud;
  float fss;
  float tauf;

  //
  float ibarca_j;
  float ibarca_sl;
  float ibark;
  float ibarna_j;
  float ibarna_sl;
  float I_Ca_junc;
  float I_Ca_sl;
  float I_Ca;
  float I_CaK;
  float I_CaNa_junc;
  float I_CaNa_sl;
  // float I_CaNa;
  // float I_Catot;

  // I_ncx: Na/Ca Exchanger flux
  float Ka_junc;
  float Ka_sl;
  float s1_junc;
  float s1_sl;
  float s2_junc;
  float s3_junc;
  float s2_sl;
  float s3_sl;
  float I_ncx_junc;
  float I_ncx_sl;
  float I_ncx;

  // I_pca: Sarcolemmal Ca Pump Current
  float I_pca_junc;
  float I_pca_sl;
  float I_pca;

  // I_cabk: Ca Background Current
  float I_cabk_junc;
  float I_cabk_sl;
  float I_cabk;

  // SR fluxes: Calcium Release, SR Ca pump, SR Ca leak
  float MaxSR;
  float MinSR;
  float kCaSR;
  float koSRCa;
  float kiSRCa;
  float RI;
  float J_SRCarel;																	// [mM/ms]
  float J_serca;
  float J_SRleak;																		//   [mM/ms]

  // Cytosolic Ca Buffers
  float J_CaB_cytosol;

  // Junctional and SL Ca Buffers
  float J_CaB_junction;
  float J_CaB_sl;

  // SR Ca Concentrations
  float oneovervsr;

  // Sodium Concentrations
  float I_Na_tot_junc;																// [uA/uF]
  float I_Na_tot_sl;																	// [uA/uF]
  float oneovervsl;

  // Potassium Concentration
  float I_K_tot;

  // Calcium Concentrations
  float I_Ca_tot_junc;																// [uA/uF]
  float I_Ca_tot_sl;																	// [uA/uF]
  // float junc_sl;
  // float sl_junc;
  // float sl_myo;
  // float myo_sl;

  //	Simulation type
  int state;																			// 0-none; 1-pace; 2-vclamp
  float I_app;
  float V_hold;
  float V_test;
  float V_clamp;
  float R_clamp;

  //	Membrane Potential
  float I_Na_tot;																		// [uA/uF]
  float I_Cl_tot;																		// [uA/uF]
  float I_Ca_tot;
  float I_tot;

  //=====================================================================
  //	EXECUTION
  //=====================================================================

  // input parameters
  cycleLength = d_params[15];

  // variable references
  offset_1 = valu_offset;
  offset_2 = valu_offset + 1;
  offset_3 = valu_offset + 2;
  offset_4 = valu_offset + 3;
  offset_5 = valu_offset + 4;
  offset_6 = valu_offset + 5;
  offset_7 = valu_offset + 6;
  offset_8 = valu_offset + 7;
  offset_9 = valu_offset + 8;
  offset_10 = valu_offset + 9;
  offset_11 = valu_offset + 10;
  offset_12 = valu_offset + 11;
  offset_13 = valu_offset + 12;
  offset_14 = valu_offset + 13;
  offset_15 = valu_offset + 14;
  offset_16 = valu_offset + 15;
  offset_17 = valu_offset + 16;
  offset_18 = valu_offset + 17;
  offset_19 = valu_offset + 18;
  offset_20 = valu_offset + 19;
  offset_21 = valu_offset + 20;
  offset_22 = valu_offset + 21;
  offset_23 = valu_offset + 22;
  offset_24 = valu_offset + 23;
  offset_25 = valu_offset + 24;
  offset_26 = valu_offset + 25;
  offset_27 = valu_offset + 26;
  offset_28 = valu_offset + 27;
  offset_29 = valu_offset + 28;
  offset_30 = valu_offset + 29;
  offset_31 = valu_offset + 30;
  offset_32 = valu_offset + 31;
  offset_33 = valu_offset + 32;
  offset_34 = valu_offset + 33;
  offset_35 = valu_offset + 34;
  offset_36 = valu_offset + 35;
  offset_37 = valu_offset + 36;
  offset_38 = valu_offset + 37;
  offset_39 = valu_offset + 38;
  offset_40 = valu_offset + 39;
  offset_41 = valu_offset + 40;
  offset_42 = valu_offset + 41;
  offset_43 = valu_offset + 42;
  offset_44 = valu_offset + 43;
  offset_45 = valu_offset + 44;
  offset_46 = valu_offset + 45;

  // stored input array
  d_initvalu_1 = d_initvalu[offset_1];
  d_initvalu_2 = d_initvalu[offset_2];
  d_initvalu_3 = d_initvalu[offset_3];
  d_initvalu_4 = d_initvalu[offset_4];
  d_initvalu_5 = d_initvalu[offset_5];
  d_initvalu_6 = d_initvalu[offset_6];
  d_initvalu_7 = d_initvalu[offset_7];
  d_initvalu_8 = d_initvalu[offset_8];
  d_initvalu_9 = d_initvalu[offset_9];
  d_initvalu_10 = d_initvalu[offset_10];
  d_initvalu_11 = d_initvalu[offset_11];
  d_initvalu_12 = d_initvalu[offset_12];
  d_initvalu_13 = d_initvalu[offset_13];
  d_initvalu_14 = d_initvalu[offset_14];
  d_initvalu_15 = d_initvalu[offset_15];
  d_initvalu_16 = d_initvalu[offset_16];
  d_initvalu_17 = d_initvalu[offset_17];
  d_initvalu_18 = d_initvalu[offset_18];
  d_initvalu_19 = d_initvalu[offset_19];
  d_initvalu_20 = d_initvalu[offset_20];
  d_initvalu_21 = d_initvalu[offset_21];
  // d_initvalu_22 = d_initvalu[offset_22];
  d_initvalu_23 = d_initvalu[offset_23];
  d_initvalu_24 = d_initvalu[offset_24];
  d_initvalu_25 = d_initvalu[offset_25];
  d_initvalu_26 = d_initvalu[offset_26];
  d_initvalu_27 = d_initvalu[offset_27];
  d_initvalu_28 = d_initvalu[offset_28];
  d_initvalu_29 = d_initvalu[offset_29];
  d_initvalu_30 = d_initvalu[offset_30];
  d_initvalu_31 = d_initvalu[offset_31];
  d_initvalu_32 = d_initvalu[offset_32];
  d_initvalu_33 = d_initvalu[offset_33];
  d_initvalu_34 = d_initvalu[offset_34];
  d_initvalu_35 = d_initvalu[offset_35];
  d_initvalu_36 = d_initvalu[offset_36];
  d_initvalu_37 = d_initvalu[offset_37];
  d_initvalu_38 = d_initvalu[offset_38];
  d_initvalu_39 = d_initvalu[offset_39];
  d_initvalu_40 = d_initvalu[offset_40];
  // d_initvalu_41 = d_initvalu[offset_41];
  // d_initvalu_42 = d_initvalu[offset_42];
  // d_initvalu_43 = d_initvalu[offset_43];
  // d_initvalu_44 = d_initvalu[offset_44];
  // d_initvalu_45 = d_initvalu[offset_45];
  // d_initvalu_46 = d_initvalu[offset_46];

  // matlab constants undefined in c
  pi = 3.1416;

  // Constants
  R = 8314;																			// [J/kmol*K]
  Frdy = 96485;																		// [C/mol]
  Temp = 310;																			// [K] 310
  FoRT = Frdy / R / Temp;																	//
  Cmem = 1.3810e-10;																	// [F] membrane capacitance
  Qpow = (Temp - 310) / 10;

  // Cell geometry
  cellLength = 100;																	// cell length [um]
  cellRadius = 10.25;																	// cell radius [um]
  // junctionLength = 160e-3;															// junc length [um]
  // junctionRadius = 15e-3;																// junc radius [um]
  // distSLcyto = 0.45;																	// dist. SL to cytosol [um]
  // distJuncSL = 0.5;																	// dist. junc to SL [um]
  // DcaJuncSL = 1.64e-6;																// Dca junc to SL [cm^2/sec]
  // DcaSLcyto = 1.22e-6;																// Dca SL to cyto [cm^2/sec]
  // DnaJuncSL = 1.09e-5;																// Dna junc to SL [cm^2/sec]
  // DnaSLcyto = 1.79e-5;																// Dna SL to cyto [cm^2/sec]
  Vcell = pi * pow(cellRadius, 2) * cellLength * 1e-15;											// [L]
  Vmyo = 0.65 * Vcell;
  Vsr = 0.035 * Vcell;
  Vsl = 0.02 * Vcell;
  Vjunc = 0.0539 * 0.01 * Vcell;
  // SAjunc = 20150*pi*2*junctionLength*junctionRadius;									// [um^2]
  // SAsl = pi*2*cellRadius*cellLength;													// [um^2]
  J_ca_juncsl = 1 / 1.2134e12;															// [L/msec]
  J_ca_slmyo = 1 / 2.68510e11;															// [L/msec]
  J_na_juncsl = 1 / (1.6382e12 / 3 * 100);													// [L/msec]
  J_na_slmyo = 1 / (1.8308e10 / 3 * 100);													// [L/msec]

  // Fractional currents in compartments
  Fjunc = 0.11;
  Fsl = 1 - Fjunc;
  Fjunc_CaL = 0.9;
  Fsl_CaL = 1 - Fjunc_CaL;

  // Fixed ion concentrations
  Cli = 15;																			// Intracellular Cl  [mM]
  Clo = 150;																			// Extracellular Cl  [mM]
  Ko = 5.4;																			// Extracellular K   [mM]
  Nao = 140;																			// Extracellular Na  [mM]
  Cao = 1.8;																			// Extracellular Ca  [mM]
  Mgi = 1;																			// Intracellular Mg  [mM]

  // Nernst Potentials
  ena_junc = (1 / FoRT) * log(Nao / d_initvalu_32);													// [mV]
  ena_sl = (1 / FoRT) * log(Nao / d_initvalu_33);													// [mV]
  ek = (1 / FoRT) * log(Ko / d_initvalu_35);														// [mV]
  eca_junc = (1 / FoRT / 2) * log(Cao / d_initvalu_36);												// [mV]
  eca_sl = (1 / FoRT / 2) * log(Cao / d_initvalu_37);													// [mV]
  ecl = (1 / FoRT) * log(Cli / Clo);														// [mV]

  // Na transport parameters
  GNa = 16.0;																		// [mS/uF]
  GNaB = 0.297e-3;																	// [mS/uF]
  IbarNaK = 1.90719;																	// [uA/uF]
  KmNaip = 11;																		// [mM]
  KmKo = 1.5;																			// [mM]
  // Q10NaK = 1.63;
  // Q10KmNai = 1.39;

  // K current parameters
  pNaK = 0.01833;
  GtoSlow = 0.06;																		// [mS/uF]
  GtoFast = 0.02;																		// [mS/uF]
  gkp = 0.001;

  // Cl current parameters
  GClCa = 0.109625;																	// [mS/uF]
  GClB = 9e-3;																		// [mS/uF]
  KdClCa = 100e-3;																	// [mM]

  // I_Ca parameters
  pNa = 1.5e-8;																		// [cm/sec]
  pCa = 5.4e-4;																		// [cm/sec]
  pK = 2.7e-7;																		// [cm/sec]
  // KmCa = 0.6e-3;																		// [mM]
  Q10CaL = 1.8;

  // Ca transport parameters
  IbarNCX = 9.0;																		// [uA/uF]
  KmCai = 3.59e-3;																	// [mM]
  KmCao = 1.3;																		// [mM]
  KmNai = 12.29;																		// [mM]
  KmNao = 87.5;																		// [mM]
  ksat = 0.27;																		// [none]
  nu = 0.35;																			// [none]
  Kdact = 0.256e-3;																	// [mM]
  Q10NCX = 1.57;																		// [none]
  IbarSLCaP = 0.0673;																	// [uA/uF]
  KmPCa = 0.5e-3;																		// [mM]
  GCaB = 2.513e-4;																	// [uA/uF]
  Q10SLCaP = 2.35;																	// [none]

  // SR flux parameters
  Q10SRCaP = 2.6;																		// [none]
  Vmax_SRCaP = 2.86e-4;																// [mM/msec] (mmol/L cytosol/msec)
  Kmf = 0.246e-3;																		// [mM]
  Kmr = 1.7;																			// [mM]L cytosol
  hillSRCaP = 1.787;																	// [mM]
  ks = 25;																			// [1/ms]
  koCa = 10;																			// [mM^-2 1/ms]
  kom = 0.06;																			// [1/ms]
  kiCa = 0.5;																			// [1/mM/ms]
  kim = 0.005;																		// [1/ms]
  ec50SR = 0.45;																		// [mM]

  // Buffering parameters
  Bmax_Naj = 7.561;																	// [mM]
  Bmax_Nasl = 1.65;																	// [mM]
  koff_na = 1e-3;																		// [1/ms]
  kon_na = 0.1e-3;																	// [1/mM/ms]
  Bmax_TnClow = 70e-3;																// [mM], TnC low affinity
  koff_tncl = 19.6e-3;																// [1/ms]
  kon_tncl = 32.7;																	// [1/mM/ms]
  Bmax_TnChigh = 140e-3;																// [mM], TnC high affinity
  koff_tnchca = 0.032e-3;																// [1/ms]
  kon_tnchca = 2.37;																	// [1/mM/ms]
  koff_tnchmg = 3.33e-3;																// [1/ms]
  kon_tnchmg = 3e-3;																	// [1/mM/ms]
  // Bmax_CaM = 24e-3;																	// [mM], CaM buffering
  // koff_cam = 238e-3;																	// [1/ms]
  // kon_cam = 34;																		// [1/mM/ms]
  Bmax_myosin = 140e-3;																// [mM], Myosin buffering
  koff_myoca = 0.46e-3;																// [1/ms]
  kon_myoca = 13.8;																	// [1/mM/ms]
  koff_myomg = 0.057e-3;																// [1/ms]
  kon_myomg = 0.0157;																	// [1/mM/ms]
  Bmax_SR = 19 * 0.9e-3;																	// [mM]
  koff_sr = 60e-3;																	// [1/ms]
  kon_sr = 100;																		// [1/mM/ms]
  Bmax_SLlowsl = 37.38e-3 * Vmyo / Vsl;													// [mM], SL buffering
  Bmax_SLlowj = 4.62e-3 * Vmyo / Vjunc * 0.1;												// [mM]
  koff_sll = 1300e-3;																	// [1/ms]
  kon_sll = 100;																		// [1/mM/ms]
  Bmax_SLhighsl = 13.35e-3 * Vmyo / Vsl;													// [mM]
  Bmax_SLhighj = 1.65e-3 * Vmyo / Vjunc * 0.1;												// [mM]
  koff_slh = 30e-3;																	// [1/ms]
  kon_slh = 100;																		// [1/mM/ms]
  Bmax_Csqn = 2.7;																	// 140e-3*Vmyo/Vsr; [mM]
  koff_csqn = 65;																		// [1/ms]
  kon_csqn = 100;																		// [1/mM/ms]

  // I_Na: Fast Na Current
  am = 0.32 * (d_initvalu_39 + 47.13) / (1 - exp(-0.1 * (d_initvalu_39 + 47.13)));
  bm = 0.08 * exp(-d_initvalu_39 / 11);
  if (d_initvalu_39 >= -40) {
    ah = 0;
    aj = 0;
    bh = 1 / (0.13 * (1 + exp(-(d_initvalu_39 + 10.66) / 11.1)));
    bj = 0.3 * exp(-2.535e-7 * d_initvalu_39) / (1 + exp(-0.1 * (d_initvalu_39 + 32)));
  } else {
    ah = 0.135 * exp((80 + d_initvalu_39) / -6.8);
    bh = 3.56 * exp(0.079 * d_initvalu_39) + 3.1e5 * exp(0.35 * d_initvalu_39);
    aj = (-127140 * exp(0.2444 * d_initvalu_39) - 3.474e-5 * exp(-0.04391 * d_initvalu_39))
        * (d_initvalu_39 + 37.78) / (1 + exp(0.311 * (d_initvalu_39 + 79.23)));
    bj = 0.1212 * exp(-0.01052 * d_initvalu_39) / (1 + exp(-0.1378 * (d_initvalu_39 + 40.14)));
  }
  d_finavalu[offset_1] = am * (1 - d_initvalu_1) - bm * d_initvalu_1;
  d_finavalu[offset_2] = ah * (1 - d_initvalu_2) - bh * d_initvalu_2;
  d_finavalu[offset_3] = aj * (1 - d_initvalu_3) - bj * d_initvalu_3;
  I_Na_junc = Fjunc * GNa * pow(d_initvalu_1, 3) * d_initvalu_2 * d_initvalu_3
      * (d_initvalu_39 - ena_junc);
  I_Na_sl = Fsl * GNa * pow(d_initvalu_1, 3) * d_initvalu_2 * d_initvalu_3
      * (d_initvalu_39 - ena_sl);
  // I_Na = I_Na_junc+I_Na_sl;

  // I_nabk: Na Background Current
  I_nabk_junc = Fjunc * GNaB * (d_initvalu_39 - ena_junc);
  I_nabk_sl = Fsl * GNaB * (d_initvalu_39 - ena_sl);
  // I_nabk = I_nabk_junc+I_nabk_sl;

  // I_nak: Na/K Pump Current
  sigma = (exp(Nao / 67.3) - 1) / 7;
  fnak =
      1
          / (1 + 0.1245 * exp(-0.1 * d_initvalu_39 * FoRT)
              + 0.0365 * sigma * exp(-d_initvalu_39 * FoRT));
  I_nak_junc = Fjunc * IbarNaK * fnak * Ko / (1 + pow((KmNaip / d_initvalu_32), 4)) / (Ko + KmKo);
  I_nak_sl = Fsl * IbarNaK * fnak * Ko / (1 + pow((KmNaip / d_initvalu_33), 4)) / (Ko + KmKo);
  I_nak = I_nak_junc + I_nak_sl;

  // I_kr: Rapidly Activating K Current
  gkr = 0.03 * sqrt(Ko / 5.4);
  xrss = 1 / (1 + exp(-(d_initvalu_39 + 50) / 7.5));
  tauxr = 1
      / (0.00138 * (d_initvalu_39 + 7) / (1 - exp(-0.123 * (d_initvalu_39 + 7)))
          + 6.1e-4 * (d_initvalu_39 + 10) / (exp(0.145 * (d_initvalu_39 + 10)) - 1));
  d_finavalu[offset_12] = (xrss - d_initvalu_12) / tauxr;
  rkr = 1 / (1 + exp((d_initvalu_39 + 33) / 22.4));
  I_kr = gkr * d_initvalu_12 * rkr * (d_initvalu_39 - ek);

  // I_ks: Slowly Activating K Current
  pcaks_junc = -log10(d_initvalu_36) + 3.0;
  pcaks_sl = -log10(d_initvalu_37) + 3.0;
  gks_junc = 0.07 * (0.057 + 0.19 / (1 + exp((-7.2 + pcaks_junc) / 0.6)));
  gks_sl = 0.07 * (0.057 + 0.19 / (1 + exp((-7.2 + pcaks_sl) / 0.6)));
  eks = (1 / FoRT) * log((Ko + pNaK * Nao) / (d_initvalu_35 + pNaK * d_initvalu_34));
  xsss = 1 / (1 + exp(-(d_initvalu_39 - 1.5) / 16.7));
  tauxs = 1
      / (7.19e-5 * (d_initvalu_39 + 30) / (1 - exp(-0.148 * (d_initvalu_39 + 30)))
          + 1.31e-4 * (d_initvalu_39 + 30) / (exp(0.0687 * (d_initvalu_39 + 30)) - 1));
  d_finavalu[offset_13] = (xsss - d_initvalu_13) / tauxs;
  I_ks_junc = Fjunc * gks_junc * pow(d_initvalu_12, 2) * (d_initvalu_39 - eks);
  I_ks_sl = Fsl * gks_sl * pow(d_initvalu_13, 2) * (d_initvalu_39 - eks);
  I_ks = I_ks_junc + I_ks_sl;

  // I_kp: Plateau K current
  kp_kp = 1 / (1 + exp(7.488 - d_initvalu_39 / 5.98));
  I_kp_junc = Fjunc * gkp * kp_kp * (d_initvalu_39 - ek);
  I_kp_sl = Fsl * gkp * kp_kp * (d_initvalu_39 - ek);
  I_kp = I_kp_junc + I_kp_sl;

  // I_to: Transient Outward K Current (slow and fast components)
  xtoss = 1 / (1 + exp(-(d_initvalu_39 + 3.0) / 15));
  ytoss = 1 / (1 + exp((d_initvalu_39 + 33.5) / 10));
  rtoss = 1 / (1 + exp((d_initvalu_39 + 33.5) / 10));
  tauxtos = 9 / (1 + exp((d_initvalu_39 + 3.0) / 15)) + 0.5;
  tauytos = 3e3 / (1 + exp((d_initvalu_39 + 60.0) / 10)) + 30;
  taurtos = 2800 / (1 + exp((d_initvalu_39 + 60.0) / 10)) + 220;
  d_finavalu[offset_8] = (xtoss - d_initvalu_8) / tauxtos;
  d_finavalu[offset_9] = (ytoss - d_initvalu_9) / tauytos;
  d_finavalu[offset_40] = (rtoss - d_initvalu_40) / taurtos;
  I_tos = GtoSlow * d_initvalu_8 * (d_initvalu_9 + 0.5 * d_initvalu_40) * (d_initvalu_39 - ek);	// [uA/uF]

  //
  tauxtof = 3.5 * exp(-d_initvalu_39 * d_initvalu_39 / 30 / 30) + 1.5;
  tauytof = 20.0 / (1 + exp((d_initvalu_39 + 33.5) / 10)) + 20.0;
  d_finavalu[offset_10] = (xtoss - d_initvalu_10) / tauxtof;
  d_finavalu[offset_11] = (ytoss - d_initvalu_11) / tauytof;
  I_tof = GtoFast * d_initvalu_10 * d_initvalu_11 * (d_initvalu_39 - ek);
  I_to = I_tos + I_tof;

  // I_ki: Time-Independent K Current
  aki = 1.02 / (1 + exp(0.2385 * (d_initvalu_39 - ek - 59.215)));
  bki = (0.49124 * exp(0.08032 * (d_initvalu_39 + 5.476 - ek))
      + exp(0.06175 * (d_initvalu_39 - ek - 594.31)))
      / (1 + exp(-0.5143 * (d_initvalu_39 - ek + 4.753)));
  kiss = aki / (aki + bki);
  I_ki = 0.9 * sqrt(Ko / 5.4) * kiss * (d_initvalu_39 - ek);

  // I_ClCa: Ca-activated Cl Current, I_Clbk: background Cl Current
  I_ClCa_junc = Fjunc * GClCa / (1 + KdClCa / d_initvalu_36) * (d_initvalu_39 - ecl);
  I_ClCa_sl = Fsl * GClCa / (1 + KdClCa / d_initvalu_37) * (d_initvalu_39 - ecl);
  I_ClCa = I_ClCa_junc + I_ClCa_sl;
  I_Clbk = GClB * (d_initvalu_39 - ecl);

  // I_Ca: L-type Calcium Current
  dss = 1 / (1 + exp(-(d_initvalu_39 + 14.5) / 6.0));
  taud = dss * (1 - exp(-(d_initvalu_39 + 14.5) / 6.0)) / (0.035 * (d_initvalu_39 + 14.5));
  fss = 1 / (1 + exp((d_initvalu_39 + 35.06) / 3.6)) + 0.6 / (1 + exp((50 - d_initvalu_39) / 20));
  tauf = 1 / (0.0197 * exp(-pow(0.0337 * (d_initvalu_39 + 14.5), 2)) + 0.02);
  d_finavalu[offset_4] = (dss - d_initvalu_4) / taud;
  d_finavalu[offset_5] = (fss - d_initvalu_5) / tauf;
  d_finavalu[offset_6] = 1.7 * d_initvalu_36 * (1 - d_initvalu_6) - 11.9e-3 * d_initvalu_6;	// fCa_junc
  d_finavalu[offset_7] = 1.7 * d_initvalu_37 * (1 - d_initvalu_7) - 11.9e-3 * d_initvalu_7;	// fCa_sl

  //
  ibarca_j = pCa * 4 * (d_initvalu_39 * Frdy * FoRT)
      * (0.341 * d_initvalu_36 * exp(2 * d_initvalu_39 * FoRT) - 0.341 * Cao)
      / (exp(2 * d_initvalu_39 * FoRT) - 1);
  ibarca_sl = pCa * 4 * (d_initvalu_39 * Frdy * FoRT)
      * (0.341 * d_initvalu_37 * exp(2 * d_initvalu_39 * FoRT) - 0.341 * Cao)
      / (exp(2 * d_initvalu_39 * FoRT) - 1);
  ibark = pK * (d_initvalu_39 * Frdy * FoRT)
      * (0.75 * d_initvalu_35 * exp(d_initvalu_39 * FoRT) - 0.75 * Ko)
      / (exp(d_initvalu_39 * FoRT) - 1);
  ibarna_j = pNa * (d_initvalu_39 * Frdy * FoRT)
      * (0.75 * d_initvalu_32 * exp(d_initvalu_39 * FoRT) - 0.75 * Nao)
      / (exp(d_initvalu_39 * FoRT) - 1);
  ibarna_sl = pNa * (d_initvalu_39 * Frdy * FoRT)
      * (0.75 * d_initvalu_33 * exp(d_initvalu_39 * FoRT) - 0.75 * Nao)
      / (exp(d_initvalu_39 * FoRT) - 1);
  I_Ca_junc = (Fjunc_CaL * ibarca_j * d_initvalu_4 * d_initvalu_5 * (1 - d_initvalu_6)
      * pow(Q10CaL, Qpow)) * 0.45;
  I_Ca_sl = (Fsl_CaL * ibarca_sl * d_initvalu_4 * d_initvalu_5 * (1 - d_initvalu_7)
      * pow(Q10CaL, Qpow)) * 0.45;
  I_Ca = I_Ca_junc + I_Ca_sl;
  d_finavalu[offset_43] = -I_Ca * Cmem / (Vmyo * 2 * Frdy) * 1e3;
  I_CaK = (ibark * d_initvalu_4 * d_initvalu_5
      * (Fjunc_CaL * (1 - d_initvalu_6) + Fsl_CaL * (1 - d_initvalu_7)) * pow(Q10CaL, Qpow)) * 0.45;
  I_CaNa_junc = (Fjunc_CaL * ibarna_j * d_initvalu_4 * d_initvalu_5 * (1 - d_initvalu_6)
      * pow(Q10CaL, Qpow)) * 0.45;
  I_CaNa_sl = (Fsl_CaL * ibarna_sl * d_initvalu_4 * d_initvalu_5 * (1 - d_initvalu_7)
      * pow(Q10CaL, Qpow)) * 0.45;
  // I_CaNa = I_CaNa_junc+I_CaNa_sl;
  // I_Catot = I_Ca+I_CaK+I_CaNa;

  // I_ncx: Na/Ca Exchanger flux
  Ka_junc = 1 / (1 + pow((Kdact / d_initvalu_36), 3));
  Ka_sl = 1 / (1 + pow((Kdact / d_initvalu_37), 3));
  s1_junc = exp(nu * d_initvalu_39 * FoRT) * pow(d_initvalu_32, 3) * Cao;
  s1_sl = exp(nu * d_initvalu_39 * FoRT) * pow(d_initvalu_33, 3) * Cao;
  s2_junc = exp((nu - 1) * d_initvalu_39 * FoRT) * pow(Nao, 3) * d_initvalu_36;
  s3_junc = (KmCai * pow(Nao, 3) * (1 + pow((d_initvalu_32 / KmNai), 3))
      + pow(KmNao, 3) * d_initvalu_36 + pow(KmNai, 3) * Cao * (1 + d_initvalu_36 / KmCai)
      + KmCao * pow(d_initvalu_32, 3) + pow(d_initvalu_32, 3) * Cao + pow(Nao, 3) * d_initvalu_36)
      * (1 + ksat * exp((nu - 1) * d_initvalu_39 * FoRT));
  s2_sl = exp((nu - 1) * d_initvalu_39 * FoRT) * pow(Nao, 3) * d_initvalu_37;
  s3_sl = (KmCai * pow(Nao, 3) * (1 + pow((d_initvalu_33 / KmNai), 3))
      + pow(KmNao, 3) * d_initvalu_37 + pow(KmNai, 3) * Cao * (1 + d_initvalu_37 / KmCai)
      + KmCao * pow(d_initvalu_33, 3) + pow(d_initvalu_33, 3) * Cao + pow(Nao, 3) * d_initvalu_37)
      * (1 + ksat * exp((nu - 1) * d_initvalu_39 * FoRT));
  I_ncx_junc = Fjunc * IbarNCX * pow(Q10NCX, Qpow) * Ka_junc * (s1_junc - s2_junc) / s3_junc;
  I_ncx_sl = Fsl * IbarNCX * pow(Q10NCX, Qpow) * Ka_sl * (s1_sl - s2_sl) / s3_sl;
  I_ncx = I_ncx_junc + I_ncx_sl;
  d_finavalu[offset_45] = 2 * I_ncx * Cmem / (Vmyo * 2 * Frdy) * 1e3;

  // I_pca: Sarcolemmal Ca Pump Current
  I_pca_junc = Fjunc * pow(Q10SLCaP, Qpow) * IbarSLCaP * pow(d_initvalu_36, float(1.6))
      / (pow(KmPCa, float(1.6)) + pow(d_initvalu_36, float(1.6)));
  I_pca_sl = Fsl * pow(Q10SLCaP, Qpow) * IbarSLCaP * pow(d_initvalu_37, float(1.6))
      / (pow(KmPCa, float(1.6)) + pow(d_initvalu_37, float(1.6)));
  I_pca = I_pca_junc + I_pca_sl;
  d_finavalu[offset_44] = -I_pca * Cmem / (Vmyo * 2 * Frdy) * 1e3;

  // I_cabk: Ca Background Current
  I_cabk_junc = Fjunc * GCaB * (d_initvalu_39 - eca_junc);
  I_cabk_sl = Fsl * GCaB * (d_initvalu_39 - eca_sl);
  I_cabk = I_cabk_junc + I_cabk_sl;
  d_finavalu[offset_46] = -I_cabk * Cmem / (Vmyo * 2 * Frdy) * 1e3;

  // SR fluxes: Calcium Release, SR Ca pump, SR Ca leak
  MaxSR = 15;
  MinSR = 1;
  kCaSR = MaxSR - (MaxSR - MinSR) / (1 + pow(ec50SR / d_initvalu_31, float(2.5)));
  koSRCa = koCa / kCaSR;
  kiSRCa = kiCa * kCaSR;
  RI = 1 - d_initvalu_14 - d_initvalu_15 - d_initvalu_16;
  d_finavalu[offset_14] = (kim * RI - kiSRCa * d_initvalu_36 * d_initvalu_14)
      - (koSRCa * pow(d_initvalu_36, 2) * d_initvalu_14 - kom * d_initvalu_15);			// R
  d_finavalu[offset_15] = (koSRCa * pow(d_initvalu_36, 2) * d_initvalu_14 - kom * d_initvalu_15)
      - (kiSRCa * d_initvalu_36 * d_initvalu_15 - kim * d_initvalu_16);			// O
  d_finavalu[offset_16] = (kiSRCa * d_initvalu_36 * d_initvalu_15 - kim * d_initvalu_16)
      - (kom * d_initvalu_16 - koSRCa * pow(d_initvalu_36, 2) * RI);			// I
  J_SRCarel = ks * d_initvalu_15 * (d_initvalu_31 - d_initvalu_36);												// [mM/ms]
  J_serca = pow(Q10SRCaP, Qpow) * Vmax_SRCaP
      * (pow((d_initvalu_38 / Kmf), hillSRCaP) - pow((d_initvalu_31 / Kmr), hillSRCaP))
      / (1 + pow((d_initvalu_38 / Kmf), hillSRCaP) + pow((d_initvalu_31 / Kmr), hillSRCaP));
  J_SRleak = 5.348e-6 * (d_initvalu_31 - d_initvalu_36);													//   [mM/ms]

  // Sodium and Calcium Buffering
  d_finavalu[offset_17] = kon_na * d_initvalu_32 * (Bmax_Naj - d_initvalu_17)
      - koff_na * d_initvalu_17;								// NaBj      [mM/ms]
  d_finavalu[offset_18] = kon_na * d_initvalu_33 * (Bmax_Nasl - d_initvalu_18)
      - koff_na * d_initvalu_18;							// NaBsl     [mM/ms]

      // Cytosolic Ca Buffers
  d_finavalu[offset_19] = kon_tncl * d_initvalu_38 * (Bmax_TnClow - d_initvalu_19)
      - koff_tncl * d_initvalu_19;						// TnCL      [mM/ms]
  d_finavalu[offset_20] = kon_tnchca * d_initvalu_38
      * (Bmax_TnChigh - d_initvalu_20 - d_initvalu_21) - koff_tnchca * d_initvalu_20;	// TnCHc     [mM/ms]
  d_finavalu[offset_21] = kon_tnchmg * Mgi * (Bmax_TnChigh - d_initvalu_20 - d_initvalu_21)
      - koff_tnchmg * d_initvalu_21;				// TnCHm     [mM/ms]
  d_finavalu[offset_22] = 0;																		// CaM       [mM/ms]
  d_finavalu[offset_23] = kon_myoca * d_initvalu_38 * (Bmax_myosin - d_initvalu_23 - d_initvalu_24)
      - koff_myoca * d_initvalu_23;				// Myosin_ca [mM/ms]
  d_finavalu[offset_24] = kon_myomg * Mgi * (Bmax_myosin - d_initvalu_23 - d_initvalu_24)
      - koff_myomg * d_initvalu_24;				// Myosin_mg [mM/ms]
  d_finavalu[offset_25] = kon_sr * d_initvalu_38 * (Bmax_SR - d_initvalu_25)
      - koff_sr * d_initvalu_25;								// SRB       [mM/ms]
  J_CaB_cytosol = d_finavalu[offset_19] + d_finavalu[offset_20] + d_finavalu[offset_21]
      + d_finavalu[offset_22] + d_finavalu[offset_23] + d_finavalu[offset_24]
      + d_finavalu[offset_25];

  // Junctional and SL Ca Buffers
  d_finavalu[offset_26] = kon_sll * d_initvalu_36 * (Bmax_SLlowj - d_initvalu_26)
      - koff_sll * d_initvalu_26;						// SLLj      [mM/ms]
  d_finavalu[offset_27] = kon_sll * d_initvalu_37 * (Bmax_SLlowsl - d_initvalu_27)
      - koff_sll * d_initvalu_27;						// SLLsl     [mM/ms]
  d_finavalu[offset_28] = kon_slh * d_initvalu_36 * (Bmax_SLhighj - d_initvalu_28)
      - koff_slh * d_initvalu_28;						// SLHj      [mM/ms]
  d_finavalu[offset_29] = kon_slh * d_initvalu_37 * (Bmax_SLhighsl - d_initvalu_29)
      - koff_slh * d_initvalu_29;						// SLHsl     [mM/ms]
  J_CaB_junction = d_finavalu[offset_26] + d_finavalu[offset_28];
  J_CaB_sl = d_finavalu[offset_27] + d_finavalu[offset_29];

  // SR Ca Concentrations
  d_finavalu[offset_30] = kon_csqn * d_initvalu_31 * (Bmax_Csqn - d_initvalu_30)
      - koff_csqn * d_initvalu_30;						// Csqn      [mM/ms]
  oneovervsr = 1 / Vsr;
  d_finavalu[offset_31] = J_serca * Vmyo * oneovervsr - (J_SRleak * Vmyo * oneovervsr + J_SRCarel)
      - d_finavalu[offset_30];   // Ca_sr     [mM/ms] %Ratio 3 leak current

  // Sodium Concentrations
  I_Na_tot_junc = I_Na_junc + I_nabk_junc + 3 * I_ncx_junc + 3 * I_nak_junc + I_CaNa_junc;// [uA/uF]
  I_Na_tot_sl = I_Na_sl + I_nabk_sl + 3 * I_ncx_sl + 3 * I_nak_sl + I_CaNa_sl;					// [uA/uF]
  d_finavalu[offset_32] = -I_Na_tot_junc * Cmem / (Vjunc * Frdy)
      + J_na_juncsl / Vjunc * (d_initvalu_33 - d_initvalu_32) - d_finavalu[offset_17];
  oneovervsl = 1 / Vsl;
  d_finavalu[offset_33] = -I_Na_tot_sl * Cmem * oneovervsl / Frdy
      + J_na_juncsl * oneovervsl * (d_initvalu_32 - d_initvalu_33)
      + J_na_slmyo * oneovervsl * (d_initvalu_34 - d_initvalu_33) - d_finavalu[offset_18];
  d_finavalu[offset_34] = J_na_slmyo / Vmyo * (d_initvalu_33 - d_initvalu_34);					// [mM/msec]

  // Potassium Concentration
  I_K_tot = I_to + I_kr + I_ks + I_ki - 2 * I_nak + I_CaK + I_kp;									// [uA/uF]
  d_finavalu[offset_35] = 0;															// [mM/msec]

  // Calcium Concentrations
  I_Ca_tot_junc = I_Ca_junc + I_cabk_junc + I_pca_junc - 2 * I_ncx_junc;						// [uA/uF]
  I_Ca_tot_sl = I_Ca_sl + I_cabk_sl + I_pca_sl - 2 * I_ncx_sl;								// [uA/uF]
  d_finavalu[offset_36] = -I_Ca_tot_junc * Cmem / (Vjunc * 2 * Frdy)
      + J_ca_juncsl / Vjunc * (d_initvalu_37 - d_initvalu_36) - J_CaB_junction
      + (J_SRCarel) * Vsr / Vjunc + J_SRleak * Vmyo / Vjunc;				// Ca_j
  d_finavalu[offset_37] = -I_Ca_tot_sl * Cmem / (Vsl * 2 * Frdy)
      + J_ca_juncsl / Vsl * (d_initvalu_36 - d_initvalu_37)
      + J_ca_slmyo / Vsl * (d_initvalu_38 - d_initvalu_37) - J_CaB_sl;									// Ca_sl
  d_finavalu[offset_38] = -J_serca - J_CaB_cytosol
      + J_ca_slmyo / Vmyo * (d_initvalu_37 - d_initvalu_38);
  // junc_sl=J_ca_juncsl/Vsl*(d_initvalu_36-d_initvalu_37);
  // sl_junc=J_ca_juncsl/Vjunc*(d_initvalu_37-d_initvalu_36);
  // sl_myo=J_ca_slmyo/Vsl*(d_initvalu_38-d_initvalu_37);
  // myo_sl=J_ca_slmyo/Vmyo*(d_initvalu_37-d_initvalu_38);

  // Simulation type
  state = 1;
  switch (state) {
    case 0:
      I_app = 0;
      break;
    case 1:																// pace w/ current injection at cycleLength 'cycleLength'
      if (fmod(timeinst, cycleLength) <= 5) {
        I_app = 9.5;
      } else {
        I_app = 0.0;
      }
      break;
    case 2:
      V_hold = -55;
      V_test = 0;
      if (timeinst > 0.5 & timeinst < 200.5) {
        V_clamp = V_test;
      } else {
        V_clamp = V_hold;
      }
      R_clamp = 0.04;
      I_app = (V_clamp - d_initvalu_39) / R_clamp;
      break;
  }

  // Membrane Potential
  I_Na_tot = I_Na_tot_junc + I_Na_tot_sl;												// [uA/uF]
  I_Cl_tot = I_ClCa + I_Clbk;															// [uA/uF]
  I_Ca_tot = I_Ca_tot_junc + I_Ca_tot_sl;
  I_tot = I_Na_tot + I_Cl_tot + I_Ca_tot + I_K_tot;
  d_finavalu[offset_39] = -(I_tot - I_app);

  // Set unused output values to 0 (MATLAB does it by default)
  d_finavalu[offset_41] = 0;
  d_finavalu[offset_42] = 0;

}
