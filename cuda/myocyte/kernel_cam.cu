//=====================================================================
//	MAIN FUNCTION
//=====================================================================
__device__ void kernel_cam(	float timeinst,
												float* d_initvalu,
												float *d_finavalu,
												int valu_offset,
												float* d_params,
												int params_offset,
												float* d_com,
												int com_offset,
												float Ca){

	//=====================================================================
	//	VARIABLES
	//=====================================================================

	// inputs
	// float CaMtot;
	float Btot;
	float CaMKIItot;
	float CaNtot;
	float PP1tot;
	float K;
	float Mg;

	// variable references
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

	// decoding input array
	float CaM;
	float Ca2CaM;
	float Ca4CaM;
	float CaMB;
	float Ca2CaMB;
	float Ca4CaMB;
	float Pb2;
	float Pb;
	float Pt;
	float Pt2;
	float Pa;
	float Ca4CaN;
	float CaMCa4CaN;
	float Ca2CaMCa4CaN;
	float Ca4CaMCa4CaN;

	// Ca/CaM parameters
	float Kd02;																		// [uM^2]
	float Kd24;																		// [uM^2]
	float k20;																			// [s^-1]
	float k02;																			// [uM^-2 s^-1]
	float k42;																			// [s^-1]
	float k24;																			// [uM^-2 s^-1]

	// CaM buffering (B) parameters
	float k0Boff;																		// [s^-1]
	float k0Bon;																		// [uM^-1 s^-1] kon = koff/Kd
	float k2Boff;																		// [s^-1]
	float k2Bon;																		// [uM^-1 s^-1]
	float k4Boff;																		// [s^-1]
	float k4Bon;																		// [uM^-1 s^-1]

	// using thermodynamic constraints
	float k20B;																		// [s^-1] thermo constraint on loop 1
	float k02B;																		// [uM^-2 s^-1]
	float k42B;																		// [s^-1] thermo constraint on loop 2
	float k24B;																		// [uM^-2 s^-1]

	// Wi Wa Wt Wp
	float kbi;																			// [s^-1] (Ca4CaM dissocation from Wb)
	float kib;																			// [uM^-1 s^-1]
	float kpp1;																		// [s^-1] (PP1-dep dephosphorylation rates)
	float Kmpp1;																		// [uM]
	float kib2;
	float kb2i;
	float kb24;
	float kb42;
	float kta;																			// [s^-1] (Ca4CaM dissociation from Wt)
	float kat;																			// [uM^-1 s^-1] (Ca4CaM reassociation with Wa)
	float kt42;
	float kt24;
	float kat2;
	float kt2a;

	// CaN parameters
	float kcanCaoff;																	// [s^-1]
	float kcanCaon;																	// [uM^-1 s^-1]
	float kcanCaM4on;																	// [uM^-1 s^-1]
	float kcanCaM4off;																	// [s^-1]
	float kcanCaM2on;
	float kcanCaM2off;
	float kcanCaM0on;
	float kcanCaM0off;
	float k02can;
	float k20can;
	float k24can;
	float k42can;

	// CaM Reaction fluxes
	float rcn02;
	float rcn24;

	// CaM buffer fluxes
	float B;
	float rcn02B;
	float rcn24B;
	float rcn0B;
	float rcn2B;
	float rcn4B;

	// CaN reaction fluxes
	float Ca2CaN;
	float rcnCa4CaN;
	float rcn02CaN;
	float rcn24CaN;
	float rcn0CaN;
	float rcn2CaN;
	float rcn4CaN;

	// CaMKII reaction fluxes
	float Pix;
	float rcnCKib2;
	float rcnCKb2b;
	float rcnCKib;
	float T;
	float kbt;
	float rcnCKbt;
	float rcnCKtt2;
	float rcnCKta;
	float rcnCKt2a;
	float rcnCKt2b2;
	float rcnCKai;

	// CaM equations
	float dCaM;
	float dCa2CaM;
	float dCa4CaM;
	float dCaMB;
	float dCa2CaMB;
	float dCa4CaMB;

	// CaMKII equations
	float dPb2;																					// Pb2
	float dPb;																					// Pb
	float dPt;																					// Pt
	float dPt2;																					// Pt2
	float dPa;																					// Pa

	// CaN equations
	float dCa4CaN;																			// Ca4CaN
	float dCaMCa4CaN;																	// CaMCa4CaN
	float dCa2CaMCa4CaN;																// Ca2CaMCa4CaN
	float dCa4CaMCa4CaN;																// Ca4CaMCa4CaN

	//=====================================================================
	//	EXECUTION
	//=====================================================================

	// inputs
	// CaMtot = d_params[params_offset];
	Btot = d_params[params_offset+1];
	CaMKIItot = d_params[params_offset+2];
	CaNtot = d_params[params_offset+3];
	PP1tot = d_params[params_offset+4];
	K = d_params[16];
	Mg = d_params[17];

	// variable references
	offset_1 = valu_offset;
	offset_2 = valu_offset+1;
	offset_3 = valu_offset+2;
	offset_4 = valu_offset+3;
	offset_5 = valu_offset+4;
	offset_6 = valu_offset+5;
	offset_7 = valu_offset+6;
	offset_8 = valu_offset+7;
	offset_9 = valu_offset+8;
	offset_10 = valu_offset+9;
	offset_11 = valu_offset+10;
	offset_12 = valu_offset+11;
	offset_13 = valu_offset+12;
	offset_14 = valu_offset+13;
	offset_15 = valu_offset+14;

	// decoding input array
	CaM				= d_initvalu[offset_1];
	Ca2CaM			= d_initvalu[offset_2];
	Ca4CaM			= d_initvalu[offset_3];
	CaMB			= d_initvalu[offset_4];
	Ca2CaMB			= d_initvalu[offset_5];
	Ca4CaMB			= d_initvalu[offset_6];
	Pb2				= d_initvalu[offset_7];
	Pb				= d_initvalu[offset_8];
	Pt				= d_initvalu[offset_9];
	Pt2				= d_initvalu[offset_10];
	Pa				= d_initvalu[offset_11];
	Ca4CaN			= d_initvalu[offset_12];
	CaMCa4CaN		= d_initvalu[offset_13];
	Ca2CaMCa4CaN	= d_initvalu[offset_14];
	Ca4CaMCa4CaN	= d_initvalu[offset_15];

	// Ca/CaM parameters
	if (Mg <= 1){
		Kd02 = 0.0025*(1+K/0.94-Mg/0.012)*(1+K/8.1+Mg/0.022);							// [uM^2]
		Kd24 = 0.128*(1+K/0.64+Mg/0.0014)*(1+K/13.0-Mg/0.153);							// [uM^2]
	}
	else{
		Kd02 = 0.0025*(1+K/0.94-1/0.012+(Mg-1)/0.060)*(1+K/8.1+1/0.022+(Mg-1)/0.068);   // [uM^2]
		Kd24 = 0.128*(1+K/0.64+1/0.0014+(Mg-1)/0.005)*(1+K/13.0-1/0.153+(Mg-1)/0.150);  // [uM^2]
	}
	k20 = 10;																			// [s^-1]
	k02 = k20/Kd02;																		// [uM^-2 s^-1]
	k42 = 500;																			// [s^-1]
	k24 = k42/Kd24;																		// [uM^-2 s^-1]

	// CaM buffering (B) parameters
	k0Boff = 0.0014;																	// [s^-1]
	k0Bon = k0Boff/0.2;																	// [uM^-1 s^-1] kon = koff/Kd
	k2Boff = k0Boff/100;																// [s^-1]
	k2Bon = k0Bon;																		// [uM^-1 s^-1]
	k4Boff = k2Boff;																	// [s^-1]
	k4Bon = k0Bon;																		// [uM^-1 s^-1]

	// using thermodynamic constraints
	k20B = k20/100;																		// [s^-1] thermo constraint on loop 1
	k02B = k02;																			// [uM^-2 s^-1]
	k42B = k42;																			// [s^-1] thermo constraint on loop 2
	k24B = k24;																			// [uM^-2 s^-1]

	// Wi Wa Wt Wp
	kbi = 2.2;																			// [s^-1] (Ca4CaM dissocation from Wb)
	kib = kbi/33.5e-3;																	// [uM^-1 s^-1]
	kpp1 = 1.72;																		// [s^-1] (PP1-dep dephosphorylation rates)
	Kmpp1 = 11.5;																		// [uM]
	kib2 = kib;
	kb2i = kib2*5;
	kb24 = k24;
	kb42 = k42*33.5e-3/5;
	kta = kbi/1000;																		// [s^-1] (Ca4CaM dissociation from Wt)
	kat = kib;																			// [uM^-1 s^-1] (Ca4CaM reassociation with Wa)
	kt42 = k42*33.5e-6/5;
	kt24 = k24;
	kat2 = kib;
	kt2a = kib*5;

	// CaN parameters
	kcanCaoff = 1;																		// [s^-1]
	kcanCaon = kcanCaoff/0.5;															// [uM^-1 s^-1]
	kcanCaM4on = 46;																	// [uM^-1 s^-1]
	kcanCaM4off = 0.0013;																// [s^-1]
	kcanCaM2on = kcanCaM4on;
	kcanCaM2off = 2508*kcanCaM4off;
	kcanCaM0on = kcanCaM4on;
	kcanCaM0off = 165*kcanCaM2off;
	k02can = k02;
	k20can = k20/165;
	k24can = k24;
	k42can = k20/2508;

	// CaM Reaction fluxes
	rcn02 = k02*pow(Ca,2)*CaM - k20*Ca2CaM;
	rcn24 = k24*pow(Ca,2)*Ca2CaM - k42*Ca4CaM;

	// CaM buffer fluxes
	B = Btot - CaMB - Ca2CaMB - Ca4CaMB;
	rcn02B = k02B*pow(Ca,2)*CaMB - k20B*Ca2CaMB;
	rcn24B = k24B*pow(Ca,2)*Ca2CaMB - k42B*Ca4CaMB;
	rcn0B = k0Bon*CaM*B - k0Boff*CaMB;
	rcn2B = k2Bon*Ca2CaM*B - k2Boff*Ca2CaMB;
	rcn4B = k4Bon*Ca4CaM*B - k4Boff*Ca4CaMB;

	// CaN reaction fluxes
	Ca2CaN = CaNtot - Ca4CaN - CaMCa4CaN - Ca2CaMCa4CaN - Ca4CaMCa4CaN;
	rcnCa4CaN = kcanCaon*pow(Ca,2)*Ca2CaN - kcanCaoff*Ca4CaN;
	rcn02CaN = k02can*pow(Ca,2)*CaMCa4CaN - k20can*Ca2CaMCa4CaN;
	rcn24CaN = k24can*pow(Ca,2)*Ca2CaMCa4CaN - k42can*Ca4CaMCa4CaN;
	rcn0CaN = kcanCaM0on*CaM*Ca4CaN - kcanCaM0off*CaMCa4CaN;
	rcn2CaN = kcanCaM2on*Ca2CaM*Ca4CaN - kcanCaM2off*Ca2CaMCa4CaN;
	rcn4CaN = kcanCaM4on*Ca4CaM*Ca4CaN - kcanCaM4off*Ca4CaMCa4CaN;

	// CaMKII reaction fluxes
	Pix = 1 - Pb2 - Pb - Pt - Pt2 - Pa;
	rcnCKib2 = kib2*Ca2CaM*Pix - kb2i*Pb2;
	rcnCKb2b = kb24*pow(Ca,2)*Pb2 - kb42*Pb;
	rcnCKib = kib*Ca4CaM*Pix - kbi*Pb;
	T = Pb + Pt + Pt2 + Pa;
	kbt = 0.055*T + 0.0074*pow(T,2) + 0.015*pow(T,3);
	rcnCKbt = kbt*Pb - kpp1*PP1tot*Pt/(Kmpp1+CaMKIItot*Pt);
	rcnCKtt2 = kt42*Pt - kt24*pow(Ca,2)*Pt2;
	rcnCKta = kta*Pt - kat*Ca4CaM*Pa;
	rcnCKt2a = kt2a*Pt2 - kat2*Ca2CaM*Pa;
	rcnCKt2b2 = kpp1*PP1tot*Pt2/(Kmpp1+CaMKIItot*Pt2);
	rcnCKai = kpp1*PP1tot*Pa/(Kmpp1+CaMKIItot*Pa);

	// CaM equations
	dCaM = 1e-3*(-rcn02 - rcn0B - rcn0CaN);
	dCa2CaM = 1e-3*(rcn02 - rcn24 - rcn2B - rcn2CaN + CaMKIItot*(-rcnCKib2 + rcnCKt2a) );
	dCa4CaM = 1e-3*(rcn24 - rcn4B - rcn4CaN + CaMKIItot*(-rcnCKib+rcnCKta) );
	dCaMB = 1e-3*(rcn0B-rcn02B);
	dCa2CaMB = 1e-3*(rcn02B + rcn2B - rcn24B);
	dCa4CaMB = 1e-3*(rcn24B + rcn4B);

	// CaMKII equations
	dPb2 = 1e-3*(rcnCKib2 - rcnCKb2b + rcnCKt2b2);										// Pb2
	dPb = 1e-3*(rcnCKib + rcnCKb2b - rcnCKbt);											// Pb
	dPt = 1e-3*(rcnCKbt-rcnCKta-rcnCKtt2);												// Pt
	dPt2 = 1e-3*(rcnCKtt2-rcnCKt2a-rcnCKt2b2);											// Pt2
	dPa = 1e-3*(rcnCKta+rcnCKt2a-rcnCKai);												// Pa

	// CaN equations
	dCa4CaN = 1e-3*(rcnCa4CaN - rcn0CaN - rcn2CaN - rcn4CaN);							// Ca4CaN
	dCaMCa4CaN = 1e-3*(rcn0CaN - rcn02CaN);												// CaMCa4CaN
	dCa2CaMCa4CaN = 1e-3*(rcn2CaN+rcn02CaN-rcn24CaN);									// Ca2CaMCa4CaN
	dCa4CaMCa4CaN = 1e-3*(rcn4CaN+rcn24CaN);											// Ca4CaMCa4CaN

	// encode output array
	d_finavalu[offset_1] = dCaM;
	d_finavalu[offset_2] = dCa2CaM;
	d_finavalu[offset_3] = dCa4CaM;
	d_finavalu[offset_4] = dCaMB;
	d_finavalu[offset_5] = dCa2CaMB;
	d_finavalu[offset_6] = dCa4CaMB;
	d_finavalu[offset_7] = dPb2;
	d_finavalu[offset_8] = dPb;
	d_finavalu[offset_9] = dPt;
	d_finavalu[offset_10] = dPt2;
	d_finavalu[offset_11] = dPa;
	d_finavalu[offset_12] = dCa4CaN;
	d_finavalu[offset_13] = dCaMCa4CaN;
	d_finavalu[offset_14] = dCa2CaMCa4CaN;
	d_finavalu[offset_15] = dCa4CaMCa4CaN;

	// write to global variables for adjusting Ca buffering in EC coupling model
	d_finavalu[com_offset] = 1e-3*(2*CaMKIItot*(rcnCKtt2-rcnCKb2b) - 2*(rcn02+rcn24+rcn02B+rcn24B+rcnCa4CaN+rcn02CaN+rcn24CaN)); // [uM/msec]
	//d_finavalu[JCa] = 1; // [uM/msec]

}
