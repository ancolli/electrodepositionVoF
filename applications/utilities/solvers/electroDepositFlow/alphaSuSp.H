
#include "physicoChemicalConstants.H"

const dimensionedScalar F = Foam::constant::physicoChemical::F;
/*const volScalarField& alphaSource = phaseChange.alphaSource();

forAll(dgdt, celli)
{
    if (dgdt[celli] > 0.0)
    {
        Sp[celli] -= dgdt[celli]/max(1.0 - alpha1[celli], 1e-4);
        Su[celli] += dgdt[celli]/max(1.0 - alpha1[celli], 1e-4);
    }
    else if (dgdt[celli] < 0.0)
    {
        Sp[celli] += dgdt[celli]/max(alpha1[celli], 1e-4);
    }
    if (alphaSource[celli] > 0.0 && alpha1[celli] > 0.0)
    {
        //Sp[celli] -= alphaSource[celli];
        Su[celli] -= alphaSource[celli];
    }
    else if (alphaSource[celli] < 0.0 && alpha1[celli] < 1.0)
    {
        Su[celli] -= alphaSource[celli];
    }

}*/

reconstructionSchemes& surf =
        mesh.lookupObjectRef<reconstructionSchemes>("reconstructionScheme");
surf.reconstruct(false);

    //const volVectorField& Centre = surf.centre();
const volVectorField& normal = surf.normal();
    
const volVectorField& interCenter = surf.centre();

//const volScalarField& curvatura = surf.curv();

const volScalarField curv = mesh.lookupObjectRef<volScalarField>("K_");

dimensionedScalar area("area",dimArea,gSum(mag(normal)().internalField()));

//Info << "Área total " << area << endl;
//Info << "interCenter " << mag(interCenter) << endl;


const scalar j0Val = j0.value();
const scalar rho1Val = rho1.value();
const scalar FVal = F.value();
const scalar neVal = ne.value();
const scalar MWVal = MW.value();
const scalar C0Val = C0.value();
const scalar baVal = ba.value();
const scalar bcVal = bc.value();


forAll(alpha1, celli)
{
    scalar alphaVal = alpha1[celli];
    scalar VolCell = alpha1.mesh().V()[celli];
    scalar normalMag = mag(normal[celli]);
    scalar fi_sVal = fi_s[celli];
    scalar fi_fVal = fi_f[celli];
    scalar CVal = C[celli];
    scalar Ae = normalMag / VolCell;

    // Caso donde se fuerza a cero:
    if (mag(interCenter[celli]) < 1e-12 || alphaVal >= 1.0)//(1.0-1e-4)
    {
        Su[celli] = Sufi[celli] = djdfi[celli] = 0.0;
        jmenosfidjdfi_f[celli] = jmenosfidjdfi_s[celli] = 0.0;
        conc[celli] = preconc[celli] = 0.0;
        continue; // Salta esta celda
    }

    // Solo calcula si es interfaz activa (0 < alpha < 1)
    scalar deltaFi = fi_sVal - fi_fVal;
    scalar expPosBa = Foam::exp(deltaFi / baVal);
    scalar expNegBc = Foam::exp(deltaFi / bcVal);

    scalar jterm = j0Val * (expPosBa - CVal * expNegBc);
    

    Su[celli] = -jterm / rho1Val * Ae / FVal / neVal * MWVal;
    Sufi[celli] = j0Val * Ae;

    djdfi[celli] = expPosBa / baVal - CVal * expNegBc / bcVal;

    scalar jmenosBase = expPosBa - CVal * expNegBc;
    jmenosfidjdfi_f[celli] = jmenosBase + fi_fVal * djdfi[celli];
    jmenosfidjdfi_s[celli] = jmenosBase - fi_sVal * djdfi[celli];

    scalar concBase = j0Val * Ae / FVal / neVal;
    conc[celli]    = expPosBa * concBase / C0Val;
    preconc[celli] = -expNegBc * concBase / C0Val;
}

/*forAll(alpha1, celli)
{
    if (mag(interCenter[celli]) < 1e-12)//(alpha1[celli] < 1e-3 || alpha1[celli] > 1-1e-3) // Usar || no && para condiciones excluyentes
    //if (alpha1[celli] < 0.01 ) 
    {
        //Sp[celli] = 0.0;
        Su[celli] = 0.0;
        Sufi[celli] = 0.0; 
        djdfi[celli] = 0.0;
        jmenosfidjdfi_f[celli] = 0.0;
        jmenosfidjdfi_s[celli] = 0.0;
        conc[celli] = 0.0;
        preconc[celli] = 0.0;
    }
    else
    {
    	if (alpha1[celli] < 1.0)//
    	{
		//Sp[celli] = -0.5*(Foam::exp((fi_s[celli]-fi_f[celli])/.02525)-Foam::exp(-(fi_s[celli]-fi_f[celli])/.02525))*mdot.value()/rho1.value()*mag(normal[celli])/alpha1.mesh().V()[celli]/96485/2*0.0635/max(alpha1[celli], 1e-4);//
		Su[celli] = -j0.value()*(Foam::exp((fi_s[celli]-fi_f[celli])/ba.value())-C[celli]*Foam::exp((fi_s[celli]-fi_f[celli])/bc.value()))/rho1.value()*mag(normal[celli])/alpha1.mesh().V()[celli]/F.value()/ne.value()*MW.value(); //estaba con menos /max(1e-5,alpha1[celli])
		Sufi[celli] = j0.value()*mag(normal[celli])/alpha1.mesh().V()[celli];//-2*96485/0.063*Su[celli]*rho1.value();//
		
		djdfi[celli] = (Foam::exp((fi_s[celli]-fi_f[celli])/ba.value())/ba.value() - C[celli]*Foam::exp((fi_s[celli]-fi_f[celli])/bc.value())/bc.value());
		
		jmenosfidjdfi_f[celli] = (Foam::exp((fi_s[celli]-fi_f[celli])/ba.value()) - C[celli]*Foam::exp((fi_s[celli]-fi_f[celli])/bc.value()))+fi_f[celli]*djdfi[celli];
		jmenosfidjdfi_s[celli] = (Foam::exp((fi_s[celli]-fi_f[celli])/ba.value()) - C[celli]*Foam::exp((fi_s[celli]-fi_f[celli])/bc.value()))-fi_s[celli]*djdfi[celli];
		
		conc[celli] = (Foam::exp((fi_s[celli]-fi_f[celli])/ba.value()))*j0.value()*mag(normal[celli])/alpha1.mesh().V()[celli]/F.value()/ne.value()/C0.value(); // 
		preconc[celli] = -Foam::exp((fi_s[celli]-fi_f[celli])/bc.value())*j0.value()*mag(normal[celli])/alpha1.mesh().V()[celli]/F.value()/ne.value()/C0.value(); // 
		//conc[celli] = (Foam::exp((fi_s[celli]-fi_f[celli])/.02525)-C[celli]*Foam::exp(-(fi_s[celli]-fi_f[celli])/.02525))*mdot.value()*mag(normal[celli])/alpha1.mesh().V()[celli]/96485/2;// - curv[celli]*1;
	}
	else
	{
		//Sp[celli] = 0.0;
		Su[celli] = 0.0;
		Sufi[celli] = 0.0; 
		djdfi[celli] = 0.0;
		jmenosfidjdfi_f[celli] = 0.0;
		jmenosfidjdfi_s[celli] = 0.0;
		conc[celli] = 0.0;
		preconc[celli] = 0.0;
        }
    }

}*/


//Info << "max Su " << max(Su).value() << ", min Su " << min(Su).value()<< endl;
//Info << "max Sufi " << max(Sufi).value() << ", min Sufi " << min(Sufi).value()<< endl;

volScalarField::Internal divU
(
    mesh.moving()
  ? fvc::div(phi + mesh.phi())
  : fvc::div(phi)
);
