#include "createRDeltaT.H"

Info<< "Reading field T\n" << endl;

volScalarField C
(
    IOobject
    (
        "C",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);


Info<< "Reading field p_rgh\n" << endl;
volScalarField p_rgh
(
    IOobject
    (
        "p_rgh",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

Info<< "Reading field U\n" << endl;
volVectorField U
(
    IOobject
    (
        "U",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

#include "createPhi.H"


Info<< "Reading transportProperties\n" << endl;

IOdictionary transportProperties
(
    IOobject
    (
        "transportProperties",
        U.time().constant(),
        U.db(),
        IOobject::MUST_READ_IF_MODIFIED,
        IOobject::NO_WRITE
    )
);


immiscibleIncompressibleTwoPhaseMixture mixture(U, phi);

volScalarField& alpha1(mixture.alpha1());
volScalarField& alpha2(mixture.alpha2());

const dimensionedScalar& rho1 = mixture.rho1();
const dimensionedScalar& rho2 = mixture.rho2();

dimensionedScalar beta_c("beta_c", dimless, transportProperties);

// Need to store rho for ddt(rho, U)
volScalarField rho
(
    IOobject
    (
        "rho",
        runTime.timeName(),
        mesh,
        IOobject::READ_IF_PRESENT
    ),
    alpha1*rho1 + alpha2*rho2*(1-beta_c*(C-1.0))
);
rho.oldTime();

// Mass flux
surfaceScalarField rhoPhi
(
    IOobject
    (
        "rhoPhi",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    fvc::interpolate(rho)*phi
);

// Construct incompressible turbulence model
autoPtr<incompressible::turbulenceModel> turbulence
(
    incompressible::turbulenceModel::New(U, phi, mixture)
);


surfaceForces surfForces(alpha1,phi,U,transportProperties);

//#include "readGravitationalAcceleration.H"
//#include "readhRef.H"
//#include "gh.H"
const volScalarField& gh = surfForces.acc();


/*-------------------------------------------*/
/*           Mass Transfer Directory         */
/*-------------------------------------------*/

Info<< "Reading j0\n" << endl;

dimensionedScalar j0("j0", dimDensity/dimTime, transportProperties);

Switch species("species", transportProperties);

Switch fluid("fluid", transportProperties);

/*-------------------------------------------*/
/*           Mass Transfer Directory         */
/*-------------------------------------------*/

//From here

Info<< "Reading field fi\n" << endl;

volScalarField fi_f
(
    IOobject
    (
	    "fi_f",
	    runTime.timeName(),
	    mesh,
	    IOobject::MUST_READ,
	    IOobject::AUTO_WRITE
    ),
    mesh
);

volScalarField fi_s
(
    IOobject
    (
	    "fi_s",
	    runTime.timeName(),
	    mesh,
	    IOobject::MUST_READ,
	    IOobject::AUTO_WRITE
    ),
    mesh
);

volScalarField Flux
(
	IOobject
	(
	    "Flux",
	    runTime.timeName(),
	    mesh,
	    IOobject::NO_READ,
	    IOobject::NO_WRITE//AUTO_WRITE//
	),
	mesh,
	dimensionedScalar("0", dimensionSet(0,-2,0,0,0,1,0), 0.0)
);

// * * * * * * * * * * * * * Reading Properties * * * * * * * * * * * * * * * * //

//#include "createPhi.H"
//#include "readTransportProperties.H"

Info<< "Reading diffusivity Sct\n" << endl;

dimensionedScalar Sct("Sct", dimless, transportProperties);

Info<< "Reading diffusivity D\n" << endl;

dimensionedScalar D("D", dimViscosity, transportProperties);

Info<< "Reading diffusivity kf\n" << endl;

dimensionedScalar kf_f("kf_f", dimensionSet (-1,-3,3,0,0,2,0), transportProperties);
dimensionedScalar kf_s("kf_s", dimensionSet (-1,-3,3,0,0,2,0), transportProperties);

Info<< "Reading C0\n" << endl;

dimensionedScalar C0("C0", dimMoles/dimVolume, transportProperties);

Info<< "Reading bc and ba\n" << endl;

dimensionedScalar bc("bc", dimless, transportProperties);

dimensionedScalar ba("ba", dimless, transportProperties);

dimensionedScalar MW("MW", dimMass/dimMoles, transportProperties);

dimensionedScalar ne("ne", dimless, transportProperties);



//dimensionedVector Upared("Upared", dimVelocity, transportProperties);

/*volScalarField keff
(
	IOobject
	(
	    "keff",
	    runTime.timeName(),
	    mesh,
	    IOobject::MUST_READ,
	    IOobject::AUTO_WRITE
	),
	//mesh,
	alpha1*kf_s + alpha2*kf_f //dimensionedScalar("keff", kf)
);*/

Info<< "Reading controlProperties\n" << endl;

IOdictionary controlProperties
(
    IOobject
    (
        "controlProperties",
        runTime.constant(),
        mesh,
        IOobject::MUST_READ_IF_MODIFIED,
        IOobject::NO_WRITE
    )
);

/*IOdictionary controlDict
(
    IOobject
    (
        "controlDict",
        runTime.system(),
        mesh,
        IOobject::MUST_READ_IF_MODIFIED,
        IOobject::AUTO_WRITE
    )
);*/

IOdictionary Results
(
    IOobject
    (
        "Results",
        runTime.constant(),
        mesh,
        IOobject::MUST_READ_IF_MODIFIED,
        IOobject::AUTO_WRITE
    )
);

//Until here


volScalarField p
(
    IOobject
    (
        "p",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::AUTO_WRITE
    ),
    p_rgh + rho*gh
);

label pRefCell = 0;
scalar pRefValue = 0.0;
setRefCell
(
    p,
    p_rgh,
    pimple.dict(),
    pRefCell,
    pRefValue
);

if (p_rgh.needReference())
{
    p += dimensionedScalar
    (
        "p",
        p.dimensions(),
        pRefValue - getRefCellValue(p, pRefCell)
    );
    p_rgh = p - rho*gh;
}

mesh.setFluxRequired(p_rgh.name());
mesh.setFluxRequired(alpha1.name());

#include "createMRF.H"
#include "createFvOptions.H"


autoPtr<advectionSchemes> advector
(
    advectionSchemes::New(alpha1,phi,U)
);


volScalarField::Internal Sp
(
    IOobject
    (
        "Sp",
        runTime.timeName(),
        mesh
    ),
    mesh,
    dimensionedScalar(dimless/dimTime, Zero)
);

/*volScalarField::Internal Su
(
    IOobject
    (
        "Su",
        runTime.timeName(),
        mesh
    ),
    mesh,
    dimensionedScalar(dimless/dimTime, Zero)
);*/

volScalarField Su
(
    IOobject
    (
        "Su",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    dimensionedScalar(dimless/dimTime, Zero)
);

/*volScalarField SuTerm
(
    IOobject
    (
        "SuTerm",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    ),
    mesh,
    Su
);*/

volScalarField::Internal Sufi
(
    IOobject
    (
        "Sufi",
        runTime.timeName(),
        mesh
    ),
    mesh,
    
    dimensionedScalar(dimensionSet(0,-3,0,0,0,0,0), Zero)  //dimensionedScalar(dimensionSet(0,-3,0,0,0,1,0), Zero)  //[1 0 -3 0 0 -1 0];// V = kg*/(A*s^3) dimensionType {
  //MASS , LENGTH , TIME , TEMPERATURE ,
  //MOLES , CURRENT , LUMINOUS_INTENSITY }
);

volScalarField::Internal jmenosfidjdfi_f
(
    IOobject
    (
        "jmenosfidjdfi_f",
        runTime.timeName(),
        mesh
    ),
    mesh,
    
    dimensionedScalar(dimensionSet(0,0,0,0,0,1,0), Zero)  //[1 0 -3 0 0 -1 0];// V = kg*/(A*s^3) dimensionType {
  //MASS , LENGTH , TIME , TEMPERATURE ,
  //MOLES , CURRENT , LUMINOUS_INTENSITY }
);

volScalarField::Internal jmenosfidjdfi_s
(
    IOobject
    (
        "jmenosfidjdfi_s",
        runTime.timeName(),
        mesh
    ),
    mesh,
    
    dimensionedScalar(dimensionSet(0,0,0,0,0,1,0), Zero)  //[1 0 -3 0 0 -1 0];// V = kg*/(A*s^3) dimensionType {
  //MASS , LENGTH , TIME , TEMPERATURE ,
  //MOLES , CURRENT , LUMINOUS_INTENSITY }
);

volScalarField::Internal djdfi
(
    IOobject
    (
        "djdfi",
        runTime.timeName(),
        mesh
    ),
    mesh,
    
    dimensionedScalar(dimensionSet(-1,-2,3,0,0,2,0), Zero)  //[1 0 -3 0 0 -1 0];// V = kg*/(A*s^3) dimensionType {
  //MASS , LENGTH , TIME , TEMPERATURE ,
  //MOLES , CURRENT , LUMINOUS_INTENSITY }
);

volScalarField::Internal conc
(
    IOobject
    (
        "conc",
        runTime.timeName(),
        mesh
    ),
    mesh,
    
    dimensionedScalar(dimless/dimTime, Zero)  //[1 0 -3 0 0 -1 0];// V = kg*/(A*s^3) dimensionType {
  //MASS , LENGTH , TIME , TEMPERATURE ,
  //MOLES , CURRENT , LUMINOUS_INTENSITY }
);

volScalarField::Internal preconc
(
    IOobject
    (
        "preconc",
        runTime.timeName(),
        mesh
    ),
    mesh,
    
    dimensionedScalar(dimless/dimTime, Zero)  //[1 0 -3 0 0 -1 0];// V = kg*/(A*s^3) dimensionType {
  //MASS , LENGTH , TIME , TEMPERATURE ,
  //MOLES , CURRENT , LUMINOUS_INTENSITY }
);

