    //#include "alphaSuSp.H"
    
    fvScalarMatrix CEqn
    (
        fvm::ddt(C) //alpha2,
      + fvm::div(phi, C) //fvc::interpolate(alpha2)*
      - fvm::laplacian(D, C) 
     ==
        fvOptions(C) 
      + conc
      + fvm::Sp(preconc, C)
    );

    CEqn.relax();
    fvOptions.constrain(CEqn);
    CEqn.solve();
    fvOptions.correct(C);
    
    //C == min(max(C, dimensionedScalar("Cero", dimless, 1e-16)),dimensionedScalar("Uno", dimless, 1.0));
    
    Info<< "Min/max C: " << min(C).value() << ' ' << max(C).value() << endl;
