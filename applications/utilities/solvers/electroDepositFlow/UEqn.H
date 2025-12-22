    MRF.correctBoundaryVelocity(U);

    fvVectorMatrix UEqn
    (
        fvm::ddt(rho, U) + fvm::div(rhoPhi, U)
      + MRF.DDt(rho, U)
      + turbulence->divDevRhoReff(rho, U)
     ==
        fvOptions(rho, U) 
      + fvm::Sp(alpha1*dimensionedScalar("Su", dimMass/dimTime/dimVolume, -1e15),U)
    );
       
    UEqn.relax();

    fvOptions.constrain(UEqn);

    if (pimple.momentumPredictor())
    {
        solve
        (
            UEqn
         ==
            fvc::reconstruct
            (
                (
                    //mixture.surfaceTensionForce()
                  surfForces.surfaceTensionForce()
                  //- ghf*fvc::snGrad(rho)
                  + surfForces.accelerationForce()
                  - fvc::snGrad(p_rgh)
                ) * mesh.magSf()
            )
        );

        fvOptions.correct(U);
    }
