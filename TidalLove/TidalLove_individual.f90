!-----------------------------------------------------------------------!
!    Copyright 2012, June 25 by F. J. Fattoyev, Commerce, TX, USA    !
!    Computation of the Neutron Star Mass-Radius Relation and    !
!    y(R) a quantity that needs to be supplemented for tidal        !
!    Love number k2 and tidal deformability lambda.            !
!
!    Modified by Tommy Tsang 5/8/2018 to be interfaced with Python
!-----------------------------------------------------------------------!
!     Purpose: This Program Calculates the Mass, the Radius and
!             the y(r) function of a neutron star to find the tidal 
!         Love number k2 and tidal polarizability.
!    UPDATE: Edited to calculate Dimensionless Tidal Polarizability.
!        May 1, 2018, Bloomington, IN, USA.
    !-----------------------------------------------------------------------!
subroutine TidalLove_individual(EOS_filename, pc, max_energy, pmin, &
                                num_checkpoint, checkpoint, &
                                mass_out, radius_out, &
                                lambda_out, checkpoint_mass, &
                                checkpoint_radius)
    implicit real*8 (a-h,k-z)    !Assign letters for reals and integers

    CHARACTER :: CR = CHAR(13)
    character(len=*),intent(in) :: EOS_filename
    integer, intent(in) :: num_checkpoint
    real*8, dimension(num_checkpoint), intent(in) :: checkpoint
    real*8, intent(in) :: max_energy
    real*8, intent(in) :: pmin
    real*8, dimension(num_checkpoint), intent(out) :: checkpoint_mass, checkpoint_radius
    real*8, intent(out) :: mass_out, radius_out, lambda_out
    real*8, intent(in) :: pc
    parameter(ipnts=1000000)    !The dimension of array for general output
    !Here (a-h,k-z) = reals, (i,j) = integers         
    dimension dens(ipnts),pres(ipnts),B(ipnts),C(ipnts),D(ipnts)     !Creates an array
        
    character*50 header                        !Max number of lines for header

    !print *, 'Opening file ', EOS_filename
    open(unit=3,file=EOS_filename)        ! Input profile: Provide an Equation of State 

    mevfm3 = 1.60217646d32            ! in J/m3 --> a converter from (MeV/fermi^3)
    p0 = 4.4173085d36            ! Pressure units in Pascal
    m0 = 1.0d0                ! Solar mass
    r0 = 1.47671618d0            ! Kilomteres
    e0 = 4.4173085d36            ! Joules/m3

    !-----------------------------------------------------------------------
    !    Read profiles from EOS file. Removes HEADERS:
    !-----------------------------------------------------------------------

    do ih=1,4                ! read headers
        read(3,'(a)') header            ! read header
    end do                    ! close ih-loop

    ir=0                    ! initialize counter
    do while (.true.)            ! read until EOF ends
       ir=ir+1                    ! update counter
       !------------------------------------------------------------------------
       !    Read the Equation of State and convert it to dimensionless numbers
       read(3,*,end=8) density, pressure, aa, bb    ! read from file
       !    Convert to dimensionless units
       dens(ir) = density*mevfm3/e0        ! dimensionless energy
       pres(ir) = pressure*mevfm3/p0        ! dimensionless pressure

    end do                    ! close while-loop
    8    continue                            ! read until EOF

    !open (unit=25,file="SoundSpeed.dat")    ! Tidal Number
    !write (25, *) '   R (km)        Density         cs^2 '
    !========================================================================
    !    In this DO loop we calculate Tidal Love Numbers 
    !========================================================================
    !open (unit=50,file="TidalLoveResults.dat")    ! Tidal Number
    !    write (50, *) '   Mass (MSun)        R (km)            Beta         k2        lambda        Rlambda (km)'
    !    write (50, *) '                                         (x1E29 (m^2 kg s^2))'
    !write (50, *) '   Mass (MSun)           R (km)              Lambda '

    !pc = 3.0d-5
     
    !do ip = 1, 1000
        !write(6,'(TL10,A,F6.1,A,A)',advance='no') "Progress = ", real(ip)/10., "%", CR
        !------------------------------------------------------------------------
        !    Matching the initial conditions to the solution
        !    An example central value for pressure:
        !    pc= 3.162805d-3            ! initial dimensionless pressure for IU-FSU, M=1.6 MSun

        p=pc*mevfm3/p0
        m=0.0d0                ! initial dimensionless mass
        r=1.0d-8            ! initial dimensionless radius
            y=2.0d0                ! initial value for y(0) 
        h=1.0d-4            ! step size
        !------------------------------------------------------------------------
        !========================================================================
        !    Interpolate a given pressure for a given density: START
        !========================================================================
        !------------------------------------------------------------------------
        !    Spline the two functions. Uses the external program or subroutine
        !    (called spline) to calculate these two values.
        !------------------------------------------------------------------------

            call csplin(ir-1,pres,dens,B,C,D)
        ec = cseval(ir-1,p,0,pres,dens,B,C,D)    ! corresponding central density     

        !print *, 'Pcentral=', pc*p0/mevfm3
        !print *, 'Ecentral=', ec*e0/mevfm3
        e=0

        icheckpoint = 1              ! counter for checkpoints
        do im= 1, 5000000            ! DO loop to calculate the mass/radius/k2
           !========================================================================
           !    Runge-Kutta method for solving ODE numerically.
           !========================================================================
           k1= h*e*r**2
           k2= h*e*(r+0.5d0*h)**2
           k3= h*e*(r+0.5d0*h)**2
           k4= h*e*(r+h)**2
           m= m + (k1+2.0d0*k2+2.0d0*k3+k4)/6.0d0            ! mass of the star
           if (m < 0) then
              go to 150
           end if

           k1= h*(e+p)*(m+p*r**3)/(2.0d0*m*r-r**2)
           k2= h*(e+(p+0.5d0*k1))*(m+(p+0.5d0*k1)*(r+0.5d0*h)**3)/(2.0d0*m*(r+0.5d0*h)-(r+0.5d0*h)**2)
           k3= h*(e+(p+0.5d0*k2))*(m+(p+0.5d0*k2)*(r+0.5d0*h)**3)/(2.0d0*m*(r+0.5d0*h)-(r+0.5d0*h)**2)
k4=        h*(e+(p+k3))*(m+(p+k3)*(r+h)**3)/(2.0d0*m*(r+h)-(r+h)**2)
           p= p + (k1+2.0d0*k2+2.0d0*k3+k4)/6.0d0            ! pressure inside the star

           !-------------------------------------------------------------------------
!           Calculation of y(r)
           !    NOTE: At phase transition the speed of sound will have a spurious oscillation! 
           !          To avoid that:
           !         (a) we assume the average constant speed at the vicinity of the phase transition.
           !         (b) we make a smooth polytropes ---> Smooth Polytropes Work!
           !          Current work is in progress...
           !    pcrit = 0.20746519d0*mevfm3/p0        !
           !    pcrit = 0.40198282d0*mevfm3/p0        ! 
           !    pcritplus = 1.02d0*pcrit 
           !    pcritminus= 0.98d0*pcrit
           !    if (p.gt.pcritplus) then
           oneovercs2 = cseval(ir-1,p,1,pres,dens,B,C,D)    ! corresponding one-over/speed-of-sound^squared    
           !    else if (p.lt.pcritminus) then
           !    oneovercs2 = cseval(ir-1,p,1,pres,dens,B,C,D)    ! corresponding one-over/speed-of-sound^squared    
           !    else 
           !    oneovercs2a = cseval(ir-1,pcritplus,1,pres,dens,B,C,D)    
           !    oneovercs2b = cseval(ir-1,pcritminus,1,pres,dens,B,C,D)    
!           oneovercs2 = (oneovercs2a*oneovercs2b)/(oneovercs2a+oneovercs2b)
           !    end if

           !    Phase Transition is taken into account!
           !-----------------------------------------------------------
           k1a = -y**2/r - (y/r)/(1.0d0-2.0d0*m/r)*(1.0d0+r**2*(p-e))
           k1b = -r/(1.0d0-2.0d0*m/r)*(5.0d0*e+9.0d0*p+(e+p)*oneovercs2)
           k1c = (6.0d0/r)/(1.0d0-2.0d0*m/r) + (1.0d0/r)*(2.0d0/(1.0d0-2.0d0*m/r)&
                   *(m+r**3*p)/r)**2 
           k1 = h*(k1a + k1b + k1c)

           k2a = -(y+0.5d0*k1)**2/(r+0.5d0*h) 
           k2a = k2a - ((y+0.5d0*k1)/(r+0.5d0*h))/(1.0d0-2.0d0*m/(r+0.5d0*h))*(1.0d0+(r+0.5d0*h)**2*(p-e))
           k2b = -(r+0.5d0*h)/(1.0d0-2.0d0*m/(r+0.5d0*h))*(5.0d0*e+9.0d0*p+(e+p)&
                   *oneovercs2)
           k2c = (6.0d0/(r+0.5d0*h))/(1.0d0-2.0d0*m/(r+0.5d0*h)) 
           k2c = k2c + (1.0d0/(r+0.5d0*h))*(2.0d0/(1.0d0-2.0d0*m/(r+0.5d0*h))*(m+&
                       (r+0.5d0*h)**3*p)/(r+0.5d0*h))**2
           k2 = h*(k2a + k2b + k2c)

           k3a = -(y+0.5d0*k2)**2/(r+0.5d0*h) 
           k3a = k3a - ((y+0.5d0*k2)/(r+0.5d0*h))/(1.0d0-2.0d0*m/(r+0.5d0*h))*(1.0d0+(r+0.5d0*h)**2*(p-e))
           k3 = h*(k3a + k2b + k2c)

           k4a = -(y+k3)**2/(r+h) - ((y+k3)/(r+h))/(1.0d0-2.0d0*m/(r+h))*(1.0d0+(&
                       r+h)**2*(p-e))
           k4b = -(r+h)/(1.0d0-2.0d0*m/(r+h))*(5.0d0*e+9.0d0*p+(e+p)*oneovercs2)
           k4c = (6.0d0/(r+h))/(1.0d0-2.0d0*m/(r+h)) + (1.0d0/(r+h))*(2.0d0/(&
                       1.0d0-2.0d0*m/(r+h))*(m+(r+h)**3*p)/(r+h))**2
           k4 = h*(k4a + k4b + k4c)

           y = y + (k1+2.0d0*k2+2.0d0*k3+k4)/6.0d0        ! y function inside the star

           !TEST TEST TEST TEST TEST TEST TEST TEST TEST ...
           !------------------------------------------------------------------
           test1 = -r/(1.0d0-2.0d0*m/r)*(5.0d0*e+9.0d0*p)
           test2 = -r/(1.0d0-2.0d0*m/r)*((e+p)*oneovercs2)
           test3 = (6.0d0/r)/(1.0d0-2.0d0*m/r) 
           test4 = (1.0d0/r)*(2.0d0/(1.0d0-2.0d0*m/r)*(m+r**3*p)/r)**2
           !--------------------------------------------------------------------

           r= r + h                    ! radius of the star    
           e = cseval(ir-1,p,0,pres,dens,B,C,D)        ! corresponding density    

           if (e .gt. max_energy*mevfm3/e0) then
              m = -1
              r = -1
              dimlambda = -1
              do idx = 1, num_checkpoint
                  checkpoint_mass(idx) = 0
                  checkpoint_radius(idx) = 0
              end do
              go to 150
           end if


           if (icheckpoint <= num_checkpoint) then
               pcheckpoint = checkpoint(icheckpoint)*mevfm3/p0

               if (P .le. pcheckpoint) then
                   checkpoint_mass(icheckpoint) = m
                   checkpoint_radius(icheckpoint) = r*r0
                   icheckpoint = icheckpoint + 1
               end if
           end if

           if (P .le. pmin*mevfm3/p0) then  
               go to 20
           end if

           !beta = m/r

           !XA = 0.5d0    ! For equal mass binary
           !XB = XA
           !CA = beta
           !CB = CA
           !XContact = 1.0d0/(XA/CA + XB/CB)
           !fcontact = XContact**(1.5d0)/(3.141592654d0*(2.0d0*m*1.98892d30*6.673d-11)/(299792458.0d0**3.0d0))
           !!print *, 'fcontact = ', fcontact, 'Hz'

           !term = 2.0d0*beta**2*(13.0d0-11.0d0*y+beta*(3.0d0*y-2.0d0)+2.0d0*beta&
           !        **2*(1.0d0+y))

           !kk2a=2.0d0*beta*(6.0d0-3.0d0*y+3.0d0*beta*(5.0d0*y-8.0d0)+term)
           !kk2b=3.0d0*(1.0d0-2.0d0*beta)**2*(2.0d0-y+2.0d0*beta*(y-1))*log(1.0d0-2.0d0*beta) 
           !k2=8.0d0/5.0d0*beta**5*(1.0d0-2.0d0*beta)**2*(2.0d0-y+2.0d0*beta*(y-1.0d0))/(kk2a+kk2b)

           !lambda = 2.0d0*k2*(r*r0*1.0d3)**5/(3.0d0*(6.673d-11))        ! in m^2 kg s^2
           !Rlambda = 1.0d-3*((6.673d-11)*lambda)**(1.0d0/5.0d0)        ! in km

           !dimlambda = lambda/((6.673d-11)**4.0d0)/((m*1.989d30)**5.0d0)*(299792458.0d0)**10.0d0
           !print *, m, r*r0, e*mevfm3/e0, P*mevfm3/p0, dimlambda

           

           !    write (25, 500) r*r0, e/mevfm3*e0/931.5d0, 1.0d0/oneovercs2

    end do

20  continue
    !print *, 'Mass is = ', m*m0, 'MSun'
    !print *, 'Radius is = ', r*r0, 'km'
    !print *, 'y(R) = ', y

    !    TEST TEST TEST TEST
    !    m = 1.0d0/m0*1.9888d0/1.98892d0
    !    r = 16.0d0/r0
    !    y = 2.0d0
    !    TEST TEST TEST

    beta = m/r

    XA = 0.5d0    ! For equal mass binary
    XB = XA
    CA = beta
    CB = CA
    XContact = 1.0d0/(XA/CA + XB/CB)
fcontact = XContact**(1.5d0)/(3.141592654d0*(2.0d0*m*1.98892d30*6.673d-11)/(299792458.0d0**3.0d0))
    !print *, 'fcontact = ', fcontact, 'Hz'

    term = 2.0d0*beta**2*(13.0d0-11.0d0*y+beta*(3.0d0*y-2.0d0)+2.0d0*beta&
            **2*(1.0d0+y))

    kk2a=2.0d0*beta*(6.0d0-3.0d0*y+3.0d0*beta*(5.0d0*y-8.0d0)+term)
    kk2b=3.0d0*(1.0d0-2.0d0*beta)**2*(2.0d0-y+2.0d0*beta*(y-1))*log(1.0d0-2.0d0*beta) 
k2 = 8.0d0/5.0d0*beta**5*(1.0d0-2.0d0*beta)**2*(2.0d0-y+2.0d0*beta*(y-1.0d0))/(kk2a+kk2b)

    lambda = 2.0d0*k2*(r*r0*1.0d3)**5/(3.0d0*(6.673d-11))        ! in m^2 kg s^2
    Rlambda = 1.0d-3*((6.673d-11)*lambda)**(1.0d0/5.0d0)        ! in km

    dimlambda = lambda/((6.673d-11)**4.0d0)/((m*1.989d30)**5.0d0)*(299792458.0d0)**10.0d0

    !print *, 'beta=', beta
    !print *, 'k2 = ', k2
    !print *, 'lambda = ', lambda, 'x 10^29 m^2 kg s^2'
    !print *, 'DimLambda = ', dimlambda
    !print *, 'R_lambda = ', Rlambda, 'km'
    sigmal = 1.0d35*((2.0d0*m)**2.5d0)/(fcontact)**(2.2d0)*(50.0d0/100.0d0)
    !print *, 'SigmaLambda(Adv. LIGO) =', sigmal
    lambdaPAdvLigo = lambda + sigmal
    lambdaMAdvLigo = lambda - sigmal
    !print *, sigmal/lambda*1.0d2, '%'
sigmal = 8.4d33*((2.0d0*m)**2.5d0)/(fcontact)**(2.2d0)*(50.0d0/100.0d0)
    !print *, 'SigmaLambda(Einstein Tel.) =', sigmal
    lambdaPEinstein = lambda + sigmal
    lambdaMEinstein = lambda - sigmal
    !print *, sigmal/lambda*1.0d2, '%'
    !print *, '=================================='

    !    write (50, 500) m, r*r0, beta, k2, lambda/1.0d29, Rlambda
    ! write (50, 500) m, r*r0, dimlambda
    !    mass_arr(ip) = m
    !    radius_arr(ip) = r*r0
    !    lambda_arr(ip) = dimlambda
    !    write (50, 500) m, lambda/1.0d29, lambdaPAdvLigo/1.0d29, lambdaMAdvLigo/1.0d29, lambdaPEinstein/1.0d29, lambdaMEinstein/1.0d29 ! Only for errors
150 continue
    !pc = pc + 2.0d-5
    mass_out = m
    radius_out = r*r0
    lambda_out = dimlambda
    rewind(3)
    close(3)
    !end do

!500    format(6e18.8)
end

    !-----------------------------------------------------------------------
    !     *********************** CUBIC SPLINE *****************************
    !-----------------------------------------------------------------------
    !     LAST MODIFIED 4-10-76 BY W. M. COUGHRAN, JR.
SUBROUTINE CSPLIN (N, X, Y, B, C, D)
    INTEGER N
DOUBLE PRECISION X(N), Y(N), B(N), C(N), D(N)
    !
    !  THIS IS A MODIFIED VERSION OF SPLINE, DESCRIBED IN THE NOTES
    !  BY FORSYTHE MALCOLM AND MOLER
    !
    !  THE COEFFICIENTS B(I), C(I), AND D(I), I=1,2,...,N ARE COMPUTED
    !  FOR A CUBIC INTERPOLATING SPLINE
    !
    !    S(X) = Y(I) + B(I)*(X-X(I)) + C(I)*(X-X(I))**2 + D(I)*(X-X(I))**3
    !
!    FOR  X(I) .LE. X .LE. X(I+1)
    !
    !  INPUT..
    !
!    N = THE NUMBER OF DATA POINTS OR KNOTS (N.GE.2)
    !    X = THE ABSCISSAS OF THE KNOTS IN STRICTLY INCREASING ORDER
    !    Y = THE ORDINATES OF THE KNOTS
    !
    !  OUTPUT..
    !
    !    B, C, D  = ARRAYS OF SPLINE COEFFICIENTS AS DEFINED ABOVE.
    !
    !  USING  P  TO DENOTE DIFFERENTIATION,
    !
    !    Y(I) = S(X(I))
!    B(I) = SP(X(I))
    !    C(I) = SPP(X(I))/2
!    D(I) = SPPP(X(I))/6  (DERIVATIVE FROM THE RIGHT)
    !
    !  THE ACCOMPANYING FUNCTION SUBPROGRAM  CSEVAL  CAN BE USED
    !  TO EVALUATE THE SPLINE, ITS DERIVATIVE OR EVEN ITS 2ND DERIVATIVE.
    !
    !       SEE COMPUTER METHODS FOR MATHEMATICAL COMPUTATIONS BY
    !           FORSYTHE, MALCOLM, AND MOLER FOR DETAILS
    !
    INTEGER LOUT, NM1, IB, I
    DOUBLE PRECISION T
    DATA LOUT/6/
    !
    !         CHECK INPUT FOR CONSISTENCY
    !
    IF (N .GE. 2) GO TO 1
    !WRITE(LOUT, 1000)N
    RETURN
    !
    1 NM1 = N-1
    DO 3 I = 1, NM1
    IF (X(I) .LT. X(I+1)) GO TO 3
!WRITE(LOUT, 1001)
    RETURN
    3 CONTINUE
    !
    5 IF ( N .EQ. 2 ) GO TO 50
    !
    !  SET UP TRIDIAGONAL SYSTEM
    !
    !  B = DIAGONAL, D = OFFDIAGONAL, C = RIGHT HAND SIDE.
    !
    D(1) = X(2) - X(1)
C(2) = (Y(2) - Y(1))/D(1)
    DO 10 I = 2, NM1
    D(I) = X(I+1) - X(I)
    B(I) = 2.*(D(I-1) + D(I))
    C(I+1) = (Y(I+1) - Y(I))/D(I)
C(I) = C(I+1) - C(I)
    10 CONTINUE
    !
!  END CONDITIONS.  THIRD DERIVATIVES AT  X(1)  AND  X(N)
    !  OBTAINED FROM DIVIDED DIFFERENCES
    !
    B(1) = -D(1)
B(N) = -D(N-1)
    C(1) = 0.
    C(N) = 0.
    IF ( N .EQ. 3 ) GO TO 15
    C(1) = C(3)/(X(4)-X(2)) - C(2)/(X(3)-X(1))
    C(N) = C(N-1)/(X(N)-X(N-2)) - C(N-2)/(X(N-1)-X(N-3))
    C(1) = C(1)*D(1)**2/(X(4)-X(1))
C(N) = -C(N)*D(N-1)**2/(X(N)-X(N-3))
    !
    !  FORWARD ELIMINATION
    !
    15 DO 20 I = 2, N
    T = D(I-1)/B(I-1)
    B(I) = B(I) - T*D(I-1)
C(I) = C(I) - T*C(I-1)
    20 CONTINUE
    !
    !  BACK SUBSTITUTION
    !
C(N) = C(N)/B(N)
    DO 30 IB = 1, NM1
    I = N-IB
C(I) = (C(I) - D(I)*C(I+1))/B(I)
    30 CONTINUE
    !
    !  C(I) IS NOW THE SIGMA(I) OF THE TEXT
    !
    !  COMPUTE POLYNOMIAL COEFFICIENTS
    !
B(N) = (Y(N) - Y(NM1))/D(NM1) + D(NM1)*(C(NM1) + 2.*C(N))
    DO 40 I = 1, NM1
    B(I) = (Y(I+1) - Y(I))/D(I) - D(I)*(C(I+1) + 2.*C(I))
    D(I) = (C(I+1) - C(I))/D(I)
C(I) = 3.*C(I)
    40 CONTINUE
    C(N) = 3.*C(N)
D(N) = D(N-1)
    RETURN
50 B(1) = (Y(2)-Y(1))/(X(2)-X(1))
    C(1) = 0.
    D(1) = 0.
    RETURN
    1000 FORMAT('-N < 2 IN CSPLIN CALL--',I10)
    1001 FORMAT('-X IS NOT IN ASCENDING ORDER IN CSPLIN CALL')
    END
DOUBLE PRECISION FUNCTION CSEVAL (N, U, IDERIV, X, Y, B, C, D)
    INTEGER N, IDERIV
DOUBLE PRECISION U, X(N), Y(N), B(N), C(N), D(N)
    !
    !  THIS IS A MODIFIED VERSION OF SEVAL, DESCRIBED IN THE NOTES
    !  BY FORSYTHE MALCOLM AND MOLER
    !
    !  THIS SUBROUTINE EVALUATES THE CUBIC SPLINE FUNCTION, ITS FIRST
    !  DERIVATIVE OR SECOND DERIVATIVE, NAMELY:
    !
    !   IDERIV=0:    Y(I)+B(I)*(U-X(I))+C(I)*(U-X(I))**2+D(I)*(U-X(I))**3
    !   IDERIV=1:    B(I)+2*C(I)*(U-X(I))+3*D(I)*(U-X(I))**2
!   IDERIV=2:    2*C(I)+6*D(I)*(U-X(I))
    !
    !    WHERE  X(I) .LT. U .LT. X(I+1), USING HORNER'S RULE
    !
    !  IF  U .LT. X(1) THEN  I = 1  IS USED.
    !  IF  U .GE. X(N) THEN  I = N  IS USED.
    !
    !  INPUT..
    !
    !    N = THE NUMBER OF DATA POINTS
    !    U = THE ABSCISSA AT WHICH THE SPLINE IS TO BE EVALUATED
    !    IDERIV = 0 TO EVALUATE S(U)
    !           = 1 TO EVALUATE SP(U)
!           = 2 TO EVALUATE SPP(U)
    !    X,Y = THE ARRAYS OF DATA ABSCISSAS AND ORDINATES
    !    B,C,D = ARRAYS OF SPLINES COEFFICIENTS COMPUTED BY SUBROUTINE CSPLI
    !
    !  IF  U  IS NOT IN THE SAME INTERVAL AS THE PREVIOUS CALL, THEN A
    !    BINARY SEARCH IS PERFORMED TO DETERMINE THE PROPER INTERVAL.
    !
    INTEGER I, J, K, LOUT
    DOUBLE PRECISION DX
    DATA I, LOUT/1, 6/
    !
    IF (IDERIV .GE. 0 .AND. IDERIV .LE. 2) GO TO 5
    !WRITE(LOUT, 1000)IDERIV
    RETURN
    !
    5 IF ( I .GE. N ) I = 1
    IF ( U .LT. X(I) ) GO TO 10
    IF ( U .LE. X(I+1) ) GO TO 30
    !
    !  BINARY SEARCH
    !
    10 I = 1
    J = N + 1
    20 K = (I+J)/2
    IF ( U .LT. X(K) ) J = K
    IF ( U .GE. X(K) ) I = K
    IF ( J .GT. I+1 ) GO TO 20
    !
    !  EVALUATE SPLINE
    !
30 DX = U - X(I)
    J = IDERIV+1
    GO TO (40,50,60), J
    !  COMPUTE S(X)
40 CSEVAL = Y(I) + DX*(B(I) + DX*(C(I) + DX*D(I)))
    RETURN
    !  COMPUTE SP(X)
50 CSEVAL = B(I) + DX*(2*C(I) + 3*DX*D(I))
    RETURN
    !  COMPUTE SPP(X)
60 CSEVAL = 2*C(I) + 6*DX*D(I)
    RETURN
    1000 FORMAT('-IDERIV IS INVALID IN CSEVAL CALL--', I10)
    END
    !-----------------------------------------------------------------------
