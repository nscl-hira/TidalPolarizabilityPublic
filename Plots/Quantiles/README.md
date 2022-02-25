This repository contains constraints on various relationships of interest in
analysis of the equation of state (EoS) of dense nuclear matter, as used in
the paper "Impact of the PSR J0740+6620 radius constraint on the properties
of high-density matter" by Legred, Chatziioannou, Essick, Han and Landry (2021)
doi: 10.1103/PhysRevD.104.063003.

This directory contains quantile files (see interpretation of .csv files)
organized hierarchically in directories by variables involved, and then
further by which distribution on equations of state we use (see
interpretation of file tags)  In total we use 5 different distributions,
and extract quantiles for 3 varaible relationships for a total of 15 quantiles.csv
files.

All quantities involving pressure and density are in cgs
pressurec2 is shorthand for pressure/c^2, where c is the speed of light
cs2c2  is shorthand for (speed of sound)^2/c^2 where c is the speed of light
Mass-Radius tables are given in units of (solar masses, km)


How to interpret file tags (Each file has a posterior tag, and a variables tag):
    Posterior Tags:
    prior: indicates quantiles are computed with the prior distribution
    psr: indicates quantiles are computed with the posterior computed using
         just heavy pulsar observations
    psrgw: indicates the quantiles are computed with the posterior using just the heavy
         pulsars and gravitational wave observations
    no_j0740: indicates the quantiles are computed with the posterior that excludes the
         any J0740+6620 radius measurement
    all_miller: indicates the quantiles are computed using all available constraints, which in
         this case indicates that the Miller+ radius measurement is used for J0740
         radius measurement.
    Variables Tags:
    prho: quantiles for pressurec2 (pressure/c^2) at select values of baryon rest mass density.
    mr : quantiles for radius (of a NS) at select values of mass (of a NS)
    rhocs2: quantiles for cs2c2 (speed of sound squared per unit speed of light squared) at
            select values of the baryon rest mass density.

NOTE: THIS DATA REPRESENTS A VISUALIZATION STRATEGY, AND IS NOT THE STARTING POINT FOR ANY ANALYSIS
      WE PERFORM.  In particular, it is not necessarily the case that if an EoS lies outside of the
      region marked by the envelope plot, that this EoS is necessarily excluded at 90% confidence..
      Many EoSs will lie outside of the 90 % confidence region at some density scale, and still be
      viable EoS candidates.  For examples of weighted EoS candidates, please contact the authors.  

How to interpret data in the .csv files:
    Each <tag>_quantiles.csv file contains a csv table with rows indexing a percentile value.
    The percentile is labeled in column 0 by field "quantile".
    Each successive column then represents the quantiles for the value which is labeled
    at the top of the column, based on our posterior distribution of equations of state
    corresponding to the tag <tag> (see above to interpret tags).
    For example in the file all_miller_prho_quantiles.csv (pressurec2 quantiles indexed by rho)
    the entry (1,1) lies in the column labeled "pressurec2(baryon_density=28000000000000.01)"
    and row labeled by "1.000000000000000021e-02" which indicates this is the first percentile
    of the quantity pressurec2 at baryon density.  It has a value of 5.529330516722413635e+10
    which indicates that the pressure/c^2 is greater than 5.529330516722413635e+10 g/cm^3 at
    99% credibility.  The median value is found, therefore, in the row representing 50%,
    labeled (approximately) .50.  Symmetric credible intervals can be found by selecting
    quantiles which are symmetric around the median (50%).  For example, a 90% symmetric credible
    interval can be formed by considering the (5% quantile value, 95% quantile value)

A script is provided for generating envelope plots (which show 90% credible intervals
at each selected value) from this data at :
  `plot_quantiles.py`
