.. _multiphysics-collisions:

Collisions
==========

WarpX includes several different models to capture collisional processes
including collisions between kinetic particles (Coulomb collisions, DSMC,
nuclear fusion, electron-impact ionization) as well as collisions between kinetic particles and a fixed
(i.e. non-evolving) background species (MCC, background stopping).

.. _multiphysics-collisions-mcc:

Background Monte Carlo Collisions (MCC)
---------------------------------------

Several types of collisions between simulation particles and a neutral
background gas are supported including elastic scattering, back scattering,
charge exchange, excitation collisions and impact ionization.

The so-called null collision strategy is used in order to minimize the
computational burden of the MCC module. This strategy is standard in PIC-MCC and
a detailed description can be found elsewhere, for example in :cite:t:`b-Birdsall1991`.
In short the maximum collision probability is found over a sensible range of
energies and is used to pre-select the appropriate number of macroparticles for
collision consideration. Only these pre-selected particles are then individually
considered for a collision based on their energy and the cross-sections of all
the different collisional processes included.

The MCC implementation assumes that the background neutral particles are **thermal**,
and are moving at non-relativistic velocities in the lab frame. For each
simulation particle considered for a collision, a velocity vector for a neutral
particle is randomly chosen given the user specified neutral temperature. The
particle velocity is then boosted to the stationary frame of the neutral through
a Galilean transformation. The energy of the collision is calculated using the
particle utility function, ``ParticleUtils::getCollisionEnergy()``, as

    .. math::

       \begin{aligned}
        E_{coll} &= \sqrt{(\gamma mc^2 + Mc^2)^2 - (mu)^2} - (mc^2 + Mc^2) \\
                 &= \frac{2Mmu^2}{M + m + \sqrt{M^2+m^2+2\gamma mM}}\frac{1}{\gamma + 1}
       \end{aligned}

where :math:`u` is the speed of the particle as tracked in WarpX (i.e.
:math:`u = \gamma v` with :math:`v` the particle speed), while :math:`m` and
:math:`M` are the rest masses of the simulation and background species,
respectively. The Lorentz factor is defined in the usual way,
:math:`\gamma \equiv \sqrt{1 + u^2/c^2}`. Note that if :math:`\gamma\to1` the above
expression reduces to the classical equation
:math:`E_{coll} = \frac{1}{2}\frac{Mm}{M+m} u^2`. The collision cross-sections
for all scattering processes are evaluated at the energy as calculated above.

Once a particle is selected for a specific collision process, that process determines how the particle is scattered as outlined below.

.. _multiphysics-collisions-pulseddecay:

Pulsed Decay
------------

This collision module can be used to have a parent species decay into two product
species with a user-defined decay rate. Mathematically, it solves the following
rate equations on a particle basis:

    .. math::

       \begin{aligned}
        \frac{dN_1}{dt} &= -\nu(t)N_1, \\
        \frac{dN_A}{dt} &= +\nu(t)N_1 = \frac{dN_B}{dt}, \\
       \end{aligned}

where :math:`N_1` is the number of the parent species particles, :math:`N_A` and :math:`N_B` are the numbers of the product species, and :math:`\nu(x,y,z,t)` is the user-specified decay rate.
For each parent particle, the decay probability is calculated as :math:`1-\exp[-\nu(x,y,z,t)\Delta t]`, where :math:`\Delta t` is the time-step size.
If that probability is greater than a randomly chosen number, the parent particle decays into the products whose number and weight is scaled by the integer weight ratio.
After decaying, the parent particle is removed.

This can be used, for example, to represent ionization of a parent species by an externally applied laser pulse.

.. _multiphysics-collisions-dsmc:

Direct Simulation Monte Carlo (DSMC)
------------------------------------

The algorithm by which binary collisions are treated is outlined below. The
description assumes collisions between different species.

1. Particles from both species are sorted by grid-cells.
2. The order of the particles in each cell is shuffled.
3. Within each cell, particles are paired to form collision partners. Particles
   of the species with fewer members in a given cell is split in half so that
   each particle has exactly one partner of the other species.
4. Each collision pair is considered for a collision using the same logic as in
   the MCC description above.
5. Particles that are chosen for collision are scattered according to the
   selected collision process.

Scattering processes
--------------------

Charge exchange
^^^^^^^^^^^^^^^

This process can occur when an ion and neutral (of the same species) collide
and results in the exchange of an electron. The ion velocity is simply replaced
with the neutral velocity and vice-versa.

Elastic scattering
^^^^^^^^^^^^^^^^^^

The ``elastic`` option uses isotropic scattering, i.e., with a differential
cross section that is independent of angle.
This scattering process as well as the ones below that relate to it, are all
performed in the center-of-momentum (COM) frame. Designating the COM velocity of
the particle as :math:`\boldsymbol{u}_c` and its labframe velocity as :math:`\boldsymbol{u}_l`,
the transformation from lab frame to COM frame is done with a general Lorentz
boost (see function ``ParticleUtils::doLorentzTransform()``):

    .. math::
            \begin{bmatrix}
                \gamma_c c \\
                u_{cx} \\
                u_{cy} \\
                u_{cz}
            \end{bmatrix}
         = \begin{bmatrix}
                \gamma & -\gamma\beta_x & -\gamma\beta_y & -\gamma\beta_z \\
                -\gamma\beta_x & 1+(\gamma-1)\frac{\beta_x^2}{\beta^2} & (\gamma-1)\frac{\beta_x\beta_y}{\beta^2} & (\gamma-1)\frac{\beta_x\beta_z}{\beta^2} \\
                -\gamma\beta_y & (\gamma-1)\frac{\beta_x\beta_y}{\beta^2} & 1 +(\gamma-1)\frac{\beta_y^2}{\beta^2} & (\gamma-1)\frac{\beta_y\beta_z}{\beta^2} \\
                -\gamma\beta_z & (\gamma-1)\frac{\beta_x\beta_z}{\beta^2} & (\gamma-1)\frac{\beta_y\beta_z}{\beta^2} & 1+(\gamma-1)\frac{\beta_z^2}{\beta^2} \\
            \end{bmatrix} \begin{bmatrix}
                \gamma_l c \\
                u_{lx} \\
                u_{ly} \\
                u_{lz}
            \end{bmatrix}

where :math:`\gamma` is the Lorentz factor of the relative speed between the lab frame and the COM frame, :math:`\beta_i = v^{COM}_i/c` is the i'th component of the relative velocity between the lab frame and the COM frame with

    .. math::

        \boldsymbol{v}^{COM} = \frac{m \boldsymbol{u}_c}{\gamma_u m + M}

The particle velocity in the COM frame is then isotropically scattered using the function ``ParticleUtils::RandomizeVelocity()``. After the direction of the velocity vector has been appropriately changed, it is transformed back to the lab frame with the reversed Lorentz transform as was done above followed by the reverse Galilean transformation using the starting neutral velocity.

Back scattering
^^^^^^^^^^^^^^^

The process is the same as for elastic scattering above except the scattering angle is fixed at :math:`\pi`, meaning the particle velocity in the COM frame is updated to :math:`-\boldsymbol{u}_c`.

Excitation
^^^^^^^^^^

The process is also the same as for elastic scattering except the excitation energy cost is subtracted from the particle energy. This is done by reducing the velocity before a scattering angle is chosen.

Forward scattering
^^^^^^^^^^^^^^^^^^

This process operates in two ways:

1. If an excitation energy cost is provided, the energy cost is subtracted from the particle energy and no scattering is performed.
2. If an excitation energy cost is not provided, the particle is not scattered and the velocity is unchanged (corresponding to a scattering angle of :math:`0` in the elastic scattering process above).

See :cite:t:`b-Janssen2016` for a recommended use of this process.

Benchmarks
----------

See the :ref:`MCC example <examples-capacitive-discharge>` for a benchmark of the MCC
implementation against literature results.

Particle cooling due to elastic collisions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is straight forward to determine the energy a projectile loses during an elastic collision with another body, as a function of scattering angle, through energy and momentum conservation.
See for example :cite:t:`b-Lim2007` for a derivation. The result is that given a projectile with mass :math:`m`, a target with mass :math:`M`, a scattering angle :math:`\theta`, and collision energy :math:`E`, the post collision energy of the projectile is given by

    .. math::

       \begin{aligned}
       E_{final} = E - &[(E + mc^2)\sin^2\theta + Mc^2 - \cos(\theta)\sqrt{M^2c^4 - m^2c^4\sin^2\theta}] \\
       &\times \frac{E(E+2mc^2)}{(E+mc^2+Mc^2)^2 - E(E+2mc^2)\cos^2\theta}
       \end{aligned}

The impact of incorporating relativistic effects in the MCC routine can be seen in the plots below where high energy collisions are considered with both a classical and relativistic implementation of MCC. It is observed that the classical version of MCC reproduces the classical limit of the above equation but especially for ions, this result differs substantially from the fully relativistic result.

.. figure:: https://user-images.githubusercontent.com/40245517/170900079-74e505a5-2790-44f5-ac84-5847eda954e6.png
   :alt: Comparison of classical and relativistic MCC collision results
   :width: 96%

   Classical v. relativistic MCC.

.. _multiphysics-collisions-electron-impact-ionization:

Electron-impact ionization
--------------------------

The ``electron_impact_ionization`` collision type models the ionization of ions by collisions with free electrons,
:math:`e^- + A^{q+} \rightarrow 2e^- + A^{(q+1)+}`.
This is the dominant ionization mechanism in dense, collisional plasmas in which the electric field is too weak
for field ionization, for instance in solid targets heated by the pedestal of a high-intensity laser pulse.
The ions are represented by a species with field ionization enabled, which provides the ionization level
of each macroparticle and the physical element of the ions.
Field ionization and electron-impact ionization can therefore be used together.

Atomic structure
^^^^^^^^^^^^^^^^

The ionization cross section of an ion in the charge state :math:`q` is the sum of the cross sections of its
occupied subshells.
The ground-state configuration of the ion is obtained by removing the :math:`q` least bound electrons of the neutral atom,
and the binding energies of its subshells are estimated as in :cite:t:`b-CarlsonAtData1970`:
the binding energies of the corresponding electrons in the neutral atom (taken from a table of free-atom binding
energies, which is also used in the Smilei code) are shifted by the same amount, such that the binding energy
of the outermost electron is equal to the ionization energy :math:`I_q` of the ion
(taken from the same table of ionization energies as for field ionization):

    .. math::

        B_s(q) = B_s(0) + I_q - B_{\mathrm{outer}}(0)

where :math:`B_s(0)` is the binding energy of the subshell :math:`s` in the neutral atom and :math:`B_{\mathrm{outer}}(0)`
the binding energy, in the neutral atom, of the outermost electron of the ion.

Cross sections
^^^^^^^^^^^^^^

Three models are available for the cross section of a subshell with :math:`N_s` electrons and binding energy :math:`B_s`,
for an incident electron of kinetic energy :math:`E` in the rest frame of the ion
(see :pp:param:`<collision_name>.cross_section_model`).
In all cases, the cross section vanishes for :math:`E \leq B_s`.

* The modified Bell (MBELL) model of :cite:t:`b-HaquePhysScr2006` is a fit of experimental and theoretical cross sections
  for the subshells :math:`1s`, :math:`2s`, :math:`2p`, :math:`3s` and :math:`3p`, and is therefore available for
  atomic numbers :math:`Z \leq 18`. It is also used in the EPOCH code :cite:p:`b-MorrisPoP2022`:

    .. math::

        \sigma_s = N_s F_{\mathrm{ion}}\, G_R\, \frac{1}{E B_s}\left[ A_{nl} \ln u + \sum_{k=1}^{7} B_{nl,k} \left(1 - \frac{1}{u}\right)^k \right]

  where :math:`u = E/B_s`, :math:`A_{nl}` and :math:`B_{nl,k}` are the fitted parameters of the subshell,
  :math:`G_R` is the relativistic factor of Gryzinski, and
  :math:`F_{\mathrm{ion}} = 1 + 3\left(q_{\mathrm{eff}}/(Z u)\right)^{\lambda}` is a correction factor for ions,
  with :math:`\lambda = 1.27` for :math:`s` subshells and :math:`\lambda = 0.542` for :math:`p` subshells, and with
  :math:`q_{\mathrm{eff}}` the atomic number minus the number of electrons in the subshell and in the subshells
  that are more bound.

* The empirical formula of :cite:t:`b-LotzZPhys1967` (see also :cite:t:`b-LotzZPhys1968`), extended to relativistic
  electrons in the same way as in the relativistic binary-encounter-Bethe model of :cite:t:`b-KimPRA2000`:

    .. math::

        \sigma_s = a N_s \frac{\ln(E/B_s) + \left(1 - B_s^2/E^2\right)\delta}{B_s\, m_e c^2 \beta^2/2},
        \qquad \delta = \ln\left(\frac{\gamma+1}{2}\right) - \beta^2

  where :math:`a = 4.5 \times 10^{-14}\,\mathrm{cm^2\,eV^2}`, and :math:`\gamma` and :math:`\beta` are the Lorentz factor
  and the normalized velocity of the incident electron in the rest frame of the ion.
  The original formula of Lotz is recovered when :math:`E \ll m_e c^2`, while at high energy the logarithmic term tends to
  :math:`\ln\left(\beta^2\gamma^2 m_e c^2/2B_s\right) - \beta^2`, which gives the relativistic rise of the Bethe formula.
  The threshold factor :math:`1 - B_s^2/E^2` ensures that the cross section remains positive above the ionization
  threshold of the inner subshells of heavy ions.

* The relativistic binary-encounter-Bethe (RBEB) model of :cite:t:`b-KimPRA2000` (their Eq. 22), in which the mean kinetic
  energy of the bound electrons is approximated by their binding energy:

    .. math::

        \sigma_s = \frac{2\pi r_e^2 N_s}{(\beta_t^2 + 2\beta_b^2)\, b'} \left[ \frac{A_3}{2}\left(1 - \frac{1}{t^2}\right)
        + 1 - \frac{1}{t} - \frac{\ln t}{t+1}\frac{1+2t'}{(1+t'/2)^2} + \frac{b'^2}{(1+t'/2)^2}\frac{t-1}{2} \right]

  where :math:`t = E/B_s`, :math:`t' = E/m_e c^2`, :math:`b' = B_s/m_e c^2`, :math:`\beta_t^2 = 1 - 1/(1+t')^2`,
  :math:`\beta_b^2 = 1 - 1/(1+b')^2` and :math:`A_3 = \ln\left(\beta_t^2/(1-\beta_t^2)\right) - \beta_t^2 - \ln(2b')`.

By default (``auto``), the MBELL model is used for :math:`Z \leq 18` and the Lotz model otherwise.
Compared with the ionization rate coefficients recommended by :cite:t:`b-DereAA2007` for the elements from hydrogen to zinc
in thermal plasmas, the MBELL model is the most accurate for :math:`Z \leq 18` (typically within 30%),
the Lotz model is accurate for ions but overestimates the ionization of neutral atoms (by up to a factor of 3),
and the RBEB model with the approximations above underestimates the ionization of ions (by up to a factor of 2.5 for
temperatures below the ionization energy).
For fast electrons, however, the RBEB model should be preferred: the cross sections of the MBELL fit increase too fast
with the energy above about 10 keV (they are about 30% larger than those of RBEB at 1 MeV for H, He and C), such that the
energy lost by fast electrons in ionization exceeds their total collisional stopping power (by 20% for H and 10% for Al at
1 MeV).
Note also that, with the approximation of the BEB model for the dipole oscillator strength, the RBEB cross sections follow
at high energy the Bethe theory with the coefficient :math:`\sum_s N_s R/(2 B_s)` of :math:`\ln E`, which differs from the
actual dipole matrix element for some atoms (0.5 instead of 0.283 for hydrogen): at equal velocities, the RBEB cross
sections of H and Ne exceed the ion-impact cross sections (see below), which follow the measured Bethe coefficients, by
40% to 50%, while those of He and Ar agree within 15%.

Algorithm
^^^^^^^^^

The electron and ion macroparticles are sorted by cell and paired as in the other binary collision models.
For each pair of macroparticles, of weights :math:`w_e` and :math:`w_i`, the expected number of ionization events
of weight :math:`w_{\min} = \min(w_e, w_i)` during the collision time step :math:`\Delta t` is computed as in the
algorithm of :cite:t:`b-HigginsonJCP2019`:

    .. math::

        x = N_{\min}\, w_{\max}\, \sigma v_{rel}\, \Delta t / \Delta V

where :math:`N_{\min}` is the number of macroparticles of the species with fewer macroparticles in the cell
(which takes into account the pairs that are not sampled), :math:`w_{\max} = \max(w_e, w_i)` and :math:`\Delta V` is
the volume of the cell. As for nuclear fusion, the reaction rate :math:`\sigma v_{rel}` is evaluated in the
center-of-momentum frame of the pair and converted to the lab frame.
Two methods are then available (see :pp:param:`<collision_name>.split_macroparticles`):

* By default, the macroparticles are not split.
  The ionization events of each pair are sampled as a Poisson process of rate :math:`x/\Delta t`
  (with :math:`x` recomputed after each event, since the energy of the electron and the charge state of the ion change),
  which allows several successive ionizations within one collision time step.
  Each event is applied to the electron with the probability :math:`w_{\min}/w_e` and to the ion with the probability
  :math:`w_{\min}/w_i` (using the same random number, so that it is always applied to the macroparticle with the smallest weight).
  When the event is applied to the electron, its momentum is updated as described below.
  When it is applied to the ion, its ionization level is increased by one, its recoil momentum is applied,
  and an ejected electron with the weight :math:`w_i` is created at the position of the ion.
  The charge is thus conserved exactly, and the energy and the momentum are conserved on average, even when the weights
  are equal, since the ejected electrons are sampled after the collisions, independently of the electrons used to
  compute the momentum changes of the colliding macroparticles.
  (The energies of the electrons ejected when an ion is ionized several times by the same electron in one collision time
  step are all sampled with the energy of the electron at the first event, which slightly overestimates them: in tests with
  helium, the energy created was less than 1% of the energy lost by the electrons for :math:`R\Delta t` up to 2, where
  :math:`R` is the ionization rate of the ions.)
  The number of macroparticles increases by one for each ionization of an ion macroparticle.
  This method is exact for any collision time step for a given pair of macroparticles, e.g. for a monoenergetic electron beam
  with as many electron as ion macroparticles.
  However, each ion is paired with one electron macroparticle (or a few, when there are more electron than ion
  macroparticles in the cell) per collision time step, and the velocity of this electron is used for the whole time step.
  The probability of ionization of an ion in a collision time step is then the average of :math:`1 - \exp(-x)` over its
  partners, rather than :math:`1 - \exp(-R\Delta t)`, which underestimates the ionization rate when :math:`R\Delta t` is not
  small and :math:`x` varies between the partners: for a thermal electron distribution, the relative error is approximately
  :math:`-(1 + C^2) R\Delta t/2`, where :math:`C` is the relative standard deviation of :math:`\sigma v` over the electron
  distribution (e.g. -5% at :math:`R\Delta t = 0.05` and -50% at :math:`R\Delta t = 1` for C\ :sup:`0` at
  :math:`T_e = I_0`, where most electrons cannot ionize, and -1% at :math:`R\Delta t = 0.2` for :math:`T_e = 10 I_0`).
  When there are :math:`r` times more electron than ion macroparticles in the cells, each ion has :math:`r` partners per time
  step, which improves the averaging when :math:`r` is an integer, but a non-integer :math:`r` gives some ions one more
  partner than the others, which also underestimates the ionization rate (at :math:`R\Delta t = 1` and
  :math:`T_e = 10 I_0`, the error is -8% for :math:`r = 1.5`, instead of -4% for :math:`r = 1` and less than 1% for
  :math:`r = 3`). Such ratios also arise when the ejected electrons are added to the electron species.
  The results should therefore be checked for convergence with respect to the collision time step when the probability
  of ionization per collision time step is not small.

* If the macroparticles are split, an ionization event occurs with the probability :math:`P = 1 - \exp(-x)`.
  In that case, the weight :math:`w_{\min}` is subtracted from both colliding macroparticles, and three macroparticles
  of weight :math:`w_{\min}` are created: the scattered electron, at the position of the incident electron and in
  the same species, the ion with its ionization level increased by one, in the same species, and the ejected electron,
  in the product species. The ion and the ejected electron are created at the position of the ion,
  so that the charge is conserved locally, and the energy and the momentum are conserved exactly in each event.
  When the electron and ion macroparticles have different weights, the number of macroparticles increases by up to
  three for each ionization event, and particle resampling may be needed in long simulations.
  Each pair undergoes at most one ionization event per collision time step, and the accuracy of this method therefore
  requires :math:`x \ll 1` for all pairs (note that :math:`x` includes the ratio :math:`w_{\max}/w_{\min}`,
  and is therefore larger than the probability of ionization of the ions per time step when the weights differ:
  for instance, for a dilute electron beam in a dense gas, :math:`x` is the probability of ionization per time step of
  a beam electron, i.e. :math:`n_{gas}\sigma v \Delta t`).

If needed, the collision time step can be reduced with :pp:param:`<collision_name>.ndt_subcycle`.

When the ejected electrons are added to an electron species that undergoes Coulomb collisions (``pairwisecoulomb``),
and their weight (that of the ion macroparticles) differs from the weight of the other electrons, it is recommended to set
:pp:param:`collisions.correct_energy_momentum`, since the Coulomb collisions between macroparticles of different weights
only conserve the energy on average (in a test of the ionization of carbon in a 50 eV plasma with a strong collisionality,
the electron temperature was otherwise in error by about 5%).

The kinematics of the ionization events are computed in the rest frame of the ion.
The ionized subshell is selected with a probability proportional to its contribution to the cross section,
and the incident electron loses the binding energy :math:`B_s` of this subshell plus the kinetic energy
:math:`W` of the ejected electron.
With the default ``differential_cross_section`` model (see :pp:param:`<collision_name>.ejected_electron_energy_model`),
:math:`W` is sampled from the singly differential cross section of the RBEB model :cite:p:`b-KimPRA2000`:

    .. math::

        \frac{d\sigma}{dw} \propto \frac{1}{(w+1)^2} + \frac{1}{(t-w)^2}
        - \frac{1+2t'}{(1+t'/2)^2}\frac{1}{t+1}\left(\frac{1}{w+1} + \frac{1}{t-w}\right)
        + \frac{b'^2}{(1+t'/2)^2} + A_3\left(\frac{1}{(w+1)^3} + \frac{1}{(t-w)^3}\right)

with :math:`w = W/B_s` and :math:`0 \leq w \leq (t-1)/2` (by convention, the ejected electron is the slower of the two
outgoing electrons). The first terms describe close (binary) collisions, in which the energy is transferred to a
quasi-free electron, and the last term distant (dipole) collisions, which mostly produce slow electrons.
For a close collision, the ejected electron is emitted at the angle given by the kinematics of a collision with a free
electron at rest, :math:`\cos^2\theta = W(E + 2m_ec^2)/\left(E(W + 2m_ec^2)\right)`, with respect to the direction of the
incident electron, and the scattered electron is deflected such that the momentum of the two electrons is conserved.
For a distant collision, the ejected electron is emitted isotropically and the scattered electron keeps its direction.
With the ``at_rest`` model, :math:`W = 0`.
In all cases, the ion recoils such that the momentum is conserved, and the recoil energy of the ion is taken from the
scattered electron, such that the total kinetic energy decreases exactly by :math:`B_s`.

Note that, as for field ionization, the mass of the ion species does not change when the ions are ionized.
As a consequence, in the lab frame, the ejected electron also carries the kinetic energy and the momentum that it had
when it was bound to the moving ion (and which are not removed from the ion).
These are smaller than the kinetic energy and the momentum of the ion by a factor of order :math:`m_e/m_{ion}`.
Finally, the relaxation of the ions after the ionization of an inner subshell (e.g. by Auger emission, which can
ionize the ion further) is not modeled, and neither are recombination, excitation-autoionization and the lowering of the
ionization energies in dense plasmas.

.. _multiphysics-collisions-ion-impact-ionization:

Ion-impact ionization
---------------------

The ``ion_impact_ionization`` collision type models the ionization of atoms, ions and molecules by collisions with
fast ions, e.g. :math:`p + A^{q+} \rightarrow p + e^- + A^{(q+1)+}`, for instance when a beam of laser-accelerated
protons travels through a gas or a plasma.
The projectile can be any positively charged species without ionization level (protons, alpha particles,
fully stripped ions), of charge :math:`Z_p e` and mass :math:`M`.
The target can be:

* atomic, i.e. an ion species with field ionization enabled, whose subshells are obtained as for electron-impact
  ionization (see above). All the elements and charge states are supported, and successive ionizations are modeled.
* molecular, i.e. a species without ionization level (e.g. :math:`\mathrm{N_2}`), which is ionized once into
  another species (e.g. :math:`\mathrm{N_2^+}`). The binding energies and the occupations of its molecular orbitals
  are given by the user (see :pp:param:`<collision_name>.binding_energies`), and its total ionization cross section
  can be read from a file (see :pp:param:`<collision_name>.cross_section`).

Model
^^^^^

The singly differential cross section of a subshell (or molecular orbital) with :math:`N` electrons of binding energy
:math:`B`, differential in the kinetic energy :math:`W` of the ejected electron, is given by the semi-empirical model of
:cite:t:`b-RuddRMP1992` (their Eqs. 37-48), which represents the experimental data of proton collisions with atoms and
molecules from 5 keV to 5 MeV, and reproduces the asymptotic behavior of the Bethe theory at high energy:

    .. math::

        \frac{d\sigma}{dW} = Z_p^2 \frac{S}{B}\, \frac{F_1 + F_2 w}{(1+w)^3}\, \frac{1}{1 + \exp\left[\alpha (w - w_c)/v\right]}

where :math:`w = W/B`, :math:`S = 4\pi a_0^2 N (R/B)^2` (with :math:`a_0` the Bohr radius and :math:`R` the Rydberg energy),
:math:`v = \sqrt{T/B}` with :math:`T = m_e v_p^2/2` the kinetic energy of an electron with the velocity :math:`v_p` of the
projectile, :math:`w_c = W_{\mathrm{free}}/B - 2v - R/(4B)` is the cutoff energy, with :math:`W_{\mathrm{free}}` the maximum
energy transferred to a free electron at rest, and

    .. math::

        F_1 = L_1 + H_1, \quad F_2 = \frac{L_2 H_2}{L_2 + H_2}, \quad
        L_1 = \frac{C_1 v^{D_1}}{1 + E_1 v^{D_1 + 4}}, \quad H_1 = \frac{A_1 \ln(1+v^2)}{v^2 + B_1/v^2}, \quad
        L_2 = C_2 v^{D_2}, \quad H_2 = \frac{A_2}{v^2} + \frac{B_2}{v^4}

The ten parameters :math:`A_1, \dots, D_2, \alpha` were determined by :cite:t:`b-RuddRMP1992` for the outer subshells of
He, Ne, Ar, Kr and of several molecules, and a universal set of parameters was determined for the inner subshells, i.e. the
subshells whose binding energy exceeds twice that of the least bound subshell.
For an atomic target, the outer subshells of a charge state with :math:`N_e` bound electrons use by default
(see :pp:param:`<collision_name>.rudd_parameters`):

* for one-electron targets (H and hydrogen-like ions), for which no parameters were determined, the parameters of He, except
  :math:`A_1 = 0.75` and :math:`A_2 = 1.2`.
  At high velocity, the cross section is :math:`\sigma \approx (S/2v^2)\left[A_1 \ln(1+v^2) + A_2\right]`, and the cross
  section for fast electrons (:math:`w \gg 1`) is :math:`A_2` times the Rutherford cross section.
  :math:`A_2` was chosen close to 1, as for the targets of :cite:t:`b-RuddRMP1992`, and :math:`A_1` was then adjusted such
  that the cross section of hydrogen agrees within 7% with the values recommended by :cite:t:`b-RuddRMP1985` from 50 keV
  to 100 MeV (above 1.5 MeV, the highest energy of the recommended values, their fit formula follows the Bethe theory).
  The values of the Bethe theory, :math:`A_1 = 2M^2 = 0.567` (where :math:`M^2 = 0.2834` is the dipole matrix element for the
  ionization of hydrogen :cite:p:`b-InokutiRMP1971`) and :math:`A_2 \approx 2`, would reproduce the total cross section,
  but with twice the Rutherford cross section for fast electrons, and an energy lost in ionization that exceeds the
  stopping power.
  Since the binding energy of hydrogen-like ions scales as :math:`Z^2`, their cross sections follow the scaling of the
  Born approximation.
* for the closed-shell configurations (:math:`N_e = 2, 10, 18, 36`), the parameters of the corresponding rare gas.
* otherwise, the parameters of He for :math:`N_e < 10`, of Ar for :math:`10 < N_e < 18`, and of Kr for :math:`N_e > 18`.
  (The parameters of Ne have a small dipole parameter :math:`A_1`, which was chosen as a compromise for the fit of the
  data of neon, and would give for the atoms N and O a dipole matrix element smaller than half that of
  :math:`\mathrm{N_2}` and :math:`\mathrm{O_2}`.)

For a molecular target, the outer orbitals use the parameters of He by default.
The parameters of :cite:t:`b-RuddRMP1992` for some common molecules are given in the table below (Tables I and V of
:cite:t:`b-RuddRMP1992`), and can be used with :pp:param:`<collision_name>.binding_energies`,
:pp:param:`<collision_name>.occupations` and :pp:param:`<collision_name>.rudd_parameters`.

.. list-table:: Molecular orbitals and parameters of the Rudd model :cite:p:`b-RuddRMP1992`
   :header-rows: 1

   * - Molecule
     - Binding energies [eV]
     - Occupations
     - Parameters :math:`A_1, B_1, C_1, D_1, E_1, A_2, B_2, C_2, D_2, \alpha`
   * - :math:`\mathrm{H_2}`
     - 15.43
     - 2
     - 0.96 2.6 0.38 0.23 2.2 1.04 5.9 1.15 0.20 0.87
   * - :math:`\mathrm{N_2}`
     - 15.59 16.96 18.78 37.3 409.9
     - 2 4 2 2 4
     - 1.05 12.0 0.74 -0.39 0.80 0.95 1.20 1.00 1.30 0.70
   * - :math:`\mathrm{O_2}`
     - 12.07 16.42 18.88 25.69 40.3 543.5
     - 2 2 4 2 2 4
     - 1.02 50 0.40 0.12 0.30 1.00 5.0 0.55 0 0.59
   * - :math:`\mathrm{H_2O}`
     - 12.61 14.73 18.55 32.2 539.7
     - 2 2 2 2 2
     - 0.97 82 0.40 -0.30 0.38 1.04 17.3 0.76 0.04 0.64
   * - :math:`\mathrm{CO_2}`
     - 13.79 17.60 18.08 19.4 37.0 38.6 297.5 541.0
     - 4 4 2 2 2 2 2 4
     - 1.09 25 0.75 0.75 0.65 0.78 3.0 0.70 0.85 0.53
   * - :math:`\mathrm{CH_4}`
     - 14.35 23 290.7
     - 6 2 2
     - 1.15 14 0.35 0.50 3.0 0.60 3.8 1.20 0.45 0.61

The model is extended to relativistic projectiles in two ways: :math:`W_{\mathrm{free}}` is computed relativistically,
:math:`W_{\mathrm{free}} = 2 m_e c^2 \beta^2\gamma^2/\left(1 + 2\gamma m_e/M + (m_e/M)^2\right)`
(which is equal to :math:`4T` for slow projectiles), and the relativistic correction of the Bethe theory,
:math:`\ln\gamma^2 - \beta^2` (see Eq. 22a of :cite:t:`b-RuddRMP1992`), is added to :math:`\ln(1+v^2)` in :math:`H_1`,
which gives the relativistic rise of the cross sections (this correction is smaller than 1% below 100 MeV per nucleon).
The factor :math:`Z_p^2` corresponds to the first Born approximation, which is accurate when the projectile is much faster
than the bound electrons and :math:`Z_p` is small compared with :math:`v_p/v_0`, where :math:`v_0` is the Bohr velocity.
The model does not include the capture of target electrons by the projectile, which dominates at low velocity but is
negligible above about 0.5 MeV per nucleon, and, as for electron-impact ionization, the relaxation of the target after
the ionization of an inner subshell and multiple ionization in a single collision are not modeled.
The model is therefore intended for projectiles faster than about 0.5 MeV per nucleon.

The cross section of each subshell, i.e. the integral of the differential cross section over :math:`0 \leq W \leq E - B`
(where :math:`E` is the kinetic energy of the projectile in the rest frame of the target), is computed at initialization by
numerical quadrature and tabulated as a function of :math:`\gamma\beta` of the projectile, from :math:`10^{-4}` to
:math:`10^4` with 64 points per decade (the relative error of the interpolation is smaller than :math:`10^{-3}`, except
for projectiles much slower than the bound electrons, i.e. :math:`T \ll B`, for which the cross sections are very small).
For a molecular target with a total cross section read from a file, the cross sections of the model are only used to
determine the relative contributions of the orbitals.

With the binding energies of WarpX, the total cross sections of He, Ne, Ar and Kr agree within 7% with the values
recommended by :cite:t:`b-RuddRMP1985` from 0.3 to 5 MeV, and that of H within 6% (and within 11% up to 1 GeV, compared
with the Bethe form of their fit).
The uncertainty is larger for the targets for which no parameters were determined: with the parameters of Kr, the cross
section of Xe is 7% to 13% lower than the recommended values, and the cross sections of the atoms C, N and O are larger
than half those of the molecules :math:`\mathrm{N_2}`, :math:`\mathrm{O_2}`, CO and :math:`\mathrm{CO_2}` by 10% to 50%
(as expected for free atoms, whose valence electrons are less bound than in molecules, but this cannot be checked against
measurements).
With the parameters of the table above, the cross sections of :math:`\mathrm{H_2}` and :math:`\mathrm{N_2}` agree within 5%
with the recommended values from 0.3 to 5 MeV, those of :math:`\mathrm{O_2}` and :math:`\mathrm{CO_2}` within 13%, and that of
:math:`\mathrm{CH_4}` is 13% too low (for this molecule, the cross section should be read from a file).
The energy lost by the projectiles in the ionization events (binding energy plus kinetic energy of the ejected electrons)
represents 80% to 100% of the electronic stopping power of the Bethe theory for protons from 1 to 100 MeV in H,
:math:`\mathrm{H_2}`, He, :math:`\mathrm{N_2}`, :math:`\mathrm{O_2}`, :math:`\mathrm{H_2O}` and Ar.
The excitation of discrete levels, which is not modeled, accounts for the remainder of the stopping power (10% to 20%
according to Sec. V.F of :cite:t:`b-RuddRMP1992`), so that the values close to 100% obtained at high energy with the
parameters of :math:`\mathrm{O_2}`, :math:`\mathrm{H_2O}` and Ar indicate that the energy of the ejected electrons is
overestimated by 10% to 20% in these cases.

Algorithm
^^^^^^^^^

The pairing of the macroparticles, the computation of the probabilities and the two methods (with or without splitting
the macroparticles) are the same as for electron-impact ionization (see above), with the ion projectiles in place of the
electrons and the targets in place of the ions.
When the macroparticles are not split and the target is molecular, the ionized target macroparticle is removed and replaced
by a macroparticle of the product ion species, of the same weight, at the same position and with the same momentum
(including its recoil); a molecular target can only be ionized once per collision time step.

The kinematics of the ionization events are computed in the rest frame of the target.
The ionized subshell is selected with a probability proportional to its contribution to the cross section,
and the projectile loses the binding energy :math:`B` of this subshell plus the kinetic energy :math:`W` of the ejected
electron, which is sampled from the differential cross section above by composition and rejection.
Electrons sampled from the term :math:`F_2 w/(1+w)^3`, which contains the free-electron (Rutherford) limit of the cross
section and dominates for fast electrons, are considered to result from close (binary) collisions, and are emitted at the
angle given by the kinematics of a collision with a free electron at rest,

    .. math::

        \cos^2\theta = \frac{W\left(E + Mc^2 + m_ec^2\right)^2}{E\left(E + 2Mc^2\right)\left(W + 2m_ec^2\right)}

with respect to the direction of the projectile (for slow projectiles, :math:`\cos^2\theta \approx W/W_{\mathrm{free}}`,
i.e. the fast electrons are emitted in the forward direction), and the projectile is deflected such that the momentum of
the projectile and of the ejected electron is conserved.
The other electrons, which mostly result from distant (dipole) collisions and are slow, are emitted isotropically, and the
projectile keeps its direction.
In all cases, the target recoils such that the momentum is conserved, and its recoil energy is taken from the projectile,
such that the total kinetic energy decreases exactly by :math:`B`.

.. bibliography::
    :keyprefix: b-
