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
the Lotz model is accurate for ions but overestimates the ionization of neutral atoms (typically by a factor up to 2),
and the RBEB model with the approximations above underestimates the ionization of ions.

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
  The charge is thus conserved exactly, and the energy and the momentum are conserved on average.
  The number of macroparticles increases by one for each ionization of an ion macroparticle.
  Since the successive events of a pair involve the same electron macroparticle, the probability that an ion is ionized several
  times within one collision time step is computed with the energy of this electron, rather than with the average over the
  electron distribution, which overestimates these successive ionizations for broad electron distributions when the
  ionization rates depend strongly on the electron energy (for instance near the ionization thresholds).
  (This effect is reduced when there are more electron macroparticles than ion macroparticles in the cells,
  since each ion is then paired with several electrons in each collision time step.)
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
  and is therefore larger than the probability of ionization of the ions per time step when the weights differ).

If needed, the collision time step can be reduced with :pp:param:`<collision_name>.ndt_subcycle`.

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

.. bibliography::
    :keyprefix: b-
