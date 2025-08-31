import os
import glob
import numpy as np
import dedalus.public as d3
import logging

logger = logging.getLogger(__name__)

# ----------------------------------

Lx, Lz = 12*np.pi, 6*np.pi
Nx, Nz = 1024, 512
beta = 1
nv = 0.0002
dealias = 3/2
stop_sim_time = 700
timestepper = d3.RK443
max_timestep = 1e-2
dtype = np.float64

coords = d3.CartesianCoordinates('x', 'z')
dist   = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
zbasis = d3.RealFourier(coords['z'], size=Nz, bounds=(-Lz/2, Lz/2), dealias=dealias)

tau_psi   = dist.Field(name='tau_psi')
cos_z     = dist.Field(name='cos_z',     bases=(xbasis, zbasis))
sin_z     = dist.Field(name='sin_z',     bases=(xbasis, zbasis))
psi_prime = dist.Field(name='psi_prime', bases=(xbasis, zbasis))

x, z = dist.local_grids(xbasis, zbasis)
ex, ez = coords.unit_vector_fields(dist)

eps = 2*np.pi / Lz

Re = 1 / nv

cos_z['g'] = np.cos(z)
sin_z['g'] = np.sin(z)

#ABC设定
A=1
B=1
C=1
omega_q=1

# ----------------------------------

psi_bar         = lambda t:  (A+B*np.cos(t) + C*np.cos(omega_q*t)) * cos_z
lap_psi_bar     = lambda t: -(A+B*np.cos(t) + C*np.cos(omega_q*t)) * cos_z
dpsi_bar_dz     = lambda t: -(A+B*np.cos(t) + C*np.cos(omega_q*t)) * sin_z
dlap_psi_bar_dz = lambda t:  (A+B*np.cos(t) + C*np.cos(omega_q*t)) * sin_z

# ----------------------------------

dx = lambda f: d3.Differentiate(f, coords['x'])
dz = lambda f: d3.Differentiate(f, coords['z'])

def jacobian(f1, f2):
    return dx(f1) * dz(f2) - dz(f1) * dx(f2)

def jacobian2(f3, t):  # jacobian(psi_prime, lap(psi_bar))
    return dx(f3) * dlap_psi_bar_dz(t)

def jacobian3(f4, t):  # jacobian(psi_bar, lap(psi_prime))
    return -dpsi_bar_dz(t) * dx(f4)

# ----------------------------------

problem = d3.IVP([psi_prime, tau_psi], namespace=locals())
problem.namespace.update({'t': problem.time})

problem.add_equation(
    "d3.TimeDerivative(lap(psi_prime)) - nv*lap(lap(psi_prime)) + beta*dx(psi_prime) + tau_psi "
    "= -jacobian2(psi_prime, t) - jacobian3(lap(psi_prime), t) - jacobian(psi_prime, lap(psi_prime))"
)
problem.add_equation("integ(psi_prime) = 0")

solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# ----------------------------------

#RESTART      = os.environ.get("RESTART", "0") == "1"
RESTART = True
RESTART_PATH = "/ocean/projects/phy240052p/zsong7/Rayleigh/quasiN/sqrt3/checkpoints/checkpoints_s1.h5"   #这里不是自动的，重启时要手动指定路径

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("checkpoints_early", exist_ok=True)

if RESTART:

    if RESTART_PATH and os.path.isfile(RESTART_PATH):
        load_file = RESTART_PATH

    else:
        cands = sorted(glob.glob("checkpoints/checkpoints_s*.h5"))
        if not cands:
            cands = sorted(glob.glob("checkpoints_early/checkpoints_early_s*.h5"))
        if not cands:
            raise FileNotFoundError("No checkpoint files found; run once without RESTART.")
        load_file = cands[-1]
    solver.load_state(load_file, index=-1)
    logger.info(f"[Restart] Loaded checkpoint: {load_file}")

else:
    psi_prime.fill_random('g', seed=42, distribution='normal', scale=1e-3)
    logger.info("[Init] Random initial condition set for psi_prime")

# ----------------------------------
# t = 0.5 记录一次（用于验证是否能成功生成checkpoints）之后每隔5记录一次

checkpoints_early = solver.evaluator.add_file_handler('checkpoints1', sim_dt=0.5, max_writes=1)
for i, f in enumerate(solver.state):
    name = f.name if getattr(f, "name", None) else f"state_{i}"
    checkpoints_early.add_task(f, name=name, layout='g')
    
checkpoints_regular = solver.evaluator.add_file_handler('checkpoints', sim_dt=5.0, max_writes=None)
for i, f in enumerate(solver.state):
    name = f.name if getattr(f, "name", None) else f"state_{i}"
    checkpoints_regular.add_task(f, name=name, layout='g')

# ----------------------------------

snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=0.1, max_writes=None)
snapshots.add_task(psi_prime,                         name='psi_prime',        layout='g')
snapshots.add_task(cos_z,                             name='cos_z',            layout='g')
snapshots.add_task(d3.Laplacian(psi_prime),           name='vorticity_prime',  layout='g')
snapshots.add_task(dx(psi_prime),                     name='w',                layout='g')
snapshots.add_task(-dz(psi_prime),                    name='u',                layout='g')
snapshots.add_task(dpsi_bar_dz(problem.time),         name='u_bar',            layout='g')
snapshots.add_task(lap_psi_bar(problem.time),         name='lap_psi_bar',      layout='g')
snapshots.add_task(dlap_psi_bar_dz(problem.time),     name='dlap_psi_bar_dz',  layout='g')
snapshots.add_task(beta - dlap_psi_bar_dz(problem.time), name='Q_rayleigh_kuo', layout='g')

snapshots_1D = solver.evaluator.add_file_handler('snapshots_1D', sim_dt=0.01, max_writes=None)
snapshots_1D.add_task(dx(psi_prime)(x=0), name='w_yz')
snapshots_1D.add_task(dx(psi_prime)(z=0), name='w_xy')
snapshots_1D.add_task(-dz(psi_prime)(x=0), name='u_yz')
snapshots_1D.add_task(-dz(psi_prime)(z=0), name='u_xy')
snapshots_1D.add_task(psi_prime(x=0), name='psi_yz')
snapshots_1D.add_task(psi_prime(z=0), name='psi_xy')

# ----------------------------------

velox = dpsi_bar_dz(problem.time) + dz(psi_prime)
veloz = dx(psi_prime)
velocity_expr = velox * ex + veloz * ez

CFL = d3.CFL(solver, initial_dt=max_timestep/5, cadence=10, safety=0.2, threshold=0.1,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(velocity_expr)

flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(psi_prime, name='psi_prime')

# -----------------Main loop-----------------

try:
    logger.info('Starting main loop')
    while solver.proceed:
        dt = CFL.compute_timestep()
        solver.step(dt)
        if (solver.iteration - 1) % 10 == 0:
            max_val = flow.max('psi_prime')
            safe_val = np.sqrt(max(max_val, 0.0))
            logger.info('Iteration=%i, Time=%e, dt=%e, max(psi_prime)=%e',
                        solver.iteration, solver.sim_time, dt, safe_val)
            if not np.isfinite(psi_prime['g']).all():
                raise RuntimeError(f"NaN detected in psi_prime at t={solver.sim_time}")

except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
