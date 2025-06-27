import vpython as vp
import numpy as np
import time

# fungsi simulasi SEBUAH proyektil supaya dapat dipanggil secara paralel
def simulate_projectile(args):
    mass, v0, params, dt, t_max = args
    def acceleration(x, v):
        g   = params['g']
        Cd  = params['Cd']
        area= params['area']
        vol = params['volume']
        v_mag = np.linalg.norm(v)

        if x[1] < 0:  # di dalam air
            rho = params['water_density']
            F_grav  = np.array([0, -mass*g, 0])
            F_buoy  = np.array([0, rho*vol*g, 0])
            if v_mag > 0:
                F_drag  = -0.5 * rho * Cd * area * v_mag * v
            else:
                F_drag  = np.zeros(3)
            F_curr  = params['current_force']
            F_net   = F_grav + F_buoy + F_drag + F_curr
        else:        # di udara
            rho     = params['air_density']
            F_grav  = np.array([0, -mass*g, 0])
            if v_mag > 0:
                F_drag  = -0.5 * rho * Cd * area * v_mag * v
            else:
                F_drag  = np.zeros(3)
            F_net   = F_grav + F_drag
        
        return F_net / mass

    def rk4_step(x, v):
        k1_x = dt * v
        k1_v = dt * acceleration(x, v)
        k2_x = dt * (v + 0.5*k1_v)
        k2_v = dt * acceleration(x + 0.5*k1_x, v + 0.5*k1_v)
        k3_x = dt * (v + 0.5*k2_v)
        k3_v = dt * acceleration(x + 0.5*k2_x, v + 0.5*k2_v)
        k4_x = dt * (v + k3_v)
        k4_v = dt * acceleration(x + k3_x, v + k3_v)

        x_next = x + (k1_x + 2*k2_x + 2*k3_x + k4_x)/6.0
        v_next = v + (k1_v + 2*k2_v + 2*k3_v + k4_v)/6.0
        return x_next, v_next

    # inisialisasi
    steps = int(t_max / dt) + 1
    traj = np.zeros((steps, 3), dtype=float)
    x = np.zeros(3)
    v = v0.copy()
    for i in range(steps):
        traj[i] = x
        x, v = rk4_step(x, v)
    return traj

# MAIN PROGRAM
# parameter global
g               = 9.8
water_density   = 1000
air_density     = 1.225
Cd              = 0.47
r               = 0.1
volume          = (4/3)*np.pi*r**3
area            = np.pi*r**2
dt              = 0.001

# parameter arus air
current_force_magnitude = 5000
current_azim_deg = 45  # sudut arah arus di bidang x-z
azim_rad = np.radians(current_azim_deg)
# vektor arah arus
current_dir = np.array([np.cos(azim_rad), 0.0, np.sin(azim_rad)])
current_force = current_force_magnitude * current_dir

# inisialisasi parameter proyektil
projectiles = []
for i in range(100):
    mass        = 50 + i*50
    elev_deg    = 10 + i*5
    azim_deg    = i*30
    power       = 2000 + i*400
    elev_rad    = np.radians(elev_deg)
    azim_rad2   = np.radians(azim_deg)
    v0_mag      = power / mass
    v0 = np.array([
        v0_mag * np.cos(elev_rad) * np.cos(azim_rad2),
        v0_mag * np.sin(elev_rad),
        v0_mag * np.cos(elev_rad) * np.sin(azim_rad2)
    ])
    params = {
        'mass': mass,
        'g': g,
        'Cd': Cd,
        'area': area,
        'volume': volume,
        'water_density': water_density,
        'air_density': air_density,
        'current_force': current_force
    }
    projectiles.append((mass, v0, params, dt, 10.0))  # 10.0 adalah waktu maksimum simulasi

# jalankan simulasi secara serial
trajs = []
for projectile in projectiles:
    traj = simulate_projectile(projectile)
    trajs.append(traj)

# visualisasikan hasil simulasi dengan vpython
vp.scene = vp.canvas(title='Simulasi Proyektil', width=800, height=600)
vp.rate(60)

for i, traj in enumerate(trajs):
    vp.sphere(pos=vp.vector(traj[0, 0], traj[0, 1], traj[0, 2]), radius=0.1, color=vp.color.red)
    for j in range(1, len(traj)):
        vp.sphere(pos=vp.vector(traj[j, 0], traj[j, 1], traj[j, 2]), radius=0.1, color=vp.color.red)
        vp.line(pos=vp.vector(traj[j-1, 0], traj[j-1, 1], traj[j-1, 2]), other=vp.vector(traj[j, 0], traj[j, 1], traj[j, 2]), color=vp.color.red)

vp.scene.autoscale = False
vp.scene.range = (10, 10, 10)
vp.scene.center = vp.vector(0, 0, 0)

while True:
    vp.rate(60)
    for i, traj in enumerate(trajs):
        for j in range(1, len(traj)):
            vp.line(pos=vp.vector(traj[j-1, 0], traj[j-1, 1], traj[j-1, 2]), other=vp.vector(traj[j, 0], traj[j, 1], traj[j, 2]), color=vp.color.red)
            vp.sphere(pos=vp.vector(traj[j, 0], traj[j, 1], traj[j, 2]), radius=0.1, color=vp.color.red)
    vp.scene.render()

    # tambahkan efek gravitasi
    for i, traj in enumerate(trajs):
        for j in range(1, len(traj)):
            vp.line(pos=vp.vector(traj[j-1, 0], traj[j-1, 1], traj[j-1, 2]), other=vp.vector(traj[j, 0], traj[j, 1], traj[j, 2]), color=vp.color.red)
            vp.sphere(pos=vp.vector(traj[j, 0], traj[j, 1], traj[j, 2]), radius=0.1, color=vp.color.red)
            # tambahkan efek gravitasi
            traj[j, 1] -= 0.1

    # tambahkan efek gesekan
    for i, traj in enumerate(trajs):
        for j in range(1, len(traj)):
            vp.line(pos=vp.vector(traj[j-1, 0], traj[j-1, 1], traj[j-1, 2]), other=vp.vector(traj[j, 0], traj[j, 1], traj[j, 2]), color=vp.color.red)
            vp.sphere(pos=vp.vector(traj[j, 0], traj[j, 1], traj[j, 2]), radius=0.1, color=vp.color.red)
            # tambahkan efek gesekan
            traj[j, 0] -= 0.01
            traj[j, 2] -= 0.01

vp.scene.delete()