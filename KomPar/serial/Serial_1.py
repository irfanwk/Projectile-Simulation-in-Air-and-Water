import vpython as vp
import numpy as np
import time

def acceleration(x, v, params):
    mass = params['mass']
    g = params['g']
    Cd = params['Cd']
    area = params['area']
    volume = params['volume']
    v_mag = np.linalg.norm(v)
    
    if x[1] < 0:  # di dalam air
        rho = params['water_density']
        F_grav  = np.array([0, -mass*g, 0])
        F_buoy  = np.array([0, rho*volume*g, 0])
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

def rk4_step(x, v, dt, params):
    k1_x = dt * v
    k1_v = dt * acceleration(x, v, params)
    k2_x = dt * (v + 0.5*k1_v)
    k2_v = dt * acceleration(x + 0.5*k1_x, v + 0.5*k1_v, params)
    k3_x = dt * (v + 0.5*k2_v)
    k3_v = dt * acceleration(x + 0.5*k2_x, v + 0.5*k2_v, params)
    k4_x = dt * (v + k3_v)
    k4_v = dt * acceleration(x + k3_x, v + k3_v, params)

    x_next = x + (k1_x + 2*k2_x + 2*k3_x + k4_x)/6.0
    v_next = v + (k1_v + 2*k2_v + 2*k3_v + k4_v)/6.0
    return x_next, v_next

# catat waktu mulai
start_time = time.perf_counter()
print(f"Waktu mulai program: {start_time:.8f} detik")

# parameter global
g               = 9.8
water_density   = 1000
air_density     = 1.225
Cd              = 0.47
r               = 0.5
volume          = (4/3)*np.pi*r**3
area            = np.pi*r**2
dt_global       = 0.001

# parameter arus air
current_force_magnitude = 3000
current_azim_deg        = 45 # sudut arah arus di bidang x-z
azim_rad                = np.radians(current_azim_deg)
current_dir             = np.array([np.cos(azim_rad), 0.0, np.sin(azim_rad)])
current_force           = current_force_magnitude * current_dir

#=============================
# testing simulation
t_max           = 7.0
number_of_proj  = 100
#=============================


# scene dan permukaan air
direction_arrow_scale = 5
scene = vp.canvas(title="Simulasi " + str(number_of_proj) + " Proyektil", width=1920, height=1080, background=vp.color.white)
water_box = vp.box(pos=vp.vector(0, -50, 0), size=vp.vector(100, 100, 100), color=vp.color.blue, opacity=0.2)
water_plane = vp.box(pos=vp.vector(0, 0, 0), size=vp.vector(100, 0.2, 100), color=vp.color.blue, opacity=0.4)

# panah arus air
arrow_spacing = 15
for x in np.arange(-40, 50, arrow_spacing):
    for z in np.arange(-40, 50, arrow_spacing):
        vp.arrow(pos=vp.vector(x, 0.5, z), axis=vp.vector(*current_dir) * direction_arrow_scale,
                 shaftwidth=0.3, color=vp.color.cyan)

# inisialisasi parameter proyektil
projectiles = []
for i in range(number_of_proj):
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
        'water_density': water_density,
        'air_density': air_density,
        'volume': volume,
        'area': area,
        'Cd': Cd,
        'current_force': current_force
    }
    projectiles.append({'params': params, 'v': v0})

# buat visual objek bola
projs = []
for p in projectiles:
    proj = vp.sphere(pos=vp.vector(0,0,0), 
                     radius=r,
                     color=vp.vector(np.random.rand(), np.random.rand(), np.random.rand()),
                     make_trail=True)
    projs.append({'obj': proj, 'x': np.array([0.0,0.0,0.0]), 'v': p['v'], 'params': p['params']})

# MAIN LOOP
t_global = 0.0
while t_global < t_max:
    vp.rate(1000)
    for p in projs:
        x_new, v_new = rk4_step(p['x'], p['v'], dt_global, p['params'])
        p['x'], p['v'] = x_new, v_new
        p['obj'].pos = vp.vector(*x_new)
    t_global += dt_global

# catat waktu selesai
end_time = time.perf_counter()
print(f"Waktu selesai program: {end_time:.8f} detik")
total_time = end_time - start_time
print(f"Total waktu eksekusi: {total_time:.8f} detik")