from multiprocessing import Process, Queue
import vpython as vp
import numpy as np

def simulate_projectile(idx, mass, v0, params, dt, t_max, out_queue):
    """
    Hitung lintasan penuh satu proyektil dan kirim (idx, traj) ke out_queue.
    traj: numpy array shape (n_steps, 3)
    """
    def acceleration(x, v):
        g   = params['g']
        Cd  = params['Cd']
        area= params['area']
        vol = params['volume']
        v_mag = np.linalg.norm(v)
        if x[1] < 0:
            rho    = params['water_density']
            F_grav = np.array([0, -mass*g, 0])
            F_buoy = np.array([0, rho*vol*g, 0])
            F_drag = -0.5 * rho * Cd * area * v_mag * v if v_mag>0 else np.zeros(3)
            F_curr = params['current_force']
            F_net  = F_grav + F_buoy + F_drag + F_curr
        else:
            rho    = params['air_density']
            F_grav = np.array([0, -mass*g, 0])
            F_drag = -0.5 * rho * Cd * area * v_mag * v if v_mag>0 else np.zeros(3)
            F_net  = F_grav + F_drag
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

    # Hitung lintasan
    steps = int(t_max / dt) + 1
    traj = np.zeros((steps, 3), dtype=float)
    x = np.zeros(3)
    v = v0.copy()
    for i in range(steps):
        traj[i] = x
        x, v = rk4_step(x, v)

    # Kirim hasil
    out_queue.put((idx, traj))


if __name__ == '__main__':
    # — Parameter global —
    g             = 9.8
    water_density = 1000
    air_density   = 1.225
    Cd            = 0.47
    r             = 0.1
    volume        = (4/3)*np.pi*r**3
    area          = np.pi*r**2
    dt            = 0.001
    t_max         = 10.0
    n_proj        = 500

    # Queue untuk menerima hasil
    result_queue = Queue()

    # Buat dan start process untuk tiap proyektil
    processes = []
    for i in range(n_proj):
        mass     = 100 + i*50
        elev_rad = np.radians(20 + i*5)
        azim_rad = np.radians(i*36)
        power    = 2000 + i*400
        v0_mag   = power / mass
        v0 = np.array([
            v0_mag * np.cos(elev_rad) * np.cos(azim_rad),
            v0_mag * np.sin(elev_rad),
            v0_mag * np.cos(elev_rad) * np.sin(azim_rad)
        ])
        params = {
            'g'             : g,
            'water_density' : water_density,
            'air_density'   : air_density,
            'volume'        : volume,
            'area'          : area,
            'Cd'            : Cd,
            'current_force' : np.zeros(3)
        }
        p = Process(target=simulate_projectile,
                    args=(i, mass, v0, params, dt, t_max, result_queue))
        processes.append(p)
        p.start()

    # Tunggu semua selesai
    for p in processes:
        p.join()

    # Kumpulkan hasil
    all_traj = [None] * n_proj
    while not result_queue.empty():
        idx, traj = result_queue.get()
        all_traj[idx] = traj

    # — Visualisasi dengan VPython —
    scene = vp.canvas(title="Simulasi 500 Proyektil (Process)", width=1280, height=720, background=vp.color.white)
    vp.box(pos=vp.vector(0, -50, 0), size=vp.vector(100, 100, 100), color=vp.color.blue, opacity=0.2)
    vp.box(pos=vp.vector(0, 0, 0),    size=vp.vector(100, 0.2, 100), color=vp.color.blue, opacity=0.5)

    # Buat sphere dan tautkan dengan traj
    spheres = []
    for traj in all_traj:
        color = vp.vector(np.random.rand(), np.random.rand(), np.random.rand())
        sph   = vp.sphere(pos=vp.vector(*traj[0]), radius=r, color=color, make_trail=True)
        spheres.append((sph, traj))

    n_steps = all_traj[0].shape[0]
    for step in range(n_steps):
        vp.rate(1/dt)
        for sph, traj in spheres:
            sph.pos = vp.vector(*traj[step])

    print("Simulasi selesai.")
