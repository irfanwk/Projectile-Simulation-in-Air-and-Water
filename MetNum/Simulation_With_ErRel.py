from multiprocessing import Pool
import vpython as vp
import numpy as np
import matplotlib.pyplot as plt # Import Matplotlib

# Fungsi simulasi SEBUAH proyektil supaya dapat dipanggil secara paralel
def simulate_projectile(args):
    """
    Menjalankan simulasi gerak proyektil menggunakan metode RK4 untuk satu set parameter.
    """
    mass, v0, params, dt, t_max = args
    def acceleration(x, v):
        """Menghitung vektor percepatan berdasarkan posisi dan kecepatan."""
        g = params['g']
        Cd = params['Cd']
        area = params['area']
        vol = params['volume']
        v_mag = np.linalg.norm(v)

        if x[1] < 0:  # di dalam air (y < 0)
            rho = params['water_density']
            F_grav = np.array([0, -mass * g, 0])
            F_buoy = np.array([0, rho * vol * g, 0])
            if v_mag > 0:
                F_drag = -0.5 * rho * Cd * area * v_mag * v
            else:
                F_drag = np.zeros(3)
            F_curr = params['current_force']
            F_net = F_grav + F_buoy + F_drag + F_curr
        else:  # di udara (y >= 0)
            rho = params['air_density']
            F_grav = np.array([0, -mass * g, 0])
            if v_mag > 0:
                F_drag = -0.5 * rho * Cd * area * v_mag * v
            else:
                F_drag = np.zeros(3)
            F_net = F_grav + F_drag
        
        return F_net / mass

    def rk4_step(x, v):
        """Satu langkah integrasi menggunakan metode Runge-Kutta orde 4."""
        k1_x = dt * v
        k1_v = dt * acceleration(x, v)
        k2_x = dt * (v + 0.5 * k1_v)
        k2_v = dt * acceleration(x + 0.5 * k1_x, v + 0.5 * k1_v)
        k3_x = dt * (v + 0.5 * k2_v)
        k3_v = dt * acceleration(x + 0.5 * k2_x, v + 0.5 * k2_v)
        k4_x = dt * (v + k3_v)
        k4_v = dt * acceleration(x + k3_x, v + k3_v)

        x_next = x + (k1_x + 2 * k2_x + 2 * k3_x + k4_x) / 6.0
        v_next = v + (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6.0
        return x_next, v_next

    # Inisialisasi
    steps = int(t_max / dt) + 1
    traj = np.zeros((steps, 3), dtype=float)
    x = np.zeros(3)
    v = v0.copy()
    for i in range(steps):
        traj[i] = x
        # Hentikan simulasi jika proyektil keluar dari area relevan
        if x[1] < -100 and i > 0:
            traj = traj[:i]  # Potong trajectory sampai titik ini
            break
        x, v = rk4_step(x, v)
    return traj


# MAIN PROGRAM
if __name__ == '__main__':
    # ==========================================================
    # Tahap 1: Konfigurasi Parameter dan Kalkulasi Lintasan
    # ==========================================================
    
    # Parameter global
    g = 9.8
    water_density = 1000
    air_density = 1.225
    Cd = 0.47
    r = 0.1
    volume = (4/3) * np.pi * r**3
    area = np.pi * r**2
    
    # Parameter arus air
    current_force_magnitude = 5000
    current_azim_deg = 45  # sudut arah arus di bidang x-z
    azim_rad_current = np.radians(current_azim_deg)
    current_dir = np.array([np.cos(azim_rad_current), 0.0, np.sin(azim_rad_current)])
    current_force = current_force_magnitude * current_dir
    
    # Parameter simulasi untuk perbandingan dt
    dt_values = [2, 1, 0.5, 0.001]
    t_max = 8.0 
    
    # Parameter proyektil tunggal untuk diuji
    mass = 100 
    elev_rad = np.radians(45)
    azim_rad_shot = np.radians(0)
    power = 5000
    v0_mag = power / mass
    v0 = np.array([
        v0_mag * np.cos(elev_rad) * np.cos(azim_rad_shot),
        v0_mag * np.sin(elev_rad),
        v0_mag * np.cos(elev_rad) * np.sin(azim_rad_shot)
    ])

    all_results_by_dt = {}

    # Lakukan simulasi untuk setiap nilai dt
    for dt_val in dt_values:
        print(f"Menjalankan simulasi dengan dt = {dt_val}...")
        
        params = {
            'g': g, 'water_density': water_density, 'air_density': air_density,
            'volume': volume, 'area': area, 'Cd': Cd, 'current_force': current_force
        }
        
        args_for_current_dt = [(mass, v0, params, dt_val, t_max)]

        with Pool() as pool:
            current_dt_trajectory = pool.map(simulate_projectile, args_for_current_dt)[0] 
            all_results_by_dt[dt_val] = current_dt_trajectory
            print(f"Simulasi dt={dt_val} selesai dengan {len(current_dt_trajectory)} langkah.")

    # ==========================================================
    # Tahap 2: Analisis dan Tampilan Error Relatif
    # ==========================================================
    
    print("\n--- Analisis Error Relatif (Referensi: dt = 0.001) ---")
    
    ref_dt = 0.001
    ref_traj = all_results_by_dt[ref_dt]
    
    # ### PERUBAHAN DI SINI ###
    # Loop untuk setiap 0.5 detik dari 0.5 hingga t_max
    for t_check in np.arange(0.5, t_max + 0.1, 0.5):
        print(f"\nPada t = {t_check:.1f} s:")
        
        # Indeks untuk data acuan
        idx_ref = int(t_check / ref_dt)

        # Pastikan data acuan ada pada waktu pengecekan
        if idx_ref >= len(ref_traj):
            print(f"  Data acuan tidak tersedia pada t={t_check:.1f}s. Analisis dihentikan.")
            break
            
        pos_acuan = ref_traj[idx_ref]
        norm_pos_acuan = np.linalg.norm(pos_acuan)

        # Loop untuk setiap dt yang akan dibandingkan
        for dt_val in dt_values:
            if dt_val == ref_dt:
                continue

            approx_traj = all_results_by_dt[dt_val]
            idx_approx = int(t_check / dt_val)
            
            # Periksa apakah lintasan aproksimasi stabil atau masih berjalan
            if idx_approx >= len(approx_traj):
                print(f"  - Error Relatif dt={dt_val}: - | Keterangan: Tidak Stabil/Simulasi Berhenti")
                continue

            pos_aproksimasi = approx_traj[idx_approx]
            
            # Hitung error
            if norm_pos_acuan > 0:
                error_absolut = np.linalg.norm(pos_aproksimasi - pos_acuan)
                error_relatif = (error_absolut / norm_pos_acuan) * 100
                print(f"  - Error Relatif dt={dt_val}: {error_relatif:.8f}%")
            else: # Kasus khusus jika posisi acuan masih di (0,0,0)
                error_absolut = np.linalg.norm(pos_aproksimasi)
                if error_absolut > 0:
                     print(f"  - Error Relatif dt={dt_val}: Infinit (Acuan di nol)")
                else:
                     print(f"  - Error Relatif dt={dt_val}: 0.0000%")

    # ==========================================================
    # Tahap 3: Visualisasi VPython dan Matplotlib
    # ==========================================================

    # --- Bagian Visualisasi VPython (3D) ---
    print("\nMemulai animasi VPython...")
    direction_arrow_scale = 5
    scene = vp.canvas(title="Simulasi Proyektil (Perbandingan dt - 3D)", width=1200, height=800, background=vp.color.white)
    
    # Lantai dan permukaan air
    vp.box(pos=vp.vector(0, -50, 0), size=vp.vector(200, 100, 200), color=vp.color.blue, opacity=0.2)
    vp.box(pos=vp.vector(0, 0, 0), size=vp.vector(200, 0.2, 200), color=vp.color.blue, opacity=0.5)

    # Panah arus air
    arrow_spacing = 30
    for x_arr in np.arange(-90, 100, arrow_spacing):
        for z_arr in np.arange(-90, 100, arrow_spacing):
            vp.arrow(pos=vp.vector(x_arr, 0.5, z_arr), axis=vp.vector(*current_dir) * direction_arrow_scale,
                     shaftwidth=0.3, color=vp.color.cyan, opacity=0.7)

    # Buat sphere dan simpan state trajektori
    spheres_data = []
    color_map = {2: vp.color.red, 1: vp.color.green, 0.5: vp.color.blue, 0.001: vp.color.black}

    for dt_val, traj in sorted(all_results_by_dt.items()):
        current_color_vpython = color_map.get(dt_val, vp.color.orange)
        label_text = f"dt = {dt_val}"
        label = vp.label(pos=vp.vector(*traj[0]), text=label_text, xoffset=10, yoffset=20,
                         color=current_color_vpython, opacity=1, box=True, height=10) 
        sph = vp.sphere(pos=vp.vector(*traj[0]), radius=r, color=current_color_vpython, make_trail=True, trail_radius=r*0.5, retain=2000)
        spheres_data.append({'sphere': sph, 'trajectory': traj, 'label': label, 'dt_value': dt_val})

    # Atur jumlah frame total untuk animasi agar durasinya sama
    animation_frames = 1000 
    animation_rate = 125 # Disesuaikan agar total durasi sekitar 8 detik (1000/125)

    print("Menjalankan animasi yang disinkronkan...")
    # Loop berdasarkan frame animasi, bukan langkah simulasi
    for i in range(animation_frames):
        vp.rate(animation_rate)
        
        # Update setiap proyektil berdasarkan progres animasi
        for data in spheres_data:
            sph = data['sphere']
            traj = data['trajectory']
            label = data['label']
            
            # Hitung indeks yang sesuai dalam lintasan berdasarkan progres animasi
            num_points = len(traj)
            # Rumus untuk memetakan progres (i / animation_frames) ke indeks lintasan
            if animation_frames > 1:
                target_idx = int((i / (animation_frames - 1)) * (num_points - 1))
            else:
                target_idx = 0
            
            # Pastikan indeks tidak melebihi batas
            if target_idx < num_points:
                new_pos = vp.vector(*traj[target_idx])
                sph.pos = new_pos
                label.pos = new_pos + vp.vector(0, r + 1, 0) # Beri sedikit jarak dari bola
    
    # Setelah loop selesai, pastikan semua proyektil berada di posisi akhir yang tepat
    for data in spheres_data:
        sph = data['sphere']
        traj = data['trajectory']
        label = data['label']
        final_pos = vp.vector(*traj[-1])
        sph.pos = final_pos
        label.pos = final_pos + vp.vector(0, r + 1, 0)
        label.text = f"dt = {data['dt_value']} (selesai)"

    # --- Bagian Visualisasi Matplotlib (2D) ---
    print("\nMembuat plot Matplotlib...")
    plt.figure(figsize=(12, 8))
    plt.title('Perbandingan Lintasan Proyektil dengan Berbagai Nilai dt')
    plt.xlabel('Jarak Horizontal (m)')
    plt.ylabel('Ketinggian (m)')
    plt.grid(True)
    plt.axhline(0, color='blue', linestyle='--', linewidth=0.8, label='Permukaan Air')

    color_map_mpl = {2: 'red', 1: 'green', 0.5: 'blue', 0.001: 'black'}

    for dt_val, traj in sorted(all_results_by_dt.items()):
        x_coords = traj[:, 0]
        y_coords = traj[:, 1]
        z_coords = traj[:, 2]
        horizontal_distance = np.sqrt(x_coords**2 + z_coords**2)
        
        plt.plot(horizontal_distance, y_coords, 
                 label=f'dt = {dt_val}', 
                 color=color_map_mpl.get(dt_val, 'orange'))
        
        plt.plot(horizontal_distance[-1], y_coords[-1], 'o', color=color_map_mpl.get(dt_val, 'orange'))

    plt.legend()
    plt.show()

    print("\nProgram selesai.")
