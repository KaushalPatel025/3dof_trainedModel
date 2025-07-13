import numpy as np
from scipy.fft import fft, fftfreq

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import eigh


class DOF3SystemGenerator:
    def __init__(self, T_min=0.001, T_max=2.0, mass_range=(5000, 50000), stiffness_range=(1e5, 1e8)):
        self.T_min = T_min
        self.T_max = T_max
        self.mass_range = mass_range
        self.stiffness_range = stiffness_range

    def generate_mass_matrix(self):
        log_m_min, log_m_max = np.log(self.mass_range[0]), np.log(self.mass_range[1])
        masses = np.exp(np.random.uniform(log_m_min, log_m_max, 3))
        return np.diag(masses)

    def generate_textbook_springs(self):
        log_k_min, log_k_max = np.log(self.stiffness_range[0]), np.log(self.stiffness_range[1])
        k1, k2, k3 = np.exp(np.random.uniform(log_k_min, log_k_max, 3))
        return k1, k2, k3

    def build_textbook_stiffness_matrix(self, k1, k2, k3):
        K = np.array([
            [k1 + k2, -k2, 0],
            [-k2, k2 + k3, -k3],
            [0, -k3, k3]
        ])
        return K

    def extract_spring_constants(self, K):
        k2 = -K[0, 1]
        k3 = -K[1, 2]
        k1 = K[0, 0] - k2
        return k1, k2, k3

    def generate_system_with_target_period(self, T_target):
        max_attempts = 100
        for _ in range(max_attempts):
            M = self.generate_mass_matrix()
            k1, k2, k3 = self.generate_textbook_springs()
            K = self.build_textbook_stiffness_matrix(k1, k2, k3)
            try:
                eigvals, _ = eigh(K, M)
                natural_freqs = np.sqrt(np.abs(eigvals))
                periods = 2 * np.pi / natural_freqs
                if np.all(eigvals > 0):
                    fundamental_period = np.max(periods)
                    scale_factor = (fundamental_period / T_target) ** 2
                    K_scaled = K * scale_factor
                    eigvals_scaled, _ = eigh(K_scaled, M)
                    if np.all(eigvals_scaled > 0):
                        k1_s, k2_s, k3_s = self.extract_spring_constants(K_scaled)
                        return M, K_scaled, eigvals_scaled, (k1_s, k2_s, k3_s)
            except:
                continue
        raise ValueError(f"Could not generate stable system for T={T_target:.3f} s")

    def generate_dataset(self, n_samples=1000, distribution='uniform'):
        if distribution == 'uniform':
            target_periods = np.random.uniform(self.T_min, self.T_max, n_samples)
        else:
            log_T_min, log_T_max = np.log(self.T_min), np.log(self.T_max)
            target_periods = np.exp(np.random.uniform(log_T_min, log_T_max, n_samples))

        np.random.shuffle(target_periods)
        systems = []
        actual_periods = []
        failed = 0

        for i, T_target in enumerate(target_periods):
            try:
                M, K, eigvals, springs = self.generate_system_with_target_period(T_target)
                freqs = np.sqrt(eigvals)
                periods = 2 * np.pi / freqs
                systems.append({
                    'M': M,
                    'K': K,
                    'eigenvalues': eigvals,
                    'natural_frequencies': freqs,
                    'periods': periods,
                    'fundamental_period': np.max(periods),
                    'target_period': T_target,
                    'spring_constants': {
                        'k1': springs[0],
                        'k2': springs[1],
                        'k3': springs[2]
                    }
                })
                actual_periods.append(np.max(periods))
            except:
                failed += 1
                continue

        print(f"Successfully generated {len(systems)} systems, failed: {failed}")
        return systems, np.array(actual_periods)

    def analyze_period_distribution(self, periods, bins=50):
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.hist(periods, bins=bins, alpha=0.7, edgecolor='blue')
        plt.xlabel('Fundamental Period (s)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Fundamental Periods')
        plt.grid(True)

        plt.subplot(2, 2, 2)
        plt.hist(periods, bins=np.logspace(np.log10(self.T_min), np.log10(self.T_max), bins), alpha=0.7, edgecolor='blue')
        plt.xscale('log')
        plt.xlabel('Fundamental Period (s) - Log Scale')
        plt.ylabel('Frequency')
        plt.title('Log-Scale Distribution')
        plt.grid(True)

        plt.subplot(2, 2, 3)
        sorted_periods = np.sort(periods)
        cumulative = np.arange(1, len(sorted_periods) + 1) / len(sorted_periods)
        plt.plot(sorted_periods, cumulative, 'blue', linewidth=2)
        plt.xlabel('Fundamental Period (s)')
        plt.ylabel('Cumulative Probability')
        plt.title('Cumulative Distribution Function')
        plt.grid(True)

        plt.subplot(2, 2, 4)
        stats_text = f"""
        Statistics:
        Mean: {np.mean(periods):.4f} s
        Std:  {np.std(periods):.4f} s
        Min:  {np.min(periods):.4f} s
        Max:  {np.max(periods):.4f} s
        25%:  {np.percentile(periods, 25):.4f} s
        50%:  {np.percentile(periods, 50):.4f} s
        75%:  {np.percentile(periods, 75):.4f} s
        """
        plt.text(0.1, 0.5, stats_text, transform=plt.gca().transAxes, fontsize=10, family='monospace')
        plt.axis('off')
        plt.title('Statistical Summary')

        plt.tight_layout()
        plt.show()

        return {
            'mean': np.mean(periods),
            'std': np.std(periods),
            'min': np.min(periods),
            'max': np.max(periods),
            'percentiles': np.percentile(periods, [25, 50, 75])
        }

    def systems_to_dataframe(self, systems):
        data = []
        for sys in systems:
            row = {
                'k1': sys['spring_constants']['k1'],
                'k2': sys['spring_constants']['k2'],
                'k3': sys['spring_constants']['k3'],
                'm1': sys['M'][0, 0],
                'm2': sys['M'][1, 1],
                'm3': sys['M'][2, 2],
                'T1': sys['periods'][0],
                'T2': sys['periods'][1],
                'T3': sys['periods'][2],
                'fundamental_period': sys['fundamental_period'],
                'target_period': sys['target_period']
            }
            data.append(row)
        df = pd.DataFrame(data)
        return df


def generate_dof3_dataset(n_samples=100, T_min=0.001, T_max=2.0, distribution='uniform'):
    generator = DOF3SystemGenerator(T_min=T_min, T_max=T_max)
    systems, periods = generator.generate_dataset(n_samples=n_samples, distribution=distribution)
    df = generator.systems_to_dataframe(systems)
    stats = generator.analyze_period_distribution(periods)
    return df, stats

        
def newmark(dt, beta, gamma, mass, stiffness, damping_ratio, ground_accel):

    damping = 2 * damping_ratio * np.sqrt(stiffness * mass)
    time_total = int((len(ground_accel)-1)*dt)  # seconds
    time1 = np.arange(0, time_total + dt, dt)
    n_steps = len(time1)
    # Initialize Response Arrays
    displacement = np.zeros(n_steps)
    velocity = np.zeros(n_steps)
    acceleration = np.zeros(n_steps)

    # Initial Conditions (at rest)
    displacement[0] = 0.0
    velocity[0] = 0.0

    # Calculate Initial Acceleration
    # m*a0 + c*v0 + k*u0 = -m*ag0
    acceleration[0] = (-mass * ground_accel[0] - damping * velocity[0] - stiffness * displacement[0]) / mass

    # Newmark-β Integration Constants
    a0 = 1.0 / (beta * dt**2)
    a1 = gamma / (beta * dt)
    a2 = 1.0 / (beta * dt)
    a3 = 1.0 / (2.0 * beta) - 1.0
    a4 = gamma / beta - 1.0
    a5 = dt / 2.0 * (gamma / beta - 2.0)
    a6 = dt * (1.0 - gamma)
    a7 = gamma * dt

    # Effective Stiffness
    K_eff = stiffness + a1 * damping + a0 * mass

    for i in range(1, n_steps):
        # Effective force at time i
        force_i = -mass * ground_accel[i]
        
        # Add contributions from previous time step
        force_eff = (force_i + 
                    mass * (a0 * displacement[i-1] + a2 * velocity[i-1] + a3 * acceleration[i-1]) +
                    damping * (a1 * displacement[i-1] + a4 * velocity[i-1] + a5 * acceleration[i-1]))
        
        # Solve for displacement at time i
        displacement[i] = force_eff / K_eff
        
        # Calculate acceleration at time i
        acceleration[i] = a0 * (displacement[i] - displacement[i-1]) - a2 * velocity[i-1] - a3 * acceleration[i-1]
        
        # Calculate velocity at time i
        velocity[i] = velocity[i-1] + a6 * acceleration[i-1] + a7 * acceleration[i]
    return time1, displacement, velocity, acceleration

def RS(dt, beta, gamma, damping_ratio, ground_accel):
    # System Properties (Single DOF)
    T = np.logspace(np.log10(0.001), np.log10(10), 100)
    dispRS = np.zeros(len(T))
    velRS = np.zeros(len(T))
    accelRS = np.zeros(len(T))

    for i in range (len(T)):

        omega = 2 * np.pi / T[i]
        mass = 1000.0      # kg
        stiffness = mass * omega**2  # N/m
        time, displacement, velocity, acceleration = newmark(dt, beta, gamma, mass, stiffness, damping_ratio, ground_accel)
        dispmax = np.max(np.abs(displacement))
        velmax = np.max(np.abs(velocity))
        accelmax = np.max(np.abs(acceleration))
        dispRS[i] = dispmax
        velRS[i] = velmax
        accelRS[i] = accelmax

    return T, dispRS, velRS, accelRS


def arias_intensity(acc, dt):
    g = 9.81
    ai = np.cumsum(acc**2) * dt * np.pi / (2 * g)
    return ai

def significant_duration(acc, dt, start_pct=5, end_pct=95):
    ai = arias_intensity(acc, dt)
    total_ai = ai[-1]
    start_time = np.argmax(ai >= (start_pct / 100) * total_ai) * dt
    end_time = np.argmax(ai >= (end_pct / 100) * total_ai) * dt
    return end_time - start_time
    
def get_predominant_frequency(acc, dt):
    N = len(acc)
    f = fftfreq(N, dt)[:N//2]
    A = np.abs(fft(acc))[:N//2]
    return f[np.argmax(A)]