# gaN_fe_mobility.py
# 実装方針：Tian et al. (2024) の各式をそのまま Python に移植。
# 注意：論文の係数は実用単位（cm^-3, eV, K, ...）を混在しているため、
#        下では論文の係数をなるべくそのまま利用しつつ、各量の単位を揃えています。

import numpy as np
import matplotlib.pyplot as plt

# --- 基本定数 ---
kB_eV = 8.617333262e-5        # eV/K
kB = 1.380649e-23             # J/K
h = 6.62607015e-34            # J s
hbar = h / (2*np.pi)
e = 1.602176634e-19           # C
m0 = 9.10938356e-31           # kg

# --- 論文で使われている物性パラメータ（Table 1 等） ---
eps0_cgs = 8.85418e-14        # F/cm (論文表記)
eps_s = 8.9                   # 低周波誘電率
eps_inf = 5.35                # 高周波誘電率 (論文)
mn_rel = 0.22                 # 有効質量比 mn = 0.22 m0
mn = mn_rel * m0
Cl = 2.65e7                   # N·cm^-2 (論文表記)
Dac = 8.3                     # eV
K2 = 0.039                    # electromechanical coupling constant squared (論文のK^2)
Theta = 1057                  # K (hbar*omega_l / kB)
hbar_omega_eV = 0.0912        # eV (論文)
# その他：論文中の経験的係数などは式内にそのまま使います

# --- ユーザーが変えるべきパラメータ（例） ---
ND = 1.0e16    # donor 全和 (cm^-3) = ND1 + ND2
NA = 1.0e17    # deep acceptor (Fe) レベル濃度 (cm^-3)
Ec_minus_EA = 1.5 #0.58  # eV (論文が良く一致するとした値)

# --- 補助関数 ---
def Nc_cm3(T):
    """有効状態密度 Nc (cm^-3)。論文式 (6) を SI->cm^(-3) に整合して実装"""
    # 論文: Nc = 2 * (2*pi*mn*m0*kB*T / h^2)^(3/2)
    # ここで mn は相対値論文では mn = 0.22 (相対), 実質質量 = mn_rel * m0
    m_eff = mn_rel * m0
    Nc_m3 = 2 * ((2*np.pi * m_eff * kB * T) / (h**2))**1.5
    Nc_cm3 = Nc_m3 * 1e-6   # m^-3 -> cm^-3
    return Nc_cm3

def fA_from_Ns(ND_val, NA_val):
    """論文での近似: fA = ND/NA"""
    return ND_val / NA_val

def EF_minus_EA_from_fA(fA, T):
    """fA = 1 / (1 + 2 exp((EA - EF)/kT)) から EA - EF を解く
       fA = ND/NA とおく（論文の近似） -> EF を求める。"""
    # fA = 1/(1 + 2*exp((EA-EF)/(kT)))
    # これを (EA-EF) について解く:
    # (EA-EF) = kT * ln( 2*(1/fA - 1) )
    # 論文の式では fA = ND/NA なので fA <= 1 が期待
    # ただし数値的安定性のため下限/上限処理
    eps = 1e-30
    fA = np.clip(fA, eps, 1-eps)
    return kB_eV * T * np.log( (1.0/fA - 1.0) / 2.0 )

def electron_density_n(T, ND_val, NA_val, Ec_minus_EA_eV=Ec_minus_EA):
    """非縮退近似 (eq.5): n = Nc * exp[-(Ec - EF)/kT]
       Ec - EF = (Ec - EA) + (EA - EF)
       EA - EF を fA と上の式から計算。
    """
    fA = fA_from_Ns(ND_val, NA_val)
    EA_minus_EF = EF_minus_EA_from_fA(fA, T)  # actually EA - EF
    Ec_minus_EF = Ec_minus_EA_eV + EA_minus_EF
    Nc = Nc_cm3(T)
    # 注意: ここで kB_eV*T を分母にする (単位 eV)
    n = Nc * np.exp( - (Ec_minus_EF) / (kB_eV * T) )
    return n  # cm^-3

# --- 各散乱による移動度（論文の式をそのまま実装） ---
def mu_II(T, n_free, ND_val, NA_val):
    """Brooks–Herring の形（論文式 (1)）を実装（論文の前置係数を使用）"""
    # 論文で与えられた式:
    # muII = 3.28e15 * T^(3/2) * eps_s^2 / ( NI * sqrt(mn) ) * [ ln(1+c) - c/(1+c) ]
    # 論文の式が間違っていたため、ほかの文献を参考に計算しています。
    NI = 2.0 * ND_val * 1e6  # 論文近似 (eq.4)
    mn_eff = mn_rel
    eps0 = eps0_cgs * 1e2
    c = 8.0 * eps0 * eps_s * m0 * mn_eff * (kB * T)**2 / (hbar**2 * e**2 * n_free * 1e6)
    bracket = np.log(1.0 + c) - c / (1.0 + c)
    muII = 16.0 * np.sqrt(2) * np.pi * (kB * T)**1.5 * ((eps0*eps_s)**2) / (e**3 * NI * np.sqrt(mn_eff*m0) * bracket ) *1e4
    return muII

def mu_ac(T):
    """音響変形ポテンシャル散乱（論文式 (8)）"""
    # µac = (2/sqrt(2π)) * e hbar^4 Cl / [3 (kB T)^(3/2) Dac^2 (m0 mn)^(5/2)]  (論文表記)
    pref = 2.0 * np.sqrt(2*np.pi)
    numerator = pref * e * (hbar**4) * (Cl*1e4)
    denom = 3.0 * (kB * T)**1.5 * ((Dac*e)**2) * ( (m0 * mn_rel)**2.5 )
    muac = numerator / denom * 1e4
    # 論文は最終単位を cm^2/(V s) に換算した数値になるように前置因子を使っているため
    return muac

def mu_pz(T):
    """ピエゾ散乱（論文式 (9)）"""
    # µpz = (16 / sqrt(2π)) * (3/??) * hbar^2 eps0 eps_s / [ e (kB T)^(1/2) K^2 (m0 mn)^(3/2) ]
    # 論文では少し整理された形で与えられている。ここでは式に忠実に実装。
    pref = 16.0 * np.sqrt(2*np.pi)
    numerator = pref * (hbar**2) * (eps0_cgs*1e2) * eps_s
    denom = 3* e * (kB * T)**0.5 * (K2) * ( (m0 * mn_rel)**1.5 )
    mupz = numerator / denom *1e4
    return mupz

def chi_Z(Z):
    """論文の近似 χ(Z) for Z>>1: χ = (3/8)*(π Z)^(1/2)"""
    return (3.0/8.0) * np.sqrt(np.pi * Z)

def mu_po(T):
    """極性光学フォノン散乱（論文式 (12)）"""
    Z = Theta / T
    # 使用される式（論文 (12)）:
    # mu_po = 0.2357 * sqrt(hbar*omega_l (eV)) / ( chi(Z) * [exp(Z)-1] * Z^{-1/2} ) * (mn)^{-3/2} * (eps_inf^{-1} - eps_s^{-1})^{-1}
    # 論文式は最終的に cm^2/(V s) の値になる形で与えられているので、それに従う。
    chi = chi_Z(Z) if Z > 5 else chi_Z(Z)  # 近似をそのまま使用
    denom_factor = chi * (np.exp(Z) - 1.0) * (Z**(-0.5))
    eps_term = (1.0/eps_inf - 1.0/eps_s)
    mu_po_val = 0.2357 / np.sqrt(hbar_omega_eV) * (denom_factor) / (mn_rel**(-1.5)) * (eps_term**(-1)*1e2) 
    return mu_po_val

# --- 合成移動度（Matthiessen の和） ---
def mu_total_by_matthiessen(mu_list):
    inv = 0.0
    for m in mu_list:
        inv += 1.0 / m
    return 1.0 / inv

# --- 温度レンジで計算してプロット ---
def compute_vs_temperature(Tmin=10, Tmax=500, nT=200):
    Ts = np.linspace(Tmin, Tmax, nT)
    mus = []
    muIIs = []
    muacs = []
    mupzs = []
    mupos = []
    ns = []
    rhos = []
    for T in Ts:
        n_free = electron_density_n(T, ND, NA, Ec_minus_EA)
        ns.append(n_free)
        muii = mu_II(T, n_free, ND, NA)
        muac_val = mu_ac(T)
        mupz_val = mu_pz(T)
        mupo_val = mu_po(T)
        mu_tot = mu_total_by_matthiessen([muii, muac_val, mupz_val, mupo_val])
        mus.append(mu_tot)
        muIIs.append(muii)
        muacs.append(muac_val)
        mupzs.append(mupz_val)
        mupos.append(mupo_val)
        # 抵抗率 rho = 1 / (e * mu_n * n)  （単位：Ω·cm を得るために変換）
        # mu の単位は cm^2/(V s) を仮定, n は cm^-3 -> rho (Ω·cm) = 1/( e[C] * mu[cm^2/Vs] * n[cm^-3] )
        # 電気抵抗率の単位合わせに注意。ここでは e[C] を使って Ω·cm に直接変換する近似式を使用します。
        if mu_tot > 0 and n_free > 0:
            rho = 1.0 / (e * mu_tot * n_free)  # ただし単位は SI 混在なので、結果は参考値
        else:
            rho = np.nan
        rhos.append(rho)
    return {
        'T': Ts,
        'mu': np.array(mus),
        'muII': np.array(muIIs),
        'muac': np.array(muacs),
        'mupz': np.array(mupzs),
        'mupo': np.array(mupos),
        'n': np.array(ns),
        'rho': np.array(rhos)
    }

if __name__ == "__main__":
    res = compute_vs_temperature(10, 500, 250)

    # プロット（移動度）
    plt.figure(figsize=(7,5))
    plt.semilogy(res['T'], res['mu'], label='mu_total')
    plt.semilogy(res['T'], res['muII'], '--', label='mu_II')
    plt.semilogy(res['T'], res['muac'], ':', label='mu_ac')
    plt.semilogy(res['T'], res['mupz'], '-.', label='mu_pz')
    plt.semilogy(res['T'], res['mupo'], '--', label='mu_po')
    plt.xlabel('Temperature (K)')
    plt.ylabel('Mobility (cm^2 V^-1 s^-1) -- approximate')
    plt.ylim(1, 1e8)
    plt.legend()
    plt.title('Calculated mobility components (approx)')
    plt.grid(True)
    plt.show()

    # 抵抗率プロット（参考）
    plt.figure(figsize=(7,5))
    plt.semilogy(res['T'], res['rho'])
    plt.xlabel('Temperature (K)')
    plt.ylabel('Resistivity (Ω·cm) -- approximate')
    plt.title('Calculated resistivity (approx)')
    plt.ylim(1, 1e12)
    plt.grid(True)
    plt.show()

    # 温度300 K 時の値の表示
    idx300 = np.argmin(np.abs(res['T'] - 300))
    print("T=300K: n = {:.3e} cm^-3, mu = {:.3e} cm^2/Vs, rho = {:.3e} (approx)".format(
        res['n'][idx300], res['mu'][idx300], res['rho'][idx300]
    ))

