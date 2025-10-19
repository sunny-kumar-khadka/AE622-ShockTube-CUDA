import numpy as np
import numpy as np
import scipy
import scipy.optimize 
import yaml
from pathlib import Path


class AnalyticalShockTube:

    def __init__(self, 
                config_path: Path | None = None,
                output_path: Path | None = None):
        self.config = self._load_config(config_path)
        shared = self.config.get('parameters', {})
        sim_cfg = self.config.get('simulation', {})
        analytical_cfg = self.config.get('analytical', {})

        self.gamma = float(shared.get('gamma', sim_cfg.get('gamma', 1.4)))
        self.dustFrac = float(shared.get('dust_fraction', sim_cfg.get('dust_fraction', 0.0)))
        self.npts = int(shared.get('n_points', analytical_cfg.get('n_points', 256)))
        self.t = float(shared.get('t', analytical_cfg.get('t', 0.2)))
        self.left_state = tuple(sim_cfg.get('left_state', (1.0, 1.0, 0.0)))
        self.right_state = tuple(sim_cfg.get('right_state', (0.1, 0.125, 0.0)))
        output_file_cfg = analytical_cfg.get('output_file', 'results/analytic.dat')
        self.output_path = (Path(__file__).resolve().parent.parent / output_file_cfg) if not Path(output_file_cfg).is_absolute() else Path(output_file_cfg)

    def _load_config(self, config_path: Path | None):
        candidates = []
        if config_path is not None:
            candidates.append(Path(config_path))
        candidates.append(Path(__file__).resolve().parent.parent / 'config.yaml')
        candidates.append(Path('config.yaml'))
        for cfg in candidates:
            try:
                if cfg.exists():
                    with open(cfg, 'r') as f:
                        return yaml.safe_load(f) or {}
            except Exception:
                pass
        return {}

    @staticmethod
    def sound_speed(gamma, pressure, density, dustFrac=0.0):
        scale = np.sqrt(1 - dustFrac)
        return np.sqrt(gamma * pressure / density) * scale

    @staticmethod
    def shock_tube_function(p4, p1, p5, rho1, rho5, gamma, dustFrac=0.0):
        z = (p4 / p5 - 1.0)
        c1 = AnalyticalShockTube.sound_speed(gamma, p1, rho1, dustFrac)
        c5 = AnalyticalShockTube.sound_speed(gamma, p5, rho5, dustFrac)
        gm1 = gamma - 1.0
        gp1 = gamma + 1.0
        g2 = 2.0 * gamma
        fact = gm1 / g2 * (c5 / c1) * z / np.sqrt(1.0 + gp1 / g2 * z)
        fact = (1.0 - fact) ** (g2 / gm1)
        return p1 * fact - p4

    def calculate_regions(self, pl, ul, rhol, pr, ur, rhor):
        gamma = self.gamma
        dustFrac = self.dustFrac
        rho1 = rhol
        p1 = pl
        u1 = ul
        rho5 = rhor
        p5 = pr
        u5 = ur
        if pl < pr:
            rho1 = rhor
            p1 = pr
            u1 = ur
            rho5 = rhol
            p5 = pl
            u5 = ul
        p4 = scipy.optimize.fsolve(self.shock_tube_function, p1, (p1, p5, rho1, rho5, gamma))[0]
        z = (p4 / p5 - 1.0)
        c5 = self.sound_speed(gamma, p5, rho5, dustFrac)
        gm1 = gamma - 1.0
        gp1 = gamma + 1.0
        gmfac1 = 0.5 * gm1 / gamma
        gmfac2 = 0.5 * gp1 / gamma
        fact = np.sqrt(1.0 + gmfac2 * z)
        u4 = c5 * z / (gamma * fact)
        rho4 = rho5 * (1.0 + gmfac2 * z) / (1.0 + gmfac1 * z)
        w = c5 * fact
        p3 = p4
        u3 = u4
        rho3 = rho1 * (p3 / p1) ** (1.0 / gamma)
        return (p1, rho1, u1), (p3, rho3, u3), (p4, rho4, u4), (p5, rho5, u5), w

    def calc_positions(self, pl, pr, region1, region3, w, xi, t):
        gamma = self.gamma
        dustFrac = self.dustFrac
        p1, rho1 = region1[:2]
        p3, rho3, u3 = region3
        c1 = self.sound_speed(gamma, p1, rho1, dustFrac)
        c3 = self.sound_speed(gamma, p3, rho3, dustFrac)
        if pl > pr:
            xsh = xi + w * t
            xcd = xi + u3 * t
            xft = xi + (u3 - c3) * t
            xhd = xi - c1 * t
        else:
            xsh = xi - w * t
            xcd = xi - u3 * t
            xft = xi - (u3 - c3) * t
            xhd = xi + c1 * t
        return xhd, xft, xcd, xsh

    @staticmethod
    def region_states(pl, pr, region1, region3, region4, region5):
        if pl > pr:
            return {'Region 1': region1,
                    'Region 2': 'RAREFACTION',
                    'Region 3': region3,
                    'Region 4': region4,
                    'Region 5': region5}
        return {'Region 1': region5,
                'Region 2': region4,
                'Region 3': region3,
                'Region 4': 'RAREFACTION',
                'Region 5': region1}

    def create_arrays(self, pl, pr, xl, xr, positions, state1, state3, state4, state5, npts, t, xi):
        gamma = self.gamma
        dustFrac = self.dustFrac
        xhd, xft, xcd, xsh = positions
        p1, rho1, u1 = state1
        p3, rho3, u3 = state3
        p4, rho4, u4 = state4
        p5, rho5, u5 = state5
        gm1 = gamma - 1.0
        gp1 = gamma + 1.0
        x_arr = np.linspace(xl, xr, npts)
        rho = np.zeros(npts, dtype=float)
        p = np.zeros(npts, dtype=float)
        u = np.zeros(npts, dtype=float)
        c1 = self.sound_speed(gamma, p1, rho1, dustFrac)
        if pl > pr:
            for i, x in enumerate(x_arr):
                if x < xhd:
                    rho[i] = rho1
                    p[i] = p1
                    u[i] = u1
                elif x < xft:
                    u[i] = 2.0 / gp1 * (c1 + (x - xi) / t)
                    fact = 1.0 - 0.5 * gm1 * u[i] / c1
                    rho[i] = rho1 * fact ** (2.0 / gm1)
                    p[i] = p1 * fact ** (2.0 * gamma / gm1)
                elif x < xcd:
                    rho[i] = rho3
                    p[i] = p3
                    u[i] = u3
                elif x < xsh:
                    rho[i] = rho4
                    p[i] = p4
                    u[i] = u4
                else:
                    rho[i] = rho5
                    p[i] = p5
                    u[i] = u5
        else:
            for i, x in enumerate(x_arr):
                if x < xsh:
                    rho[i] = rho5
                    p[i] = p5
                    u[i] = -u1
                elif x < xcd:
                    rho[i] = rho4
                    p[i] = p4
                    u[i] = -u4
                elif x < xft:
                    rho[i] = rho3
                    p[i] = p3
                    u[i] = -u3
                elif x < xhd:
                    u[i] = -2.0 / gp1 * (c1 + (xi - x) / t)
                    fact = 1.0 + 0.5 * gm1 * u[i] / c1
                    rho[i] = rho1 * fact ** (2.0 / gm1)
                    p[i] = p1 * fact ** (2.0 * gamma / gm1)
                else:
                    rho[i] = rho1
                    p[i] = p1
                    u[i] = -u1
        return x_arr, p, rho, u

    def solve(self, left_state, right_state, geometry, t, npts):
        pl, rhol, ul = left_state
        pr, rhor, ur = right_state
        xl, xr, xi = geometry
        if xl >= xr:
            print('xl has to be less than xr!')
            exit()
        if xi >= xr or xi <= xl:
            print('xi has in between xl and xr!')
            exit()
        region1, region3, region4, region5, w = self.calculate_regions(pl, ul, rhol, pr, ur, rhor)
        regions = self.region_states(pl, pr, region1, region3, region4, region5)
        x_positions = self.calc_positions(pl, pr, region1, region3, w, xi, t)
        pos_description = ('Head of Rarefaction', 'Foot of Rarefaction', 'Contact Discontinuity', 'Shock')
        positions = dict(zip(pos_description, x_positions))
        x, p, rho, u = self.create_arrays(pl, pr, xl, xr, x_positions, region1, region3, region4, region5, npts, t, xi)
        energy = p / (rho * (self.gamma - 1.0))
        rho_total = rho / (1.0 - self.dustFrac)
        val_dict = {'x': x, 'p': p, 'rho': rho, 'u': u, 'energy': energy, 'rho_total': rho_total}
        return positions, regions, val_dict

    def run(self):
        positions, regions, values = self.solve(
            left_state=self.left_state,
            right_state=self.right_state,
            geometry=(0.0, 1.0, 0.5),
            t=self.t,
            npts=self.npts,
        )
        e = values['p'] / (self.gamma - 1.0) / values['rho']
        E = values['p'] / (self.gamma - 1.0) + 0.5 * values['rho'] * values['u'] ** 2
        T = values['p'] / values['rho']
        c = np.sqrt(self.gamma * values['p'] / values['rho'])
        M = values['u'] / c
        h = e + values['p'] / values['rho']
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, 'w') as f:
            f.write('Variables = x, rho, u, p, e, Et, T, c, M, h\n')
            for i in range(self.npts):
                text = str(values['x'][i]) + ' ' + str(values['rho'][i]) + ' ' + str(values['u'][i]) + ' ' + \
                       str(values['p'][i]) + ' ' + str(e[i]) + ' ' + str(E[i]) + ' ' + str(T[i]) + ' ' + \
                       str(c[i]) + ' ' + str(M[i]) + ' ' + str(h[i]) + '\n'
                f.write(text)


if __name__ == '__main__':
    AnalyticalShockTube().run()
