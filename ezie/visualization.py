#%% Import
from typing import Optional, Union
import numpy as np
import matplotlib.pyplot as plt
from .model import Model
from polplot import Polarplot
from .regularization_optimizer import RegularizationOptimizer
from .validation import Validation

#%% Functions

def calc_gini(x):
    N = x.size
    x = np.sort(np.abs(x))
    norm_l1 = np.sum(np.abs(x))
    k = np.arange(1, N + 1)
    gini = np.sum((x / norm_l1) * ((N - k + 0.5) / N))
    gini = 1 - 2 * gini
    return gini

def calc_gini_matrix(x):
    N = x.shape[0]
    k = np.tile(np.arange(1, N+1), (N, 1))    
    x = np.sort(abs(x), axis=1)
    norm_l1 = np.tile(np.sum(abs(x), axis=1).reshape(-1, 1), (1, N))    
    gini = (x / norm_l1) * ((N - k + 0.5)/N)
    gini = 1 - 2*np.sum(gini, axis=1)    
    return gini

#%% Model class

class Plotter(object):
    def __init__(self, 
                 model: Model = None,
                 regOpt: Optional[RegularizationOptimizer] = None,
                 Blvls: Optional[np.ndarray] = np.linspace(-600, 600, 41),
                 Bcmap: Optional[str] = 'bwr',
                 Bucmap: Optional[str] = 'Reds',
                 Bulvls: Optional[np.ndarray] = np.linspace(0, 400, 40),
                 Jscale: Optional[Union[float, int]] = 1e1,
                 userid: Optional[str] = None):
        
        
        self.model = model
        self.userid = userid
        self.Blvls = Blvls
        self.Bulvls = Bulvls
        self.Bcmap = Bcmap
        self.Bucmap = Bucmap
        self.Jscale = Jscale
        self._regOpt = regOpt
        self._val = None
        
    @property
    def regOpt(self):
        if self._regOpt is None:
            return self.model.regOpt
        else:
            return self._regOpt
    
    @property
    def val(self):
        if self._val is None:
            self._val = Validation(self.model, self.userid)
        return self._val
    
    @property
    def grid(self):
        return self.model.grid
    
    @property
    def ev(self):
        return self.model.ev
    
    @property
    def data(self):
        return self.model.data
    
#%%    
    
    def plot_validation(self):
        
        figs, axs_ = [], []
        for i, IAGA in enumerate(self.val.IAGA):
            fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
            sm_B, ev_B, ev_B_var = self.val.get_IAGA(IAGA)
            
            for j, c in enumerate(['Be', 'Bn', 'Bu']):
                axs[j].scatter(np.zeros(sm_B[0].size), sm_B[j]*1e9, alpha=.5)
                axs[j].plot([-.1, .1], [np.median(sm_B[j]*1e9)]*2, color='tab:red', linewidth=2)
                
                mean = ev_B[j]*1e9
                low = mean - ev_B_var[j]*1e9
                high = mean + ev_B_var[j]*1e9
                axs[j].plot([.5]*2, [low, high], color='C1')
                axs[j].plot([.4, .6], [low]*2, color='C1')
                axs[j].plot([.4, .6], [mean]*2, color='C1', linewidth=2)
                axs[j].plot([.4, .6], [high]*2, color='C1')
                
                axs[j].set_title(c)
                axs[j].set_xticks([0, .5])
                axs[j].set_xticklabels(['Supermag', 'EZIE'])
                axs[j].yaxis.grid()
            
            axs[0].set_xlim(-.5, 1)
            axs[0].set_ylabel('nT')
            plt.suptitle(IAGA)
            
            figs.append(fig)
            axs_.append(axs)
        
        return figs, axs_
    
    def plot_validation_map(self):
        
        if self.val.IAGA.size == 0:
            return None, None
        
        x = self.grid.xi.max() - self.grid.xi.min()
        y = self.grid.eta.max() - self.grid.eta.min()
        
        fig = plt.figure(figsize=(10, y/(x) * 10))
        ax = plt.gca()
        
        self.plot_latlon(ax)
        self.plot_tracks(ax, lw=8)
        
        lon, lat, Be_sm, Bn_sm = [], [], [], []
        Be_sm_avg, Bn_sm_avg, Bu_sm_avg = [], [], []
        Be_avg, Bn_avg, Bu_avg = [], [], []
        for IAGA, lo, la in zip(self.val.IAGA, self.val.lon, self.val.lat):
            sm_B, ev_B, ev_B_var = self.val.get_IAGA(IAGA)
            Be_sm.append(sm_B[0])
            Bn_sm.append(sm_B[1])
            lon.append(np.ones(sm_B[0].size)*lo)
            lat.append(np.ones(sm_B[0].size)*la)
            
            Be_sm_avg.append(np.median(sm_B[0]))
            Bn_sm_avg.append(np.median(sm_B[1]))
            Bu_sm_avg.append(np.median(sm_B[2]))
            
            Be_avg.append(ev_B[0])
            Bn_avg.append(ev_B[1])
            Bu_avg.append(ev_B[2])
            
        xi, eta, Bxi, Beta = self.grid.projection.vector_cube_projection(np.hstack(Be_sm), np.hstack(Bn_sm), np.hstack(lon), np.hstack(lat))
        Bscale = 10*np.median(np.sqrt(Bxi**2 + Beta**2))        
        plt.quiver(xi, eta, Bxi, Beta, scale=Bscale, width=.001, color='k', alpha=.3)
        
        xi, eta, Bxi, Beta = self.grid.projection.vector_cube_projection(np.array(Be_sm_avg), np.array(Bn_sm_avg), self.val.lon, self.val.lat)
        plt.quiver(xi, eta, Bxi, Beta, scale=Bscale, width=.005, color='C0')
        
        xi, eta, Bxi, Beta = self.grid.projection.vector_cube_projection(np.array(Be_avg), np.array(Bn_avg), self.val.lon, self.val.lat)
        plt.quiver(xi, eta, Bxi, Beta, scale=Bscale, width=.005, color='C1')
        
        dBu = (np.array(Bu_avg) - np.array(Bu_sm_avg))*1e9
        dBumax = np.max(abs(dBu))
        xi, eta = self.grid.projection.geo2cube(self.val.lon, self.val.lat)
        cc = plt.scatter(xi, eta, 700, dBu, cmap='bwr', marker='*', vmin=-dBumax, vmax=dBumax, edgecolor='k', linewidths=.8)
        
        cbar = plt.colorbar(cc, ax=ax, orientation='horizontal',fraction=0.03, pad=0.02)
        cbar.ax.tick_params(labelsize=20)
        cbar.set_label('Bu (EZIE) - Bu (SuperMAG) [nT] at 0 km (geocentric)', fontsize=25)
        
        ax.set_xticks([])
        ax.set_yticks([])
        
        ax.set_title(self.data.date, fontsize=25)
        
        return fig, ax
        
    def plot_solution(self):        
        fig, axs = plt.subplots(3, 2, figsize=(16, 14), layout='constrained')
        paxs = [Polarplot(ax, sector='night') for ax in axs.flatten()]
        
        
        lat, _, lt, Be, Bn = self.ev.geo2qd(self.grid.lat_mesh.flatten(), 
                                               self.grid.lon_mesh.flatten(), 
                                               80, 
                                               self.ev.Be, self.ev.Bn)
        cc = self.plot_pp(paxs[0], lat, lt, Be*1e9, self.Bcmap, self.Blvls)
        cc = self.plot_pp(paxs[2], lat, lt, Bn*1e9, self.Bcmap, self.Blvls)
        cc = self.plot_pp(paxs[4], lat, lt, self.ev.Bu*1e9, self.Bcmap, self.Blvls)
        
        lat, _, lt, Be, Bn = self.ev.geo2qd(self.grid.lat_mesh.flatten(), 
                                               self.grid.lon_mesh.flatten(), 
                                               80, 
                                               self.ev.Be_u, self.ev.Bn_u)
        cc = self.plot_pp(paxs[1], lat, lt, Be*1e9, self.Bucmap, self.Bulvls)
        cc = self.plot_pp(paxs[3], lat, lt, Bn*1e9, self.Bucmap, self.Bulvls)
        cc = self.plot_pp(paxs[5], lat, lt, self.ev.Bu_u*1e9, self.Bucmap, self.Bulvls)
        
        
        lat, _, lt, Je, Jn = self.ev.geo2qd(self.grid.lat_mesh.flatten(), 
                                               self.grid.lon_mesh.flatten(), 
                                               110, 
                                               self.ev.Je, self.ev.Jn)
        for ax in paxs[::2]:
            self.plot_quiver_pp(ax, lat, lt, Je, Jn, self.Jscale)
        
        lat, _, lt, Je, Jn = self.ev.geo2qd(self.grid.lat_mesh.flatten(), 
                                               self.grid.lon_mesh.flatten(), 
                                               110, 
                                               self.ev.Je_u, self.ev.Jn_u)
        for ax in paxs[1::2]:
            self.plot_quiver_pp(ax, lat, lt, Je, np.zeros_like(Je), self.Jscale)
            self.plot_quiver_pp(ax, lat, lt, np.zeros_like(Je), Jn, self.Jscale)
        
        # Add a shared colorbar below all subplots
        cbar = fig.colorbar(cc, ax=axs.ravel().tolist(), orientation='horizontal',fraction=0.03, pad=0.02)
        cbar.set_label('dB$_i$ [nT] at 80 km (geocentric)', fontsize=15)
        
        return fig, axs

    def plot_solution_cs(self):
        
        x = self.grid.xi.max() - self.grid.xi.min()
        y = self.grid.eta.max() - self.grid.eta.min()
        
        fig, axs = plt.subplots(1, 3, figsize=(10, y/(3*x) * 10))
        cc = self.plot_map(axs[0], self.ev.Be*1e9, self.Bcmap, self.Blvls)
        cc = self.plot_map(axs[1], self.ev.Bn*1e9, self.Bcmap, self.Blvls)
        cc = self.plot_map(axs[2], self.ev.Bu*1e9, self.Bcmap, self.Blvls)
        
        for ax in axs:
            self.plot_quiver(ax, self.Jscale)
        
        # Add a shared colorbar below all subplots
        cbar = fig.colorbar(cc, ax=axs.ravel().tolist(), orientation='horizontal',fraction=0.03, pad=0.02)
        cbar.set_label('dB$_i$ [nT] at 80 km (geocentric)', fontsize=15)
        
        axs[0].set_title('Be')
        axs[1].set_title('Bn')
        axs[2].set_title('Bu')
        axs[1].text(.5, 1.06, self.data.date, ha='center', va='center', fontsize=15, transform=axs[1].transAxes)
        
        return fig, axs
    
    def plot_regOpt(self):
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        
        self.plot_id(axs[0])
        self.plot_lambda_relation(axs[1])
        self.plot_L_curve(axs[2])
        
        return fig, axs

    def plot_data_model_comparison(self):
        
        nmem = len(self.data.mems)
        ncomp = len(self.data.comp)
        #fig, axs = plt.subplots(nmem, ncomp, figsize=(10*(ncomp/nmem), 10), sharex=True, sharey=True)
        fig, axs = plt.subplots(nmem, ncomp, figsize=(10, 10), sharex=True, sharey=True)
        if ncomp == 1:
            axs = axs.reshape(-1, 1)
        
        for i, mem in enumerate(self.data.mems):
            Be, Bn, Bu, Be_u, Bn_u, Bu_u = self.ev.get_B_predictions(lat=mem.lat, lon=mem.lon, r=mem.r, ret=True)
            x = mem.lat
            for j, (ax, c) in enumerate(zip(axs[i], self.data.comp)):
                if c =='s':
                    B0, Be0, Bn0, Bu0, _ = self.ev.calc_main_field(mem.lat, mem.lon, mem.r, ret=True)                    
                    yd, ydu = mem.B, np.sqrt(mem.cov_mag)
                    yp, ypu = np.sqrt((Be+Be0)**2 + (Bn+Bn0)**2 + (Bu+Bu0)**2) - B0, np.sqrt(Be_u**2 + Bn_u**2 + Bu_u**2)
                if c =='e':
                    yd, ydu = mem.Be, np.sqrt(mem.cov_ee)
                    yp, ypu = Be, Be_u
                if c =='n':
                    yd, ydu = mem.Bn, np.sqrt(mem.cov_nn)
                    yp, ypu = Bn, Bn_u
                if c =='u':
                    yd, ydu = mem.Bu, np.sqrt(mem.cov_uu)
                    yp, ypu = Bu, Bu_u
                
                s = 1e9
                
                ydl, ydh = yd-ydu, yd+ydu
                ypl, yph = yp-ypu, yp+ypu
                
                ax.fill_between(x, ydl*s, ydh*s, color='tab:blue', alpha=.2)
                ax.fill_between(x, ypl*s, yph*s, color='tab:orange', alpha=.2)
                
                ax.plot(x, ydl*s, color='k', linewidth=.2)
                ax.plot(x, ydh*s, color='k', linewidth=.2)
                ax.plot(x, ypl*s, color='k', linewidth=.2)
                ax.plot(x, yph*s, color='k', linewidth=.2)
                
                ax.plot(x, yd*s, color='k', linewidth=2)
                ax.plot(x, yd*s, color='tab:blue', linewidth=1, label='Observations (L2)')
                ax.plot(x, yp*s, color='k', linewidth=2)
                ax.plot(x, yp*s, color='tab:orange', linewidth=1, label='Model')
        
        for i, ax in enumerate(axs[:, 0]):
            ax.set_ylabel('nT')
            ax.text(-.1, .5, f'Beam {i+1}', ha='center', va='center', rotation='vertical', transform=ax.transAxes)
        
        for i, ax in enumerate(axs[-1, :]):
            ax.set_xlabel('lat')
        
        for i, (ax, c) in enumerate(zip(axs[0, :], self.data.comp)):
            ax.text(.5, 1.1, f'B{c}', ha='center', va='center', transform=ax.transAxes)
        
        axs[0, 0].legend()
        plt.suptitle(self.data.date, y=.95)
        
        return fig, axs

#%%
    
    def pax_format(self):
        1+1
    
    def ax_format(self):
        1+1

    def plot_pp(self, ax, lat, lt, var, cmap, lvls):

        cc = ax.tricontourf(lat, lt, var, cmap=cmap, levels=lvls)
        
        return cc
        

    def plot_quiver_pp(self, ax, lat, lt, Je, Jn, scale):
        s = 8
        ax.quiver(lat.reshape(self.grid.xi_mesh.shape)[::s, ::s].flatten(),
                  lt.reshape(self.grid.xi_mesh.shape)[::s, ::s].flatten(),
                  Jn.reshape(self.grid.xi_mesh.shape)[::s, ::s].flatten(),
                  Je.reshape(self.grid.xi_mesh.shape)[::s, ::s].flatten(),
                  scale=scale, width=.001)

    def plot_quiver(self, ax, scale):
        s = 5
        ax.quiver(self.grid.xi_mesh[::s, ::s].flatten(),
                  self.grid.eta_mesh[::s, ::s].flatten(),
                  self.ev.Jxi.reshape(self.grid.xi_mesh.shape)[::s, ::s].flatten(),
                  self.ev.Jeta.reshape(self.grid.xi_mesh.shape)[::s, ::s].flatten(),
                  scale=scale, width=.003)
    
    def plot_map(self, ax, var, cmap, lvls):
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        cc = ax.tricontourf(self.grid.xi_mesh.flatten(),
                            self.grid.eta_mesh.flatten(),
                            var.flatten(),
                            cmap = cmap,
                            levels = lvls)
        self.plot_latlon(ax)
        self.plot_tracks(ax)
        return cc
    
    def plot_tracks(self, ax, lw=2):
        for i, mem in enumerate(self.data.mems):
            xi, eta = self.grid.projection.geo2cube(mem.lon, mem.lat)
            ax.plot(xi, eta, linewidth=lw+1, color='k')
            ax.plot(xi, eta, linewidth=lw, color=f'C{i}')
    
    def plot_latlon(self, ax):
        for la in range(0, 90, 5):
            xi, eta = self.grid.projection.geo2cube(np.linspace(0, 360, 1000), 
                                                          np.ones(1000)*la)
            ax.plot(xi, eta, linewidth=.5, color='k')
        
        for lo in range(0, 360, 15):
            xi, eta = self.grid.projection.geo2cube(np.ones(1000)*lo,
                                                          np.linspace(0, 90, 1000))
            ax.plot(xi, eta, linewidth=.5, color='k')
            
        ax.set_xlim(self.grid.xi_mesh.min(),  self.grid.xi_mesh.max())
        ax.set_ylim(self.grid.eta_mesh.min(), self.grid.eta_mesh.max())
    
    def plot_id(self, ax):
        ax.set_xticks([])
        ax.set_yticks([])
        self.plot_latlon(ax)
        self.plot_tracks(ax)        
        ax.plot(self.grid.xi.flatten()[self.regOpt.m_id],
                self.grid.eta.flatten()[self.regOpt.m_id],
                '*', color='k', markersize=12)
        ax.plot(self.grid.xi.flatten()[self.regOpt.m_id],
                self.grid.eta.flatten()[self.regOpt.m_id],
                '*', color='tab:red', markersize=10, label='Optimization zone')
        ax.legend(fontsize=12)
    
    def plot_L_curve(self, ax):
        ax.plot(self.regOpt.rnorm, self.regOpt.mnorm, 
                '.-', label='L-curve')
        ax.plot(self.regOpt.rnorm[self.regOpt.opt_id], 
                self.regOpt.mnorm[self.regOpt.opt_id], 
                '*', markersize=10, 
                label=f'l1, l2 = {np.round(self.regOpt.l1_opt, 2)}, {np.round(self.regOpt.l2_opt, 2)}')
        ax.plot(self.regOpt.rnorm[self.regOpt.offset], 
                self.regOpt.mnorm[self.regOpt.offset], 
                '*', markersize=10,
                label='Concave offset')
        ax.set_xlabel('misfit norm')
        ax.set_ylabel('model norm')
        ax.legend(fontsize=12)
        ax.plot(self.regOpt.rnorm_fit, self.regOpt.mnorm_fit)
    
    def plot_lambda_relation(self, ax):
        y, x = np.meshgrid(self.regOpt.l2s, self.regOpt.l1s)
        gini = np.copy(self.regOpt.gini_map)
        gini /= np.max(gini, axis=1, keepdims=True)
        ax.tricontourf(x.flatten(), y.flatten(), gini.flatten(), 
                       levels=40, cmap='magma')
        ax.plot(self.regOpt.l1s, self.regOpt.l2_ridge, 
                '.--', color='tab:blue', linewidth=1, zorder=10,
                label='Detected ridge')
        for i in range(self.regOpt.iterations):
            if i == 0:
                ax.plot(self.regOpt.l1s, self.regOpt.ridge_exp[:, i, 0], 
                        '.--', color='k', linewidth=.5, zorder=5,
                        label='Iterations')
            else:
                ax.plot(self.regOpt.l1s, self.regOpt.ridge_exp[:, i, 0], 
                        '.--', color='k', linewidth=.5, zorder=5)
        ax.plot(self.regOpt.l1_fit, self.regOpt.l2_fit, 
                '-', color='k', linewidth=3, zorder=11)
        ax.plot(self.regOpt.l1_fit, self.regOpt.l2_fit, 
                '-', color='tab:red', linewidth=2, zorder=11, 
                label='Spline fit')
        ax.legend(fontsize=12)
        ax.set_xlabel('l1')
        ax.set_ylabel('l2')
        
        
        
        
        
        
        
        
        
        
        
        