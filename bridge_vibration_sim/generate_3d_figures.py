"""Corrected 3D figure generation — consistent with the report's quantities."""
from __future__ import annotations
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from bridge_sim import (Bridge, Section, STEEL, modal_analysis, newmark_beta,
                        CAR, TRUCK, HEAVY_TRUCK, moving_load, traffic_stream,
                        Earthquake, combine, rayleigh_coefficients, modal_damping_ratio)
from bridge_sim.loads import Vehicle

OUT = "figures3d"; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titlesize": 13,
                     "axes.labelsize": 11, "figure.dpi": 140, "savefig.dpi": 175})
INK="#0e1b2b"; ACC="#22d3ee"; AMBER="#fbbf24"; RED="#f87171"; GRN="#34d399"

b, h, t = 4.0, 2.0, 0.025
sec = Section.box_girder(b, h, t, name=f"Caisson {b}x{h} m")
bridge = Bridge(length=60.0, n_elements=60, material=STEEL, section=sec)
E, rho = STEEL.E, STEEL.rho; A, I = sec.A, sec.I; L = bridge.length; m_bar = rho*A
modes = modal_analysis(bridge, n_modes=6); f = modes.natural_frequencies_hz; w = modes.angular_frequencies
n = np.arange(1,7); f_analytic = (n**2*np.pi/2)*np.sqrt(E*I/(m_bar*L**4))
zeta = 0.02; alpha, beta = rayleigh_coefficients(w[0], w[2], zeta, zeta)
zeta_modes = np.array([modal_damping_ratio(wi, alpha, beta) for wi in w])
dt = 0.005; gamma, beta_nm = 0.5, 0.25
a_nm = [1/(beta_nm*dt*dt), gamma/(beta_nm*dt), 1/(beta_nm*dt), 1/(2*beta_nm)-1,
        gamma/beta_nm-1, dt*(gamma/(2*beta_nm)-1), dt*(1-gamma), dt*gamma]
gN, c = 9.81, h/2
def static_central(P): return P*L**3/(48*E*I), P*L/4
static = {}
for name, mkg in {"Voiture (1,5 t)":1500, "Camion (32 t)":32000, "Lourd articulé (44 t)":44000}.items():
    P = mkg*gN; d, M = static_central(P)
    static[name] = dict(P_kN=P/1e3, delta_mm=d*1e3, M_kNm=M/1e3, sigma_MPa=M*c/I/1e6)
M_cap = 355e6*I/c

vdof = bridge.nodal_vertical_dofs(); x = bridge.node_x; mid = bridge.midspan_node()
def run(F, dur): return newmark_beta(bridge, F, duration_s=dur, dt=0.005, zeta_1=zeta, zeta_2=zeta)
def midmm(r): return r.u[2*mid, :]*1e3

res_truck = run(moving_load(bridge, HEAVY_TRUCK), L/HEAVY_TRUCK.speed_mps+1)
mid_truck = midmm(res_truck); truck_peak = float(np.max(np.abs(mid_truck)))
ti = int(np.argmax(np.abs(mid_truck))); w_truck = res_truck.u[vdof, ti]*1e3
truck_x = HEAVY_TRUCK.speed_mps*res_truck.t[ti]
eq = Earthquake(bridge, duration_s=20, dt=0.005, pga_g=0.35, dominant_frequency_hz=3.0, seed=7)
stream = traffic_stream(bridge, [Vehicle(mass_kg=44000, speed_mps=13.9, entry_time_s=0.5),
                                 Vehicle(mass_kg=32000, speed_mps=16.7, entry_time_s=3.0),
                                 Vehicle(mass_kg=18000, speed_mps=16.7, entry_time_s=6.0)])
res_comb = run(combine(stream, eq), 20); mid_comb = midmm(res_comb); comb_peak = float(np.max(np.abs(mid_comb)))
res_car = run(moving_load(bridge, CAR), L/CAR.speed_mps+1); mid_car = midmm(res_car)
res_t32 = run(moving_load(bridge, TRUCK), L/TRUCK.speed_mps+1); mid_t32 = midmm(res_t32)
print("f", np.round(f,2), "| peaks car",round(float(np.max(np.abs(mid_car))),1),
      "t32",round(float(np.max(np.abs(mid_t32))),1),"heavy",round(truck_peak,1),"comb",round(comb_peak,1))

def style3d(ax):
    ax.set_facecolor("white")
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_facecolor((0.965,0.975,0.99,1)); a.pane.set_edgecolor((0.80,0.84,0.90,1))
        a._axinfo["grid"]["color"]=(0.86,0.89,0.94,1); a._axinfo["grid"]["linewidth"]=0.6

def cuboid(ax, ctr, size, color, alpha=1, ec="none", lw=0.4):
    cx,cy,cz=ctr; dx,dy,dz=size; X=[cx-dx/2,cx+dx/2]; Y=[cy-dy/2,cy+dy/2]; Z=[cz-dz/2,cz+dz/2]
    v=np.array([[X[i],Y[j],Z[k]] for i in(0,1) for j in(0,1) for k in(0,1)])
    fc=[[v[0],v[1],v[3],v[2]],[v[4],v[5],v[7],v[6]],[v[0],v[1],v[5],v[4]],
        [v[2],v[3],v[7],v[6]],[v[0],v[2],v[6],v[4]],[v[1],v[3],v[7],v[5]]]
    ax.add_collection3d(Poly3DCollection(fc,facecolor=color,edgecolor=ec,linewidths=lw,alpha=alpha))

def deck(ax, xs, w_mm, width, deck_h, scale, cmap, norm, top_color=None, ref=True):
    zt=w_mm*scale; ny=10; Y=np.linspace(-width/2,width/2,ny); Xg,Yg=np.meshgrid(xs,Y); Zt=np.tile(zt,(ny,1))
    if top_color is None:
        fc=cmap(norm(np.tile(np.abs(w_mm),(ny,1))))
        ax.plot_surface(Xg,Yg,Zt,facecolors=fc,rstride=1,cstride=1,linewidth=0,antialiased=True,shade=False)
    else:
        ax.plot_surface(Xg,Yg,Zt,color=top_color,rstride=1,cstride=2,linewidth=0,antialiased=True,shade=True)
    ax.plot_surface(Xg,Yg,Zt-deck_h,color="#9fb0c3",rstride=2,cstride=2,linewidth=0,antialiased=True,shade=False)
    for yk in (-width/2, width/2):
        Xs=np.vstack([xs,xs]); Ys=np.full_like(Xs,yk); Zw=np.vstack([zt-deck_h,zt])
        ax.plot_surface(Xs,Ys,Zw,color="#b9c6d6",linewidth=0,antialiased=True,shade=False)
    if ref: ax.plot(xs,np.zeros_like(xs),np.zeros_like(xs),color="#9aa7b6",lw=0.8,ls=(0,(4,3)))

def supports(ax, width, ztop, hp):
    for xp in (0.0, L): cuboid(ax,(xp,0,ztop-hp/2),(1.8,width*1.2,hp),"#5a6b7e",ec="#3c4a5a",lw=0.5)

# A: model
fig=plt.figure(figsize=(11,5.0)); ax=fig.add_subplot(111,projection="3d"); style3d(ax)
width=4.0; xs=np.linspace(0,L,60)
deck(ax,xs,np.zeros_like(xs),width,deck_h=2.0,scale=1.0,cmap=None,norm=None,top_color="#7f93a8",ref=False)
ax.plot(xs,np.zeros_like(xs),np.zeros_like(xs)+0.03,color=AMBER,lw=1.8,ls=(0,(9,9)))
supports(ax,width,ztop=-2.0,hp=9)
ax.plot([0,L],[0,0],[-12,-12],color=INK,lw=1.4)
ax.text(L/2,0,-13.6,"Travée L = 60 m",ha="center",fontsize=11,color=INK,weight="bold")
ax.text(0,0,-2.0,"Appui\nsimple",ha="right",fontsize=8.5,color="#475569")
ax.text(L,0,-2.0,"Appui à\nrouleau",ha="left",fontsize=8.5,color="#475569")
ax.text(L*0.5,width*1.5,1.0,"Section caisson acier 4,0 × 2,0 m  (I = 0,226 m⁴)",fontsize=9,color="#334155")
ax.set_xlabel("x  [m]"); ax.set_ylabel("largeur [m]"); ax.set_zlabel("z [m]")
ax.set_box_aspect((6,0.9,1.5)); ax.set_zlim(-15,5); ax.set_yticks([-2,0,2]); ax.set_xticks([0,20,40,60])
ax.view_init(elev=20,azim=-60)
ax.set_title("Modèle 3D du pont — caisson acier sur appuis simples (travée 60 m)",weight="bold",pad=0)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_model.png",bbox_inches="tight",facecolor="white"); plt.close(fig)

# B: deformed under 44t
fig=plt.figure(figsize=(11,5.2)); ax=fig.add_subplot(111,projection="3d"); style3d(ax)
width=9.0; scale=6.0/max(truck_peak,1e-6); amp=scale*1000
norm=Normalize(0,np.max(np.abs(w_truck))); cmap=cm.turbo
deck(ax,x,w_truck,width,deck_h=1.4,scale=scale,cmap=cmap,norm=norm)
supports(ax,width,ztop=0.0,hp=7)
zt_at=np.interp(truck_x,x,w_truck)*scale
cuboid(ax,(truck_x,0,zt_at+1.7),(12,2.8,2.8),"#1f2937",ec="#0b1220")
cuboid(ax,(truck_x-4.5,0,zt_at+1.2),(3.2,2.6,1.8),"#374151",ec="#0b1220")
m=cm.ScalarMappable(norm=norm,cmap=cmap); m.set_array([])
cb=fig.colorbar(m,ax=ax,shrink=0.5,pad=0.10,aspect=16); cb.set_label("|flèche|  [mm]",fontsize=10)
ax.set_xlabel("x  [m]"); ax.set_ylabel("largeur [m]"); ax.set_zlabel("")
ax.set_box_aspect((6,1.3,1.4)); ax.set_zticks([]); ax.set_yticks([-4,0,4]); ax.set_xticks([0,20,40,60])
ax.view_init(elev=18,azim=-60)
ax.set_title(f"Réponse dynamique 3D du tablier sous lourd articulé 44 t  "
             f"(déformation amplifiée ×{amp:.0f})",weight="bold",pad=0)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_truck3d.png",bbox_inches="tight",facecolor="white"); plt.close(fig)

# C: mode shapes
fig=plt.figure(figsize=(11,7.2)); width=7.0; subs=["₁","₂","₃","₄"]
for k in range(4):
    ax=fig.add_subplot(2,2,k+1,projection="3d"); style3d(ax)
    shp=modes.mode_shapes[vdof,k]; shp=shp/np.max(np.abs(shp)); ny=8
    Y=np.linspace(-width/2,width/2,ny); Xg,Yg=np.meshgrid(x,Y)
    Z=np.tile(shp*5.0,(ny,1)); fc=cm.coolwarm(Normalize(-1,1)(np.tile(shp,(ny,1))))
    ax.plot_surface(Xg,Yg,Z,facecolors=fc,rstride=1,cstride=2,linewidth=0,antialiased=True,shade=False)
    ax.plot(x,np.zeros_like(x),np.zeros_like(x),color="#9aa7b6",lw=0.7,ls=(0,(4,3)))
    ax.set_box_aspect((6,1.2,1.4)); ax.set_zlim(-6,6); ax.set_xticks([0,30,60])
    ax.set_xlabel("x [m]",labelpad=4,fontsize=9); ax.set_yticks([]); ax.set_zticks([])
    ax.tick_params(axis="x",labelsize=8,pad=1); ax.view_init(elev=22,azim=-62)
    ax.set_title(f"Mode {k+1} — f{subs[k]} = {f[k]:.2f} Hz",weight="bold",fontsize=11,pad=-2)
fig.suptitle("Déformées modales du pont (analyse modale — 4 premiers modes)",weight="bold",fontsize=13.5,y=1.0)
fig.tight_layout(rect=(0,0,1,0.97)); fig.savefig(f"{OUT}/fig_modes3d.png",bbox_inches="tight",facecolor="white"); plt.close(fig)

# D: spatio-temporal
fig=plt.figure(figsize=(11,6.0)); ax=fig.add_subplot(111,projection="3d"); style3d(ax)
ts=res_comb.t; st=max(1,len(ts)//120); Tg,Xg=np.meshgrid(ts[::st],x); W=res_comb.u[vdof][:,::st]*1e3
surf=ax.plot_surface(Xg,Tg,W,cmap="turbo",norm=Normalize(np.min(W),np.max(W)),rstride=1,cstride=1,linewidth=0,antialiased=True)
cb=fig.colorbar(surf,ax=ax,shrink=0.5,pad=0.11,aspect=16); cb.set_label("flèche w  [mm]",fontsize=10)
ax.set_xlabel("position x  [m]"); ax.set_ylabel("temps t  [s]"); ax.set_zlabel("w [mm]",labelpad=2)
ax.set_box_aspect((5,4,2.2)); ax.view_init(elev=26,azim=-58)
ax.set_title("Réponse spatio-temporelle w(x,t) — séisme 0,35 g + trafic",weight="bold",pad=0)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_surface3d.png",bbox_inches="tight",facecolor="white"); plt.close(fig)

# E: scenario ribbons
fig=plt.figure(figsize=(11,5.8)); ax=fig.add_subplot(111,projection="3d"); style3d(ax)
ribbons=[("Voiture 1,5 t",res_car.t,mid_car,ACC),("Camion 32 t",res_t32.t,mid_t32,GRN),
         ("Lourd 44 t",res_truck.t,mid_truck,AMBER),("Séisme 0,35 g + trafic",res_comb.t,mid_comb,RED)]
for i,(lab,tt,hh,col) in enumerate(ribbons):
    y=np.full_like(tt,i); ax.plot(tt,y,hh,color=col,lw=2.0)
    verts=[list(zip(tt,y,hh))+list(zip(tt[::-1],y[::-1],np.zeros_like(hh)))]
    ax.add_collection3d(Poly3DCollection(verts,facecolor=col,alpha=0.15))
handles=[Line2D([0],[0],color=col,lw=3,label=lab) for lab,_,_,col in ribbons]
ax.legend(handles=handles,loc="upper left",bbox_to_anchor=(0.0,0.92),fontsize=9,framealpha=0.92)
ax.set_xlabel("temps t  [s]"); ax.set_ylabel(""); ax.set_zlabel("flèche mi-portée [mm]",labelpad=2)
ax.set_yticks(range(len(ribbons))); ax.set_yticklabels([])
ax.set_box_aspect((5,3,2.2)); ax.view_init(elev=20,azim=-62)
ax.set_title("Histoire temporelle de la flèche à mi-portée — comparaison des scénarios",weight="bold",pad=0)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_scenarios3d.png",bbox_inches="tight",facecolor="white"); plt.close(fig)

calc = dict(geom=dict(b=b,h=h,t=t,A=A,I=I,L=L,n_el=bridge.n_elements),
    mat=dict(E=E,rho=rho,sigma_y=355e6), m_bar=m_bar,
    f_fem=f.tolist(), f_analytic=f_analytic.tolist(), omega=w.tolist(),
    rayleigh=dict(alpha=alpha,beta=beta,zeta_target=zeta,modes_used=[1,3],zeta_modes=zeta_modes.tolist()),
    newmark=dict(dt=dt,gamma=gamma,beta=beta_nm,a=a_nm),
    static=static, M_cap_kNm=M_cap/1e3,
    dyn=dict(car_mm=float(np.max(np.abs(mid_car))), t32_mm=float(np.max(np.abs(mid_t32))),
             heavy_mm=truck_peak, comb_mm=comb_peak,
             DAF_heavy=truck_peak/static["Lourd articulé (44 t)"]["delta_mm"]))
with open("_calcs.json","w") as fp: json.dump(calc, fp, indent=2)
print("done")
