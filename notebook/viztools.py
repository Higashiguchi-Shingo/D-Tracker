import geopandas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse, sys
import glob
import folium
from mpl_toolkits.mplot3d import Axes3D   

sys.path.append("../src")
import utils

# default color map
def make_cmaps():
    return ["Blues", "Oranges", "Greens", "Reds", "Purples", "Greys"]

# gdfの国名コードをiso-a2に変換
# geo_listで指定された国の行だけを抽出してreturn
def load_gdf(geo_list):
    df = pd.read_csv("../data/all.csv") # 国の情報
    gdf = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))

    # gdfライブラリのエラーの修正
    gdf.loc[gdf['name'] == 'France', 'iso_a3'] = 'FRA'
    gdf.loc[gdf['name'] == 'Norway', 'iso_a3'] = 'NOR'

    # gdfの国名コードをiso-a2に変換
    for id, value in gdf.iterrows():
        if len(df[df["alpha-3"]==value["iso_a3"]]) != 0:
            gdf.loc[id, "iso_a3"] = df[df["alpha-3"]==value["iso_a3"]]["alpha-2"].values[0]

    gdf_geo = pd.DataFrame(columns=gdf.columns)

    # geo_listで指定された国の行だけを抽出
    for geo in sorted(geo_list):
        tmp = gdf[gdf["iso_a3"]==geo]
        gdf_geo = pd.concat([gdf_geo, tmp])
        
    return geopandas.GeoDataFrame(gdf_geo)

# --------------------------------------------#
# 地域のheatmapを作る
# --------------------------------------------#
# Wlocとidxを指定する
def loc_heatmap_column(Wloc, geo_list, idx_list, save_path=None, figsize=[10,5], legend=True):
    gdf = load_gdf(geo_list)
    cmaps = make_cmaps()
    for id in idx_list:
        gdf["value"] = Wloc.T[id]
        gdf.plot(column="value", legend=legend, figsize=figsize, cmap=cmaps[id])
        if save_path:
            plt.savefig(save_path + "/Wloc_map_" + str(id) + ".jpg")
        else:
            plt.show()

def loc_heatmap_columns(Wloc, geo_list, idx1, idx2, save_path=None, figsize=[10,5], legend=True, cmap="RdBu"):
    gdf = load_gdf(geo_list)
    cmaps = make_cmaps()
    gdf["value"] = Wloc.T[idx1] - Wloc.T[idx2]
    gdf.plot(column="value", figsize=figsize, cmap=cmap, legend=legend)
    if save_path:
        plt.savefig(save_path + "/Wloc_map.jpg")
    else:
        plt.show()

# --------------------------------------------#
# keyword factor のheatmapを作成
# --------------------------------------------#
# 1行ずつ作成する
def key_heatmap_column(Wkey, key_list, idx_list, save_path=None, linewidth=1, xticklabels=False, yticklabels=True):
    cmaps = make_cmaps()
    for id in idx_list:
        Wkey_df = pd.DataFrame(Wkey.T[id], index=key_list)
        sns.heatmap(Wkey_df, cmap=cmaps[id], xticklabels=xticklabels, yticklabels=yticklabels, square=True, linecolor="white", linewidths=linewidth)
        if save_path:
            plt.savefig(save_path + "/Wkey_heatmap_" + str(id) + ".jpg")
        else:
            plt.show()

# matrixで一気に作成
def key_heatmap(Wkey, key_list, save_path=None, linewidth=1, xticklabels=False, yticklabels=True, vmin=-10, vmax=9):
    Wkey_df = pd.DataFrame(Wkey, index=key_list)
    plt.figure(figsize=(10,5))
    ax = sns.heatmap(Wkey_df, cmap='bwr', square=True, linecolor="white", linewidths=linewidth, xticklabels=xticklabels, yticklabels=yticklabels, vmin=vmin, vmax=vmax)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=20)  # x軸ラベルの文字サイズ
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=20) 
    if save_path:
        plt.savefig(save_path + "/Wkey_heatmap.jpg")
    else:
        plt.show()

# --------------------------------------------#
# latent dynamicsを可視化
# --------------------------------------------#
def latent_dynamics(Wcore, figsize=(6,5), linewidth=1, save_path=None, label_num=[0,53,100], label_name=["2010-01","2011-01","2011-12"], color=None):
    lc = Wcore.shape[0]
    dk = Wcore.shape[1]
    dl = Wcore.shape[2]
    for id_l in range(dl):
        plt.figure(figsize=figsize)
        plt.grid(True, color="grey", linestyle="--", alpha=0.7)
        plt.plot(Wcore[:,:,id_l], linewidth=linewidth, color=color)
        plt.xticks(label_num, label_name, fontsize=18)
        plt.yticks(fontsize=18)
        plt.xlabel("Time", fontsize=18)
        plt.ylabel("Value", fontsize=18)
        if save_path:
            plt.savefig(save_path + "/Wcore_gen" + str(id_l) + ".jpg")
        else:
            plt.show()

# 拡散の図のためのmapを作成
def diff_map(Wloc, geo_list, drop_cols=None):
    df = pd.DataFrame(Wloc, columns=[str(i) for i in range(Wloc.shape[1])], index=geo_list)

    gdf = load_gdf(geo_list)
    gdf = gdf.set_crs("EPSG:4326")

    if drop_cols:
        df = df.drop(drop_cols, axis=1)
    df["group"] = df.idxmax(axis=1) # それぞれのfactorの重みを比較して最も値が大きいところにgroupを割り振る

    m = folium.Map(location=[0, 0], zoom_start=2)

    for idx, value in df[df["group"]=="0"].iterrows():
        folium.GeoJson(gdf[gdf['iso_a3']==idx], 
                style_function=lambda x: {'fillColor': 'blue', "color":"blue", "weight":0.3}
                ).add_to(m)

    for idx, value in df[df["group"]=="1"].iterrows():
        folium.GeoJson(gdf[gdf['iso_a3']==idx], 
                style_function=lambda x: {'fillColor': 'orange', "color":"orange", "weight":1}
                ).add_to(m)
        
    for idx, value in df[df["group"]=="2"].iterrows():
        folium.GeoJson(gdf[gdf['iso_a3']==idx], 
                style_function=lambda x: {'fillColor': "green", "color":"green", "weight":1}
                ).add_to(m)
    
    for idx, value in df[df["group"]=="3"].iterrows():
        folium.GeoJson(gdf[gdf['iso_a3']==idx], 
                style_function=lambda x: {'fillColor': "red", "color":"red", "weight":1}
                ).add_to(m)
    
    for idx, value in df[df["group"]=="4"].iterrows():
        folium.GeoJson(gdf[gdf['iso_a3']==idx], 
                style_function=lambda x: {'fillColor': "purple", "color":"purple", "weight":1}
                ).add_to(m)
    
    return m

def make_3D_dynamics(Wcore):
    # (x, y, z)
    x = np.arange(104)
    y = np.zeros(104)
    z = Wcore[:,0,0]

    # 3Dでプロット
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')

    y = y + 0.05
    z = Wcore[:,0,1]
    ax.plot(x, y, z, color="royalblue")

    z = Wcore[:,1,1]
    ax.plot(x, y, z, color="darkorange")

    z = Wcore[:,2,1]
    ax.plot(x, y, z, color="limegreen")

    y = y - 0.1
    z = Wcore[:,0,0]
    ax.plot(x, y, z, color="royalblue")

    z = Wcore[:,1,0]
    ax.plot(x, y, z, color="darkorange")

    z = Wcore[:,2,0]
    ax.plot(x, y, z, color="limegreen", linewidth=3)

    ax.set_ylim(-0.1,0.1)

    # 軸ラベル
    #ax.set_xlabel('Time')
    #ax.set_ylabel('')
    #ax.set_zlabel('z')
    ax.set_zticks([])
    ax.set_yticks([])
    # 表示
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    ax.axes.zaxis.set_visible(False)
    #ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    #ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    #ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    plt.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
    plt.show()