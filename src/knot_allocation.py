#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 14:06:26 2024

@author: naim769
"""
from scipy.interpolate import interp1d
import numpy as np
from typing import List

def binned_knots(
    data: np.ndarray,
    n_knots: int,
    data_bin_edges: List,
    f: List,
    data_bin_weights: List,
    log_data=False,
) -> np.ndarray:
    """Returns the knot vector"""

    # d = data.copy()
    # if log_data:
    #     d = np.log(data)

    N = len(data)
    
    if data_bin_weights is None:
        # weights w.r.t number of points
        mybins=np.concatenate(([min(f)], data_bin_edges, [max(f)]))
        bin_counts, _ = np.histogram(f, bins=mybins)
        data_bin_weights = bin_counts / np.sum(bin_counts)
        #equal weights
        # data_bin_weights = np.ones(len(data_bin_edges) + 1)
        # data_bin_weights = data_bin_weights / np.sum(data_bin_weights)
    if (len(data_bin_edges) + 1) != len(data_bin_weights):
        raise ValueError(
            "length of data_bin_edges is incorrect"
            f"data_bin_edges: {len(data_bin_edges)}, "
            f"data_bin_weights: {len(data_bin_weights)}"
        )

    # Knots placement based on log periodogram (Patricio code) This is when nfreqbin is an array
    

    data_bin_weights = data_bin_weights / np.sum(data_bin_weights)
    n_bin_weights = len(data_bin_weights)

    data_bin_edges = np.sort(data_bin_edges)
    # Transforming data_bin_edges to the interval [0,1] mx+c
    # make it different when including the data as x sth array
    m = 1 / (max(f) - min(f))
    c=min(f)
    data_bin_edges = m * (data_bin_edges-c)  
    eqval = np.concatenate(([0], data_bin_edges, [1]))  # Interval [0,1]
    eqval = np.column_stack(
        (eqval[:-1], eqval[1:])
    )  # Each row represents the bin
    j = np.linspace(0, 1, num=N)
    s = np.arange(1, N + 1)
    index = []

    for i in range(n_bin_weights):
        cond = (j >= eqval[i, 0]) & (j <= eqval[i, 1])
        if np.any(cond):  # Only append if there are valid elements
            index.append((np.min(s[cond]), np.max(s[cond])))
        else:
            print(f"No data points found in bin {i} with edges {eqval[i, 0]} and {eqval[i, 1]}")

    Nindex = len(index)

    n_knots = n_knots - 2  # to include 0 and 1 in the knot vector
    kvec = np.round(n_knots * np.array(data_bin_weights))
    kvec = kvec.astype(int)

    while np.sum(kvec) > n_knots:
        kvec[np.argmax(kvec)] = np.max(kvec) - 1

    while np.sum(kvec) < n_knots:
        kvec[np.argmin(kvec)] = np.min(kvec) + 1

    knots = []
    
    
    for i in range(Nindex):
        aux = data[index[i][0] : index[i][1]]
        if not log_data:
            aux = np.sqrt(aux) #in case using pdgrm
        dens = np.abs(aux - np.mean(aux)) / np.std(aux)

        Naux = len(aux)

        dens = dens / np.sum(dens)
        cumf = np.cumsum(dens)
        x = np.linspace(eqval[i][0], eqval[i][1], num=Naux)

        # Distribution function
        df = interp1d(x, cumf,  kind="linear", fill_value=(0, 1))
        dfvec = df(x)
        invDf = interp1d(
            dfvec,
            x,
            kind="linear",
            fill_value=(x[0], x[-1]),
            bounds_error=False,
        )
        v = np.linspace(0, 1, num=kvec[i] + 2)
        v = v[1:-1]
        knots = np.concatenate((knots, invDf(v)))

    knots = np.concatenate(([0], knots, [1]))

    return knots

def data_peak_knots(data: np.ndarray, n_knots: int) -> np.ndarray:#based on Patricio's pspline paper
    aux = np.sqrt(data)
    dens = np.abs(aux - np.mean(aux)) / np.std(aux)
    n = len(data)
    
    dens = dens / np.sum(dens)
    cumf = np.cumsum(dens)
    
    df = interp1d(
        np.linspace(0, 1, num=n), cumf, kind="linear", fill_value=(0, 1)
    )
    
    invDf = interp1d(
        df(np.linspace(0, 1, num=n)),
        np.linspace(0, 1, num=n),
        kind="linear",
        fill_value=(0, 1),
        bounds_error=False,
    )
    knots=invDf(np.linspace(0, 1, num=n_knots))
    unique_knots=np.unique(knots)
    while len(unique_knots) < n_knots:
         additional_knots = np.random.uniform(low=0, high=1, size=(n_knots - len(unique_knots)))
         unique_knots=sorted(set(unique_knots).union(set(additional_knots)))
         unique_knots = np.unique(unique_knots)
    return unique_knots

def knot_loc(
    data: np.ndarray,
    n_knots: int,
    f: List,
    data_bin_edges= None,
    data_bin_weights= None,
    log_data=False,
    equidistant=False,
) -> np.ndarray:
    if equidistant:
        return(np.linspace(min(f), max(f), num=n_knots))
    if data_bin_edges is None:
        return(data_peak_knots(data=data, n_knots=n_knots))
    else:
        return(binned_knots(data=data,n_knots=n_knots,data_bin_edges=data_bin_edges,f=f,data_bin_weights= data_bin_weights,log_data=log_data))
    
    