# import numpy as np
# from numpy.typing import NDArray

# from ..map import MapGeometry

# def background_mask(tod, pixels, valid, geometry: MapGeometry, *, beam_fwhm_pix=3.0, n_sigma=3.0, dilate_beams=1.0):
#     # tod, pixels, valid: (n_samples, n_det) come in make_map
#     # return: (n_samples, n_det) bool, True = campione lontano dalla sorgente (da usare per il rumore)
#     # 1. mappa grezza: np.bincount(pixels[valid], weights=tod[valid], minlength=n_pixels) / hits,
#     #    reshape (ny, nx); i pixel con hits == 0 non sono osservati (esclusi da tutto)
#     # 2. smoothing con gaussiana larga quanto il beam (scipy.ndimage.gaussian_filter,
#     #    sigma = beam_fwhm_pix / 2.355); con pixel non osservati dividi per lo smoothing
#     #    della loro maschera, altrimenti il bordo si abbassa
#     # 3. med = mediana della mappa smussata, sigma = 1.4826 * MAD, solo sui pixel osservati
#     # 4. mask_map = |m - med| > n_sigma * sigma   (simmetrica)
#     # 5. scipy.ndimage.binary_dilation(mask_map, iterations=round(dilate_beams * beam_fwhm_pix))
#     # 6. torna al TOD: source = mask_map.ravel()[pixels]; dove valid è False pixels può valere
#     #    qualsiasi cosa, quindi usa np.where(valid, pixels, 0) per indicizzare e poi source &= valid
#     # 7. stampa la frazione di campioni mascherati (se è molta, la soglia è troppo bassa)
#     # return ~source

#     binned_map = np.bincount(pixels[valid], tod[valid], minlength=geometry.n_pixels) / hits
