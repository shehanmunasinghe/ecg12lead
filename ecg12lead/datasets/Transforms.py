# import pywt
# from skimage.restoration import denoise_wavelet

# def denoise(data,threshold = 0.04,maxlev=None, wavelet_fn='sym4'):
#     ## threshold = Threshold for filtering
#     w = pywt.Wavelet(wavelet_fn)
#     N = data.shape[0]

#     if not maxlev:
#         maxlev = pywt.dwt_max_level(N, w.dec_len)
#     coeffs = pywt.wavedec(data, wavelet_fn, level=maxlev)
#     #cA = pywt.threshold(cA, threshold*max(cA))
#     # plt.figure()
#     for i in range(1, len(coeffs)):
#         # plt.subplot(maxlev, 1, i)
#         # plt.plot(coeffs[i])
#         coeffs[i] = pywt.threshold(coeffs[i], threshold*max(coeffs[i]))
#         # plt.plot(coeffs[i])

#     return pywt.waverec(coeffs, wavelet_fn)