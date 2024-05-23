psd_arma <- function(freq, ar, ma, sigma2) {
  
  # Analytical spectral density for ARMA(p,q) model
  # Assumes no intercept terms - i.e., mean centered
  
  # freq: frequencies defined on [0, pi]
  # ar: AR coefficients of length p - must be named
  # ma: MA coefficients of length q - must be named
  # sigma2: Variance of white noise
  
  # MA component (numerator)
  if (any(is.na(ma))) {
    numerator <- rep(1, length(freq))
  }
  else {
    numerator <- matrix(NA, ncol = length(ma), nrow = length(freq))
    for (j in 1:length(ma)) {
      numerator[, j] <- ma[j] * exp(-1i * j * freq)  # CAUTION!!!
    }
    numerator <- Mod(1 + apply(numerator, 1, sum)) ^ 2  # Note the PLUS
  }
  
  # AR component (denominator)
  if (any(is.na(ar))) {
    denominator <- rep(1, length(freq))
  }
  else {
    denominator <- matrix(NA, ncol = length(ar), nrow = length(freq))
    for (j in 1:length(ar)) {
      denominator[, j] <- ar[j] * exp(-1i * j * freq)  # CAUTION!!!
    }
    denominator <- Mod(1 - apply(denominator, 1, sum)) ^ 2  # Note the MINUS
  }
  
  psd <- (sigma2 / (2 * pi)) * (numerator / denominator)
  
  return(psd)
  
}