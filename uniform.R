uniformmax=function (sample) 
{
  max(abs(sample - stats::median(sample))/stats::mad(sample), 
      na.rm = TRUE)
}